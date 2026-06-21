#!/usr/bin/env python3
"""Hook PostToolUse : signale les skills périmés quand le code de leur module change.

Pipeline :
  1. Lit `CLAUDE.md` et en extrait dynamiquement le mapping dossier → skill via les
     lignes « → détails : .claude/skills/<nom>.md » (aucun mapping codé en dur).
  2. Associe chaque nom de skill au dossier homonyme du dépôt (ex. `systems` →
     `src/ats_system/systems`, `scripts` → `scripts`).
  3. Détecte les fichiers modifiés via `git diff --name-only HEAD`.
  4. Si un fichier modifié appartient à un module mappé ET a été touché après la
     dernière mise à jour de son skill (comparaison de mtime, pour ne pas re-signaler
     un skill déjà resynchronisé), écrit `.claude/.stale_skills` puis sort en erreur.

Le code de sortie est `2` : pour un hook PostToolUse, c'est le seul code dont la sortie
d'erreur (stderr) est réinjectée à Claude. Un `exit(1)` ne ferait qu'afficher le message
à l'utilisateur sans déclencher la resynchronisation par Claude (cf. CLAUDE.md §
« Maintenance automatique »).

Autonome : stdlib uniquement (+ la commande `git`).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# Dossiers à ignorer lors de la recherche du dossier homonyme d'un skill.
IGNORE_DIRS = {".git", "ats_syst", ".venv", "venv", "node_modules", "__pycache__", ".claude"}

# Capture le nom de skill dans « → détails : `.claude/skills/<nom>.md` ».
DETAIL_RE = re.compile(r"d[ée]tails\s*:\s*`?\.claude/skills/([A-Za-z0-9_-]+)\.md`?")


def repo_root() -> Path:
    """Racine du dépôt git (repli sur le cwd si git indisponible)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except Exception:
        return Path.cwd()


def parse_skill_names(claude_md: Path) -> list[str]:
    """Extrait les noms de skills référencés dans CLAUDE.md (sans extension)."""
    if not claude_md.exists():
        return []
    names: list[str] = []
    for line in claude_md.read_text(encoding="utf-8").splitlines():
        match = DETAIL_RE.search(line)
        if match and match.group(1) not in names:
            names.append(match.group(1))
    return names


def find_module_dirs(root: Path, names: list[str]) -> dict[str, Path]:
    """Associe chaque nom de skill au dossier homonyme présent dans le dépôt."""
    wanted = set(names)
    mapping: dict[str, Path] = {}
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for d in dirnames:
            if d in wanted and d not in mapping:
                mapping[d] = Path(dirpath) / d
    return mapping


def changed_files(root: Path) -> list[Path]:
    """Fichiers modifiés (suivis) vs HEAD, en chemins absolus."""
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        )
    except Exception:
        return []
    return [root / line.strip() for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    # Consomme l'éventuel payload JSON du hook sur stdin (évite un broken pipe).
    try:
        sys.stdin.read()
    except Exception:
        pass

    root = repo_root()
    skills_dir = root / ".claude" / "skills"
    stale_file = root / ".claude" / ".stale_skills"

    skill_names = parse_skill_names(root / "CLAUDE.md")
    module_dirs = find_module_dirs(root, skill_names)
    if not module_dirs:
        return 0

    stale: set[str] = set()
    for changed in changed_files(root):
        for name, module_dir in module_dirs.items():
            try:
                changed.resolve().relative_to(module_dir.resolve())
            except (ValueError, OSError):
                continue
            # Skill considéré périmé seulement si le fichier de code a été modifié
            # APRÈS la dernière mise à jour du skill (sinon il vient d'être resynchronisé).
            skill_path = skills_dir / f"{name}.md"
            skill_mtime = skill_path.stat().st_mtime if skill_path.exists() else 0.0
            try:
                code_mtime = changed.stat().st_mtime
            except OSError:
                continue
            if code_mtime > skill_mtime:
                stale.add(name)

    if not stale:
        # Plus rien de périmé : on nettoie un éventuel marqueur résiduel.
        if stale_file.exists():
            stale_file.unlink()
        return 0

    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("\n".join(sorted(stale)) + "\n", encoding="utf-8")

    skills_list = ", ".join(f".claude/skills/{n}.md" for n in sorted(stale))
    print(
        f"[sync_skills] Code modifié dans un module documenté — skill(s) périmé(s) : {skills_list}.\n"
        f"Avant de continuer : relis le code modifié du/des module(s), mets à jour ce(s) "
        f"skill(s) pour refléter les vrais changements, puis supprime .claude/.stale_skills.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
