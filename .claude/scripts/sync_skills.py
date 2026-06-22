#!/usr/bin/env python3
"""Hook PostToolUse : signale (sans bloquer) un skill périmé quand le code de son module change.

Pipeline :
  1. Lit le JSON du hook sur stdin et récupère le fichier réellement édité
     (`tool_input.file_path`). Pas de `file_path` (ex. Bash) → on ne fait rien.
  2. Lit `CLAUDE.md` et en extrait le mapping dossier → skill via les lignes
     « → détails : .claude/skills/<nom>.md » (aucun mapping codé en dur).
  3. Associe chaque nom de skill au dossier homonyme du dépôt (ex. `systems` →
     `src/ats_system/systems`).
  4. Si le fichier édité appartient à un module mappé ET a été touché après la dernière
     mise à jour de son skill (comparaison de mtime, pour ne pas re-signaler un skill
     déjà resynchronisé) ET n'est pas déjà marqué périmé, l'ajoute à `.claude/.stale_skills`
     et émet un rappel **non bloquant** (JSON `additionalContext`).

Conçu pour minimiser la consommation de tokens : raisonne sur le seul fichier édité (et
non `git diff HEAD`), déduplique via `.stale_skills` (un skill n'est signalé qu'une fois),
et sort toujours en `0` — le rappel passe par `hookSpecificOutput.additionalContext`, ce
qui le rend visible à Claude sans interrompre sa tâche.

Autonome : stdlib uniquement.
"""
from __future__ import annotations

import json
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


def edited_file(payload: dict) -> Path | None:
    """Chemin du fichier réellement modifié par l'outil, ou None (ex. Bash sans fichier)."""
    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path:
        return None
    return Path(file_path)


def skill_for_file(target: Path, module_dirs: dict[str, Path]) -> str | None:
    """Nom du skill dont le module contient `target`, ou None."""
    try:
        resolved = target.resolve()
    except OSError:
        return None
    for name, module_dir in module_dirs.items():
        try:
            resolved.relative_to(module_dir.resolve())
            return name
        except (ValueError, OSError):
            continue
    return None


def read_stale(stale_file: Path) -> set[str]:
    """Skills déjà marqués périmés."""
    if not stale_file.exists():
        return set()
    return {line.strip() for line in stale_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    # Lit le payload JSON du hook sur stdin (évite aussi un broken pipe côté appelant).
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    target = edited_file(payload)
    if target is None:
        return 0  # Pas de fichier édité (Bash, etc.) : rien à faire.

    root = repo_root()
    skills_dir = root / ".claude" / "skills"
    stale_file = root / ".claude" / ".stale_skills"

    skill_names = parse_skill_names(root / "CLAUDE.md")
    module_dirs = find_module_dirs(root, skill_names)
    if not module_dirs:
        return 0

    name = skill_for_file(target, module_dirs)
    if name is None:
        return 0  # Fichier hors d'un module documenté.

    already = read_stale(stale_file)
    if name in already:
        return 0  # Déjà signalé : on ne re-signale pas (déduplication).

    # Skill considéré périmé seulement si le code a été modifié APRÈS la dernière mise à
    # jour du skill (sinon il vient d'être resynchronisé).
    skill_path = skills_dir / f"{name}.md"
    skill_mtime = skill_path.stat().st_mtime if skill_path.exists() else 0.0
    try:
        code_mtime = target.stat().st_mtime
    except OSError:
        return 0
    if code_mtime <= skill_mtime:
        return 0

    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("\n".join(sorted(already | {name})) + "\n", encoding="utf-8")

    message = (
        f"[sync_skills] Code modifié dans le module « {name} » — skill .claude/skills/{name}.md "
        f"potentiellement périmé. À l'occasion : mets-le à jour pour refléter les vrais "
        f"changements, puis supprime .claude/.stale_skills. (Rappel non bloquant.)"
    )
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": message,
    }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
