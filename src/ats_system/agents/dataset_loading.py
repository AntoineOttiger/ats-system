"""Chargement d'un dataset synthétique de CVs (manifest + textes) pour les optimiseurs.

Mutualise la lecture du ``manifest.json`` d'un dataset ``synthetic_cvs_<...>/`` entre les
optimiseurs de CV (agent ReAct itératif et optimiseur one-shot) : repérage du CV « à
optimiser » (entrée ``optimize: true``) et chargement des CVs concurrents.
"""

import json
from pathlib import Path

from ats_system.data import import_pdf


def find_optimize_cv(dataset_dir: Path) -> tuple[str, str]:
    """Localise le CV « à optimiser » (entrée ``optimize: true``) → ``(nom_fichier, texte)``.

    Args:
        dataset_dir: Dossier d'un dataset synthétique contenant un ``manifest.json``.

    Returns:
        Le couple ``(nom_fichier, texte)`` du CV marqué ``optimize: true``.

    Raises:
        ValueError: Si aucun CV « à optimiser » n'est présent dans le manifest.
    """
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["cvs"]:
        if entry.get("optimize"):
            cv_path = dataset_dir / entry["file"]
            return entry["file"], import_pdf(str(cv_path))["content"]
    raise ValueError(f"Aucun CV « à optimiser » (optimize: true) dans {dataset_dir}.")


def load_competitors(dataset_dir: Path) -> list[dict]:
    """Lit les CVs concurrents du dataset (dicts ``{"id", "content"}``).

    Le CV « à optimiser » (entrée ``optimize: true`` du manifest) est exclu : c'est le
    candidat dont l'optimiseur fait justement varier le rang.

    Args:
        dataset_dir: Dossier d'un dataset synthétique contenant un ``manifest.json``.

    Returns:
        La liste des CVs concurrents, dicts ``{"id", "content"}`` (cf. ``import_pdf``).
    """
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))

    competitors: list[dict] = []
    for entry in manifest["cvs"]:
        if entry.get("optimize"):
            continue
        cv_path = dataset_dir / entry["file"]
        competitors.append({"id": entry["file"], "content": import_pdf(str(cv_path))["content"]})
    return competitors
