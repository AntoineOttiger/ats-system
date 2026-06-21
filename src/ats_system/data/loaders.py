"""Chargement des données par défaut : annonce et CVs d'une catégorie.

Mutualise le boilerplate jusqu'ici dupliqué dans les scripts (glob ``*.pdf`` +
``import_pdf``), afin que les méthodes ``run()`` des systèmes chargent leurs données
en une ligne.
"""

from pathlib import Path
from typing import Optional

from ats_system.config import CV_DIR, DEFAULT_ANNOUNCEMENT, DEFAULT_CV_CATEGORY
from ats_system.data.pdf_loader import import_pdf


def load_announcement(announcement: Path = DEFAULT_ANNOUNCEMENT) -> dict:
    """Charge une annonce PDF → dict ``{"id", "content"}`` (cf. ``import_pdf``)."""
    return import_pdf(str(announcement))


def load_cvs(category: str = DEFAULT_CV_CATEGORY, limit: Optional[int] = None) -> list[dict]:
    """Charge les CVs PDF d'une catégorie (triés par nom de fichier).

    Args:
        category: Sous-dossier de ``CV_DIR`` (ex. ``"ENGINEERING"``).
        limit:    Nombre maximum de CVs à charger. ``None`` ou ``<= 0`` = tous.

    Returns:
        Liste de dicts ``{"id", "content"}`` (cf. ``import_pdf``).
    """
    cv_files = sorted((CV_DIR / category).glob("*.pdf"))
    if limit is not None and limit > 0:
        cv_files = cv_files[:limit]
    return [import_pdf(str(cv_path)) for cv_path in cv_files]
