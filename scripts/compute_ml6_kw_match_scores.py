"""Classe les CVs face à l'annonce via les mots-clés ml6team.

Toute la logique (chargement, scoring, sauvegarde) vit dans ``Ml6KeywordMatcher.run()``.
"""

import argparse
from pathlib import Path

from ats_system.config import DEFAULT_ANNOUNCEMENT, DEFAULT_CV_DIR
from ats_system.systems import Ml6KeywordMatcher


def main():
    parser = argparse.ArgumentParser(description="Calcule le ml6_kw_match_score des CVs.")
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument(
        "--announcement", type=str, default=str(DEFAULT_ANNOUNCEMENT),
        help="Chemin du PDF de l'annonce (défaut : annonce par défaut du projet)",
    )
    parser.add_argument(
        "--cv-dir", type=str, default=str(DEFAULT_CV_DIR),
        help="Dossier contenant les CVs PDF (défaut : data/cv/ENGINEERING/)",
    )
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=True,
        help="Enregistrer les résultats (--no-save pour désactiver)",
    )
    args = parser.parse_args()

    Ml6KeywordMatcher().run(
        limit=args.limit,
        announcement=Path(args.announcement),
        cv_dir=Path(args.cv_dir),
        save=args.save,
    )


if __name__ == "__main__":
    main()
