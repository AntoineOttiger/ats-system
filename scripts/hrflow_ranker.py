"""Classe les CVs face à l'annonce via l'API HrFlow (indexation JSON + scoring).

Toute la logique (chargement, scoring, sauvegarde) vit dans ``HrflowCVRanker.run()``.
"""

import argparse
from pathlib import Path

from ats_system.config import DEFAULT_ANNOUNCEMENT, DEFAULT_CV_DIR
from ats_system.systems import HrflowCVRanker


def main():
    parser = argparse.ArgumentParser(
        description="Classe les CVs face à l'annonce via l'API HrFlow."
    )
    parser.add_argument("--limit", type=int, default=2, help="Nombre maximum de CVs à traiter (0 = tous)")
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

    HrflowCVRanker().run(
        limit=args.limit,
        announcement=Path(args.announcement),
        cv_dir=Path(args.cv_dir),
        save=args.save,
    )


if __name__ == "__main__":
    main()
