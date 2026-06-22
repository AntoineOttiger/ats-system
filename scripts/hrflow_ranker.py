"""Classe les CVs face à l'annonce par défaut via l'API HrFlow (indexation JSON + scoring).

Toute la logique (chargement, scoring, sauvegarde) vit dans ``HrflowCVRanker.run()``.
"""

import argparse

from ats_system.systems import HrflowCVRanker


def main():
    parser = argparse.ArgumentParser(
        description="Classe les CVs face à l'annonce par défaut via l'API HrFlow."
    )
    parser.add_argument("--limit", type=int, default=2, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=True,
        help="Enregistrer les résultats (--no-save pour désactiver)",
    )
    args = parser.parse_args()

    ranker = HrflowCVRanker()
    ranker.run(limit=args.limit, save=args.save)


if __name__ == "__main__":
    main()
