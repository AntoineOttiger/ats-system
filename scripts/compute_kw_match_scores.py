"""Classe les CVs ENGINEERING face à l'annonce par défaut via les mots-clés baseline.

Toute la logique (chargement, scoring, sauvegarde) vit dans ``BaselineKeywordMatcher.run()``.
"""

import argparse

from ats_system.systems import BaselineKeywordMatcher


def main():
    parser = argparse.ArgumentParser(description="Calcule le baseline_kw_match_score des CVs ENGINEERING.")
    parser.add_argument("--limit", type=int, default=0, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=True,
        help="Enregistrer les résultats (--no-save pour désactiver)",
    )
    args = parser.parse_args()

    BaselineKeywordMatcher().run(limit=args.limit, save=args.save)


if __name__ == "__main__":
    main()
