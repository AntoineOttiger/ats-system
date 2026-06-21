"""Classe les CVs ENGINEERING face à l'annonce par défaut via le sliding window ranker (LLM).

Toute la logique (chargement, classement, sauvegarde) vit dans ``SlidingWindowCVRanker.run()``.
"""

import argparse

from ats_system.config import SLIDING_WINDOW_MODEL
from ats_system.systems import SlidingWindowCVRanker


def main():
    parser = argparse.ArgumentParser(
        description="Classe les CVs ENGINEERING face à l'annonce par défaut via le sliding window "
        "ranker (LLM, modèle défini dans config.py)."
    )
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument("--window-size", type=int, default=4, help="Nombre de CVs comparés par appel LLM")
    parser.add_argument("--passes", type=int, default=3, help="Nombre maximum de passes")
    parser.add_argument(
        "--model", type=str, default=SLIDING_WINDOW_MODEL,
        help="Identifiant du modèle (fournisseur déduit du préfixe : Claude ou Mistral)",
    )
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=True,
        help="Enregistrer les résultats (--no-save pour désactiver)",
    )
    args = parser.parse_args()

    ranker = SlidingWindowCVRanker(window_size=args.window_size, num_passes=args.passes, model=args.model)
    ranker.run(limit=args.limit, save=args.save)


if __name__ == "__main__":
    main()
