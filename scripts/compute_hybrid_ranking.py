"""Classe les CVs ENGINEERING face à l'annonce par défaut via le système hybride.

Pipeline : présélection mots-clés (``Ml6KeywordMatcher``) puis affinage LLM par fenêtre
glissante (``SlidingWindowCVRanker``). Toute la logique (chargement, classement, sauvegarde
de l'historique dans ``results/hybrid_ranking/<timestamp>/``) vit dans
``HybridMl6SlidingWindowRanker.run()``.
"""

import argparse

from ats_system.config import SLIDING_WINDOW_MODEL
from ats_system.systems import HybridMl6SlidingWindowRanker


def main():
    parser = argparse.ArgumentParser(
        description="Classe les CVs ENGINEERING face à l'annonce par défaut via le système hybride "
        "(présélection mots-clés ml6 puis affinage LLM par fenêtre glissante)."
    )
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument("--window-size", type=int, default=4, help="Fenêtre glissante : CVs par appel LLM")
    parser.add_argument("--passes", type=int, default=3, help="Fenêtre glissante : nombre maximum de passes")
    parser.add_argument(
        "--model", type=str, default=SLIDING_WINDOW_MODEL,
        help="Modèle de l'affinage LLM (fournisseur déduit du préfixe : Claude ou Mistral)",
    )
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=True,
        help="Enregistrer les résultats (--no-save pour désactiver)",
    )
    args = parser.parse_args()

    ranker = HybridMl6SlidingWindowRanker(
        window_size=args.window_size,
        num_passes=args.passes,
        model=args.model,
    )
    ranker.run(limit=args.limit, save=args.save)


if __name__ == "__main__":
    main()
