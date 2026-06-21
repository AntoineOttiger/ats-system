"""Classe les CVs ENGINEERING face à l'annonce par défaut via la similarité d'embeddings.

Toute la logique (chargement, scoring, sauvegarde) vit dans ``EmbeddingCosineScorer.run()``.
"""

import argparse

from ats_system.systems import EmbeddingCosineScorer


def main():
    parser = argparse.ArgumentParser(description="Calcule le emb_cos_score des CVs ENGINEERING.")
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=True,
        help="Enregistrer les résultats (--no-save pour désactiver)",
    )
    args = parser.parse_args()

    EmbeddingCosineScorer().run(limit=args.limit, save=args.save)


if __name__ == "__main__":
    main()
