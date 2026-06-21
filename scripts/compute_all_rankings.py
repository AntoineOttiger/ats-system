"""Classe les CVs ENGINEERING face à l'annonce par défaut via les quatre méthodes.

Toute la logique (chargement unique, exécution des 4 méthodes, sauvegarde dans un dossier
horodaté ``results/all_rankings/<timestamp>/``) vit dans ``AllRankingsRunner.run()``.
"""

import argparse

from ats_system.systems import AllRankingsRunner


def main():
    parser = argparse.ArgumentParser(
        description="Classe les CVs ENGINEERING face à l'annonce par défaut via les quatre méthodes."
    )
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à classer (0 = tous)")
    parser.add_argument("--window-size", type=int, default=4, help="Fenêtre glissante : CVs par appel LLM")
    parser.add_argument("--passes", type=int, default=10, help="Fenêtre glissante : nombre maximum de passes")
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=True,
        help="Enregistrer les résultats (--no-save pour désactiver)",
    )
    args = parser.parse_args()

    runner = AllRankingsRunner(window_size=args.window_size, num_passes=args.passes)
    runner.run(limit=args.limit, save=args.save)


if __name__ == "__main__":
    main()
