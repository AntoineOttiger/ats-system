"""Lance l'OneShotCVOptimizer sur le CV « à optimiser » d'un dataset synthétique.

Baseline non itérative face à l'agent adversarial : le CV est réécrit en **un seul appel
LLM**, à l'aveugle (annonce + CV uniquement). Toute la logique (localisation du CV à
optimiser, mesure du rang avant/après, réécriture, sauvegarde du PDF optimisé + méta sous
``results/cv_optimizer_oneshot/<timestamp>/``) vit dans ``OneShotCVOptimizer.run()``.

Lancement : ``python scripts/launch_one_shot_cv_optimizer.py``
(nécessite la clé API du fournisseur dans .env et un dataset sous data/generated_data/).
"""

import argparse
from pathlib import Path

from ats_system.agents import OneShotCVOptimizer
from ats_system.agents.dataset_rankers import CV_OPTIMIZER_RANKERS
from ats_system.config import (
    CV_OPTIMIZER_MODEL,
    CV_OPTIMIZER_RANKER,
    DEFAULT_ANNOUNCEMENT,
    GENERATED_DATA_DIR,
)
from ats_system.data import import_pdf


def _latest_dataset() -> Path:
    """Retourne le dataset synthétique le plus récent sous GENERATED_DATA_DIR."""
    datasets = sorted(GENERATED_DATA_DIR.glob("synthetic_cvs_*"))
    if not datasets:
        raise FileNotFoundError(
            f"Aucun dataset synthétique trouvé sous {GENERATED_DATA_DIR}. "
            "Lancez d'abord : python scripts/generate_synthetic_cvs.py"
        )
    return datasets[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Lance l'OneShotCVOptimizer : optimise le CV « à optimiser » d'un dataset "
        "synthétique en un seul appel LLM (à l'aveugle), avec mesure du rang avant/après."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dossier du dataset synthétique (défaut : le plus récent sous data/generated_data/)",
    )
    parser.add_argument(
        "--announcement", type=str, default=str(DEFAULT_ANNOUNCEMENT),
        help="Chemin du PDF de l'annonce (défaut : annonce par défaut du projet)",
    )
    parser.add_argument(
        "--model", type=str, default=CV_OPTIMIZER_MODEL,
        help="Modèle (fournisseur déduit du préfixe : Mistral ou Claude)",
    )
    parser.add_argument(
        "--ranker", type=str, default=CV_OPTIMIZER_RANKER, choices=list(CV_OPTIMIZER_RANKERS),
        help="Méthode de classement pour mesurer le rang avant/après (défaut : config)",
    )
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=True,
        help="Enregistrer le CV optimisé + méta (--no-save pour désactiver)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset) if args.dataset else _latest_dataset()
    announcement_path = Path(args.announcement)

    print(f"Dataset    : {dataset_dir}")
    print(f"Annonce    : {announcement_path}")
    print(f"Ranker     : {args.ranker}")

    print("\nExtraction du texte de l'annonce...")
    announcement_text = import_pdf(str(announcement_path))["content"]

    print("Initialisation de l'optimiseur (client LLM + ranker + cache des CV concurrents)...")
    optimizer = OneShotCVOptimizer(
        dataset_dir=dataset_dir,
        announcement_text=announcement_text,
        announcement_name=announcement_path.name,
        model=args.model,
        ranker_name=args.ranker,
    )
    optimizer.import_model()

    optimizer.run(save=args.save)


if __name__ == "__main__":
    main()
