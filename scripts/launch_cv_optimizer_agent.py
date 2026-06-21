"""Lance le CVOptimizerAgent sur le CV « à optimiser » d'un dataset synthétique.

Toute la logique (localisation du CV à optimiser, optimisation en streamant les pensées de
l'agent, sauvegarde du PDF optimisé + méta sous ``results/cv_optimizer/<timestamp>/``) vit
dans ``CVOptimizerAgent.run()``.

Lancement : ``python scripts/launch_cv_optimizer_agent.py``
(nécessite MISTRAL_API_KEY dans .env et un dataset sous data/generated_data/).
"""

import argparse
from pathlib import Path

from ats_system.agents import CVOptimizerAgent
from ats_system.config import CV_OPTIMIZER_MODEL, DEFAULT_ANNOUNCEMENT, GENERATED_DATA_DIR
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
        description="Lance le CVOptimizerAgent : optimise le CV « à optimiser » d'un dataset "
        "synthétique face à l'annonce, en affichant toutes les pensées de l'agent."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dossier du dataset synthétique (défaut : le plus récent sous data/generated_data/)",
    )
    parser.add_argument(
        "--announcement", type=str, default=str(DEFAULT_ANNOUNCEMENT),
        help="Chemin du PDF de l'annonce (défaut : annonce par défaut du projet)",
    )
    parser.add_argument("--model", type=str, default=CV_OPTIMIZER_MODEL, help="Modèle Mistral")
    parser.add_argument(
        "--max-iterations", type=int, default=20,
        help="Limite de récursion du graphe (étapes LLM + outils)",
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

    print("\nExtraction du texte de l'annonce...")
    announcement_text = import_pdf(str(announcement_path))["content"]

    print("Initialisation de l'agent (LLM Mistral + ranker + cache des CV concurrents)...")
    agent = CVOptimizerAgent(
        dataset_dir=dataset_dir,
        announcement_text=announcement_text,
        announcement_name=announcement_path.name,
        model=args.model,
        max_iterations=args.max_iterations,
    )
    agent.import_model()

    agent.run(save=args.save)


if __name__ == "__main__":
    main()
