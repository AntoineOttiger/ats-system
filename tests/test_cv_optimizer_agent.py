"""Test exploratoire du CVOptimizerAgent.

Définit une annonce (DEFAULT_ANNOUNCEMENT) et un CV (le CV « à optimiser » d'un dataset
synthétique) en entrée, lance l'agent et imprime l'ENSEMBLE de ses « pensées » (raisonnement,
appels d'outils, observations de rang) puis le CV optimisé final.

Lancement : ``python tests/test_cv_optimizer_agent.py``
(nécessite MISTRAL_API_KEY dans .env et un dataset sous data/generated_data/).
"""

import argparse
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

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


def _find_optimize_cv(dataset_dir: Path) -> Path:
    """Trouve le CV « à optimiser » (entrée optimize: true) du manifest."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["cvs"]:
        if entry.get("optimize"):
            return dataset_dir / entry["file"]
    raise ValueError(f"Aucun CV « à optimiser » (optimize: true) dans {dataset_dir}.")


def _print_step(update: dict) -> str:
    """Affiche les messages d'une étape du graphe ; retourne le dernier contenu d'AIMessage."""
    last_ai_content = ""
    for node, payload in update.items():
        for message in payload.get("messages", []):
            if isinstance(message, AIMessage):
                if message.content:
                    content = message.content if isinstance(message.content, str) else str(message.content)
                    print(f"\n[PENSEE - {node}] Raisonnement :\n{content}")
                    last_ai_content = content
                for call in message.tool_calls or []:
                    args = {k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                            for k, v in call["args"].items()}
                    print(f"\n[OUTIL - {node}] Appel -> {call['name']}({args})")
            elif isinstance(message, ToolMessage):
                print(f"\n[OBSERVATION - {node}] ({message.name}) :\n{message.content}")
    return last_ai_content


def main():
    parser = argparse.ArgumentParser(
        description="Teste le CVOptimizerAgent : optimise le CV « à optimiser » d'un dataset "
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
    args = parser.parse_args()

    dataset_dir = Path(args.dataset) if args.dataset else _latest_dataset()
    cv_path = _find_optimize_cv(dataset_dir)

    print(f"Dataset    : {dataset_dir}")
    print(f"Annonce    : {args.announcement}")
    print(f"CV à opti. : {cv_path}")

    print("\nExtraction des textes...")
    announcement = import_pdf(args.announcement)["content"]
    cv_text = import_pdf(str(cv_path))["content"]

    print("Initialisation de l'agent (LLM Mistral + modèle ml6 + cache du dataset)...")
    agent = CVOptimizerAgent(
        dataset_dir=dataset_dir,
        announcement_text=announcement,
        model=args.model,
        max_iterations=args.max_iterations,
    )
    agent.import_model()

    print("\n" + "=" * 70)
    print("PENSÉES DE L'AGENT")
    print("=" * 70)
    final_cv = ""
    for update in agent.stream(cv_text):
        content = _print_step(update)
        if content:
            final_cv = content

    print("\n" + "=" * 70)
    print("CV OPTIMISÉ FINAL")
    print("=" * 70)
    print(final_cv)


if __name__ == "__main__":
    main()
