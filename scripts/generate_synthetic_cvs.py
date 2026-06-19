import argparse
from pathlib import Path

from ats_system.config import CV_GENERATOR_MODEL, DEFAULT_ANNOUNCEMENT
from ats_system.data import import_pdf
from ats_system.generators import SyntheticCVGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Génère des CVs synthétiques (PDF) face à une annonce via l'API Mistral. "
        "Chaque CV cible un niveau de profil discret (du candidat idéal au hors-sujet) ; "
        "un manifest.json de vérité-terrain accompagne les CVs."
    )
    parser.add_argument("--count", type=int, default=8, help="Nombre de CVs à générer")
    parser.add_argument(
        "--announcement",
        type=str,
        default=str(DEFAULT_ANNOUNCEMENT),
        help="Chemin du PDF de l'annonce (défaut : annonce par défaut du projet)",
    )
    parser.add_argument(
        "--model", type=str, default=CV_GENERATOR_MODEL, help="Identifiant du modèle Mistral"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Température d'échantillonnage du modèle"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Ne pas générer le CV « à optimiser » (généré par défaut, en plus du lot)",
    )
    parser.add_argument(
        "--optimize-prompt",
        type=str,
        default=None,
        help="Consigne personnalisée pour le CV « à optimiser » (candidat excellent sur le "
        "fond mais au vocabulaire non aligné avec l'annonce). Défaut : consigne intégrée.",
    )
    args = parser.parse_args()

    print("Extraction du texte de l'annonce...")
    announcement = import_pdf(str(Path(args.announcement)))

    print("Initialisation du générateur Mistral...")
    generator = SyntheticCVGenerator(model=args.model, temperature=args.temperature)
    generator.import_model()

    optimize_msg = "" if args.no_optimize else " + 1 CV « à optimiser »"
    print(f"Génération de {args.count} CVs synthétiques{optimize_msg} (appels LLM)...")
    run_dir = generator.generate_cvs(
        announcement["content"],
        n=args.count,
        include_optimize=not args.no_optimize,
        optimize_instruction=args.optimize_prompt,
    )

    print(f"\nCVs générés dans : {run_dir}")


if __name__ == "__main__":
    main()
