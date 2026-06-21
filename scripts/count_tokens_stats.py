import argparse
import statistics

from transformers import AutoTokenizer

from ats_system.config import CV_DIR, DEFAULT_CV_CATEGORY, ML6_KEYWORD_MODEL
from ats_system.data import import_pdf

MODEL_NAME = ML6_KEYWORD_MODEL

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def main():
    parser = argparse.ArgumentParser(
        description=f"Calcule la moyenne et l'écart type des tokens des CVs du dossier {DEFAULT_CV_CATEGORY}."
    )
    parser.add_argument("--limit", type=int, default=0, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument("--model", default=MODEL_NAME, help="Modèle HuggingFace dont utiliser le tokenizer")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    cv_files = sorted(CATEGORY_DIR.glob("*.pdf"))
    if limit is not None:
        cv_files = cv_files[:limit]

    token_counts = []
    for i, cv_path in enumerate(cv_files, 1):
        text = import_pdf(str(cv_path))["content"]
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        token_counts.append(n_tokens)
        print(f"Traitement CV {i}/{len(cv_files)} : {cv_path.name} ({n_tokens} tokens)")

    if not token_counts:
        print(f"\nAucun PDF trouvé dans {CATEGORY_DIR}")
        return

    mean = statistics.mean(token_counts)
    stdev = statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0

    print(f"\nDossier   : {CATEGORY_DIR}")
    print(f"Modèle    : {args.model}")
    print(f"CVs       : {len(token_counts)}")
    print(f"Min       : {min(token_counts)}")
    print(f"Max       : {max(token_counts)}")
    print(f"Moyenne   : {mean:.1f}")
    print(f"Écart type: {stdev:.1f}")


if __name__ == "__main__":
    main()
