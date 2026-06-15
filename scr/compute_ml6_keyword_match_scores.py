import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from LLMs.word_extractor.ml6team import import_model
from tools.scores import ml6_keyword_match_score
from tools.data_manager import import_pdf

ANNOUNCEMENT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "announcement", "mechanical_engineer_job_posting_2016.pdf"
)
CV_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "cv", "ENGINEERING")


def main():
    parser = argparse.ArgumentParser(description="Calcule le ml6_keyword_match_score de tous les CVs ENGINEERING.")
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    print("Chargement du modèle...")
    model = import_model()

    print("Extraction du texte de l'annonce...")
    offre_text = import_pdf(ANNOUNCEMENT_PATH)

    cv_files = [f for f in os.listdir(CV_DIR) if f.endswith(".pdf")]
    if limit is not None:
        cv_files = cv_files[:limit]

    results = []
    for i, filename in enumerate(cv_files, 1):
        print(f"Traitement CV {i}/{len(cv_files)} : {filename}")
        cv_path = os.path.join(CV_DIR, filename)
        cv_text = import_pdf(cv_path)
        score = ml6_keyword_match_score(model, offre_text, cv_text)["score"]
        results.append((filename, score))

    results.sort(key=lambda x: x[1], reverse=True)

    print()
    for filename, score in results:
        print(f"{score:5.1f}%  {filename}")


if __name__ == "__main__":
    main()
