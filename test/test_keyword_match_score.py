import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.scores import keyword_match_score
from tools.data_manager import import_pdf

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DEFAULT_OFFRE = os.path.join(DATASET_DIR, "announcement", "mechanical_engineer_job_posting_2016.pdf")
DEFAULT_CV = os.path.join(DATASET_DIR, "cv", "ENGINEERING", "12472574.pdf")


def main():
    parser = argparse.ArgumentParser(description="Calcule le keyword_match_score entre une annonce et un CV (PDFs).")
    parser.add_argument("--offre", default=DEFAULT_OFFRE, help="Chemin vers le PDF de l'annonce")
    parser.add_argument("--cv", default=DEFAULT_CV, help="Chemin vers le PDF du CV")
    args = parser.parse_args()

    offre_text = import_pdf(args.offre)
    cv_text = import_pdf(args.cv)

    result = keyword_match_score(offre_text, cv_text)

    print(f"Annonce   : {os.path.basename(args.offre)}")
    print(f"CV        : {os.path.basename(args.cv)}")
    print(f"Score     : {result['score']}%")
    print(f"Matching  ({len(result['matching'])} mots) : {', '.join(sorted(result['matching']))}")
    print(f"Missing   ({len(result['missing'])} mots) : {', '.join(sorted(result['missing']))}")


if __name__ == "__main__":
    main()
