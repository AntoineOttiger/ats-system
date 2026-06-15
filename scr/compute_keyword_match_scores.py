import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.scores import keyword_match_score
from tools.data_manager import import_pdf

ANNOUNCEMENT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "announcement", "mechanical_engineer_job_posting_2016.pdf"
)
CV_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "cv", "ENGINEERING")


def main():
    offre_text = import_pdf(ANNOUNCEMENT_PATH)

    results = []
    for filename in os.listdir(CV_DIR):
        if not filename.endswith(".pdf"):
            continue
        cv_path = os.path.join(CV_DIR, filename)
        cv_text = import_pdf(cv_path)
        score = keyword_match_score(offre_text, cv_text)["score"]
        results.append((filename, score))

    results.sort(key=lambda x: x[1], reverse=True)

    for filename, score in results:
        print(f"{score:5.1f}%  {filename}")


if __name__ == "__main__":
    main()
