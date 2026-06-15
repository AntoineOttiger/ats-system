from ats_system.config import DEFAULT_ANNOUNCEMENT, CV_DIR, DEFAULT_CV_CATEGORY
from ats_system.data import import_pdf
from ats_system.scoring import kw_match_score

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def main():
    offre_text = import_pdf(str(DEFAULT_ANNOUNCEMENT))

    results = []
    for cv_path in sorted(CATEGORY_DIR.glob("*.pdf")):
        cv_text = import_pdf(str(cv_path))
        score = kw_match_score(offre_text, cv_text)["score"]
        results.append((cv_path.name, score))

    results.sort(key=lambda x: x[1], reverse=True)

    for filename, score in results:
        print(f"{score:5.1f}%  {filename}")


if __name__ == "__main__":
    main()
