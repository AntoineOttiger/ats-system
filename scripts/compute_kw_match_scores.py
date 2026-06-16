from ats_system.config import DEFAULT_ANNOUNCEMENT, CV_DIR, DEFAULT_CV_CATEGORY
from ats_system.data import import_pdf
from ats_system.scoring import baseline_extract_keywords, match_score

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def main():
    offre = import_pdf(str(DEFAULT_ANNOUNCEMENT))
    keywords_offre = baseline_extract_keywords(offre["content"])

    results = []
    for cv_path in sorted(CATEGORY_DIR.glob("*.pdf")):
        cv = import_pdf(str(cv_path))
        keywords_cv = baseline_extract_keywords(cv["content"])
        score = match_score(keywords_offre, keywords_cv)["score"]
        results.append((cv["id"], score))

    results.sort(key=lambda x: x[1], reverse=True)

    for filename, score in results:
        print(f"{score:5.1f}%  {filename}")


if __name__ == "__main__":
    main()
