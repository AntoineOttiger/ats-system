import argparse

from ats_system.config import DEFAULT_ANNOUNCEMENT, CV_DIR, DEFAULT_CV_CATEGORY
from ats_system.data import import_pdf
from ats_system.results_io import build_ranking, save_results
from ats_system.scoring import baseline_extract_keywords, match_score

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def main():
    parser = argparse.ArgumentParser(description="Calcule le baseline_kw_match_score de tous les CVs ENGINEERING.")
    parser.add_argument("--limit", type=int, default=0, help="Nombre maximum de CVs à traiter (0 = tous)")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    offre = import_pdf(str(DEFAULT_ANNOUNCEMENT))
    keywords_offre = baseline_extract_keywords(offre["content"])

    cv_files = sorted(CATEGORY_DIR.glob("*.pdf"))
    if limit is not None:
        cv_files = cv_files[:limit]

    results = []
    for cv_path in cv_files:
        cv = import_pdf(str(cv_path))
        keywords_cv = baseline_extract_keywords(cv["content"])
        score = match_score(keywords_offre, keywords_cv)["score"]
        results.append((cv["id"], score))

    results.sort(key=lambda x: x[1], reverse=True)

    for filename, score in results:
        print(f"{score:5.1f}%  {filename}")

    params = {
        "announcement": DEFAULT_ANNOUNCEMENT.name,
        "category": DEFAULT_CV_CATEGORY,
        "limit": args.limit,
        "num_cvs": len(results),
    }
    out_path = save_results("baseline_keyword_match", build_ranking(results), params)
    print(f"\nRésultats sauvegardés dans : {out_path}")


if __name__ == "__main__":
    main()
