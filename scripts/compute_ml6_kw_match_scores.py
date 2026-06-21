import argparse

from ats_system.config import DEFAULT_ANNOUNCEMENT, CV_DIR, DEFAULT_CV_CATEGORY, ML6_KEYWORD_MODEL
from ats_system.data import import_pdf
from ats_system.results_io import build_ranking, save_results
from ats_system.systems import Ml6KeywordMatcher

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def main():
    parser = argparse.ArgumentParser(description="Calcule le ml6_kw_match_score de tous les CVs ENGINEERING.")
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    print("Chargement du modèle...")
    matcher = Ml6KeywordMatcher()
    matcher.import_model()

    print("Extraction du texte de l'annonce...")
    offre = import_pdf(str(DEFAULT_ANNOUNCEMENT))
    keywords_offre = matcher.extract_keywords(offre["content"])

    cv_files = sorted(CATEGORY_DIR.glob("*.pdf"))
    if limit is not None:
        cv_files = cv_files[:limit]

    results = []
    for i, cv_path in enumerate(cv_files, 1):
        print(f"Traitement CV {i}/{len(cv_files)} : {cv_path.name}")
        cv = import_pdf(str(cv_path))
        keywords_cv = matcher.extract_keywords(cv["content"])
        score = matcher.match(keywords_offre, keywords_cv)["score"]
        results.append((cv["id"], score))

    results.sort(key=lambda x: x[1], reverse=True)

    print()
    for filename, score in results:
        print(f"{score:5.1f}%  {filename}")

    params = {
        "announcement": DEFAULT_ANNOUNCEMENT.name,
        "category": DEFAULT_CV_CATEGORY,
        "model": ML6_KEYWORD_MODEL,
        "limit": args.limit,
        "num_cvs": len(results),
    }
    out_path = save_results("ml6_keyword_match", build_ranking(results), params)
    print(f"\nRésultats sauvegardés dans : {out_path}")


if __name__ == "__main__":
    main()
