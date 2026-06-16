import argparse

from ats_system.config import DEFAULT_ANNOUNCEMENT, CV_DIR, DEFAULT_CV_CATEGORY
from ats_system.data import import_pdf
from ats_system.models.keyphrase_extractor import import_model
from ats_system.scoring import ml6_extract_keywords, match_score

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def main():
    parser = argparse.ArgumentParser(description="Calcule le ml6_kw_match_score de tous les CVs ENGINEERING.")
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    print("Chargement du modèle...")
    model = import_model()

    print("Extraction du texte de l'annonce...")
    offre = import_pdf(str(DEFAULT_ANNOUNCEMENT))
    keywords_offre = ml6_extract_keywords(model, offre["content"])

    cv_files = sorted(CATEGORY_DIR.glob("*.pdf"))
    if limit is not None:
        cv_files = cv_files[:limit]

    results = []
    for i, cv_path in enumerate(cv_files, 1):
        print(f"Traitement CV {i}/{len(cv_files)} : {cv_path.name}")
        cv = import_pdf(str(cv_path))
        keywords_cv = ml6_extract_keywords(model, cv["content"])
        score = match_score(keywords_offre, keywords_cv)["score"]
        results.append((cv["id"], score))

    results.sort(key=lambda x: x[1], reverse=True)

    print()
    for filename, score in results:
        print(f"{score:5.1f}%  {filename}")


if __name__ == "__main__":
    main()
