import argparse

from ats_system.config import DEFAULT_ANNOUNCEMENT, DEFAULT_CV
from ats_system.data import import_pdf
from ats_system.systems import Ml6KeywordMatcher


def main():
    parser = argparse.ArgumentParser(description="Calcule le ml6_kw_match_score entre une annonce et un CV (PDFs).")
    parser.add_argument("--offre", default=str(DEFAULT_ANNOUNCEMENT), help="Chemin vers le PDF de l'annonce")
    parser.add_argument("--cv", default=str(DEFAULT_CV), help="Chemin vers le PDF du CV")
    args = parser.parse_args()

    print("Chargement du modèle...")
    matcher = Ml6KeywordMatcher()
    matcher.import_model()

    print("Extraction du texte...")
    offre_text = import_pdf(args.offre)["content"]
    cv_text = import_pdf(args.cv)["content"]

    print("Calcul du score...")
    keywords_offre = matcher.extract_keywords(offre_text)
    keywords_cv = matcher.extract_keywords(cv_text)
    result = matcher.match(keywords_offre, keywords_cv)

    print(f"\nAnnonce   : {args.offre}")
    print(f"CV        : {args.cv}")
    print(f"Score     : {result['score']}%")
    print(f"Matching  ({len(result['matching'])} mots) : {', '.join(sorted(result['matching']))}")
    print(f"Missing   ({len(result['missing'])} mots) : {', '.join(sorted(result['missing']))}")


if __name__ == "__main__":
    main()
