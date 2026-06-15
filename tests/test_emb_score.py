import argparse

from ats_system.config import DEFAULT_ANNOUNCEMENT, DEFAULT_CV
from ats_system.data import import_pdf
from ats_system.models.embedding_model import import_model
from ats_system.scoring import emb_cos_score


def main():
    parser = argparse.ArgumentParser(description="Calcule le emb_cos_score entre une annonce et un CV (PDFs).")
    parser.add_argument("--offre", default=str(DEFAULT_ANNOUNCEMENT), help="Chemin vers le PDF de l'annonce")
    parser.add_argument("--cv", default=str(DEFAULT_CV), help="Chemin vers le PDF du CV")
    args = parser.parse_args()

    print("Chargement du modèle...")
    model = import_model()

    print("Extraction du texte...")
    offre_text = import_pdf(args.offre)
    cv_text = import_pdf(args.cv)

    print("Calcul du score...")
    score = emb_cos_score(model, offre_text, cv_text)

    print(f"\nAnnonce : {args.offre}")
    print(f"CV      : {args.cv}")
    print(f"Score   : {score}%")


if __name__ == "__main__":
    main()
