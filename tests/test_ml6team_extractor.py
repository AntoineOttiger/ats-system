from ats_system.config import CV_DIR
from ats_system.data import import_pdf
from ats_system.models.keyphrase_extractor import import_model, infer_model

CV_PATH = CV_DIR / "ENGINEERING" / "10030015.pdf"


if __name__ == "__main__":
    print("Chargement du modèle...")
    model = import_model()

    print("Extraction du texte du CV...")
    cv_text = import_pdf(str(CV_PATH))["content"]

    print("Inférence...")
    keywords = infer_model(model, cv_text)

    print(f"\n{len(keywords)} keywords extraits :")
    for kw in keywords:
        print(f"  - {kw}")
