from ats_system.config import CV_DIR
from ats_system.data import import_pdf
from ats_system.systems import Ml6KeywordMatcher

CV_PATH = CV_DIR / "ENGINEERING" / "10030015.pdf"


if __name__ == "__main__":
    print("Chargement du modèle...")
    matcher = Ml6KeywordMatcher()
    matcher.import_model()

    print("Extraction du texte du CV...")
    cv_text = import_pdf(str(CV_PATH))["content"]

    print("Inférence...")
    keywords = matcher.extract_keywords(cv_text)

    print(f"\n{len(keywords)} keywords extraits :")
    for kw in keywords:
        print(f"  - {kw}")
