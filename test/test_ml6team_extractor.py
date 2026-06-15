import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from LLMs.word_extractor.ml6team import import_model, infer_model
from tools.data_manager import import_pdf

CV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "cv", "ENGINEERING", "10030015.pdf"
)


if __name__ == "__main__":
    print("Chargement du modèle...")
    model = import_model()

    print("Extraction du texte du CV...")
    cv_text = import_pdf(CV_PATH)

    print("Inférence...")
    keywords = infer_model(model, cv_text)

    print(f"\n{len(keywords)} keywords extraits :")
    for kw in keywords:
        print(f"  - {kw}")
