import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.data_manager import import_pdf

PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "dataset",
    "cv",
    "ACCOUNTANT",
    "10554236.pdf",
)


def test_import_pdf_returns_string():
    result = import_pdf(PDF_PATH)
    assert isinstance(result, str)


def test_import_pdf_not_empty():
    result = import_pdf(PDF_PATH)
    assert len(result.strip()) > 0


def test_import_pdf_content_preview():
    result = import_pdf(PDF_PATH)
    print("\n--- Aperçu du contenu extrait ---")
    print(result[:500])
    print("---")


if __name__ == "__main__":
    test_import_pdf_returns_string()
    print("OK - Le resultat est bien une chaine de caracteres")

    test_import_pdf_not_empty()
    print("OK - Le contenu n'est pas vide")

    test_import_pdf_content_preview()
