from ats_system.config import CV_DIR
from ats_system.data import import_pdf

PDF_PATH = CV_DIR / "ACCOUNTANT" / "10554236.pdf"


def test_import_pdf_returns_string():
    result = import_pdf(str(PDF_PATH))
    assert isinstance(result, str)


def test_import_pdf_not_empty():
    result = import_pdf(str(PDF_PATH))
    assert len(result.strip()) > 0


def test_import_pdf_content_preview():
    result = import_pdf(str(PDF_PATH))
    print("\n--- Aperçu du contenu extrait ---")
    print(result[:500])
    print("---")


if __name__ == "__main__":
    test_import_pdf_returns_string()
    print("OK - Le resultat est bien une chaine de caracteres")

    test_import_pdf_not_empty()
    print("OK - Le contenu n'est pas vide")

    test_import_pdf_content_preview()
