from ats_system.config import CV_DIR
from ats_system.data import import_pdf

PDF_PATH = CV_DIR / "ACCOUNTANT" / "10554236.pdf"


def test_import_pdf_returns_dict():
    result = import_pdf(str(PDF_PATH))
    assert isinstance(result, dict)
    assert set(result) == {"id", "content"}


def test_import_pdf_id_is_filename():
    result = import_pdf(str(PDF_PATH))
    assert result["id"] == PDF_PATH.name


def test_import_pdf_content_is_string():
    result = import_pdf(str(PDF_PATH))
    assert isinstance(result["content"], str)


def test_import_pdf_not_empty():
    result = import_pdf(str(PDF_PATH))
    assert len(result["content"].strip()) > 0


def test_import_pdf_content_preview():
    result = import_pdf(str(PDF_PATH))
    print("\n--- Aperçu du contenu extrait ---")
    print(f"id : {result['id']}")
    print(result["content"][:500])
    print("---")


if __name__ == "__main__":
    test_import_pdf_returns_dict()
    print("OK - Le resultat est bien un dictionnaire {id, content}")

    test_import_pdf_id_is_filename()
    print("OK - L'id correspond au nom du fichier")

    test_import_pdf_content_is_string()
    print("OK - Le contenu est bien une chaine de caracteres")

    test_import_pdf_not_empty()
    print("OK - Le contenu n'est pas vide")

    test_import_pdf_content_preview()
