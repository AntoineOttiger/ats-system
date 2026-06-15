from pypdf import PdfReader


def import_pdf(file_path: str) -> str:
    """Extract and return all text content from a PDF file."""
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
