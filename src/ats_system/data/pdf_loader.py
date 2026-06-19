from pathlib import Path

from pypdf import PdfReader


def import_pdf(file_path: str) -> dict:
    """Extract a PDF and return it as ``{"id": <filename>, "content": <text>}``."""
    reader = PdfReader(file_path)
    content = "\n".join(page.extract_text() or "" for page in reader.pages)
    return {"id": Path(file_path).name, "content": content}
