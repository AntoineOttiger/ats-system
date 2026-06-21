"""Écriture d'un texte brut dans un PDF (police Helvetica, encodage latin-1).

Mutualisé entre le générateur de CVs synthétiques et l'agent d'optimisation (PDF du
CV optimisé). La police core Helvetica de fpdf2 n'accepte que le latin-1 (qui couvre
les accents français) ; la ponctuation typographique unicode est convertie en ASCII.
"""

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Ponctuation typographique unicode → équivalents latin-1.
_REPLACEMENTS = {
    "•": "-",
    "‣": "-",
    "◦": "-",
    "–": "-",
    "—": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    " ": " ",  # espace insécable
}


def _sanitize(text: str) -> str:
    """Remplace la ponctuation typographique unicode par des équivalents ASCII."""
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def write_text_pdf(text: str, path: Path) -> None:
    """Écrit ``text`` dans un PDF à ``path`` (police Helvetica, encodage latin-1)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    safe = _sanitize(text).encode("latin-1", errors="replace").decode("latin-1")
    for line in safe.split("\n"):
        # multi_cell gère le retour à la ligne ; une ligne vide marque un saut.
        # new_x=LMARGIN / new_y=NEXT : ramène le curseur à la marge gauche et passe
        # à la ligne suivante (sinon la cellule suivante n'a plus de largeur).
        pdf.multi_cell(0, 6, line if line else " ", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(str(path))
