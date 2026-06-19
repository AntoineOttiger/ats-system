"""Système ATS par mots-clés baseline (regex + stopwords FR/EN).

Extraction naïve des mots-clés (minuscules, suppression de la ponctuation et des
stopwords), puis score d'adéquation = proportion des mots-clés de l'offre retrouvés
dans le CV.
"""

import re

import nltk
from nltk.corpus import stopwords


class BaselineKeywordMatcher:
    """Score d'adéquation offre/CV par intersection de mots-clés (baseline)."""

    def __init__(self) -> None:
        self._stop_words: set[str] = set()

    def import_model(self) -> None:
        """Télécharge les stopwords nltk et pré-calcule le set FR/EN. À appeler avant usage."""
        nltk.download("stopwords", quiet=True)
        self._stop_words = set(stopwords.words("french")) | set(stopwords.words("english"))

    def extract_keywords(self, text: str) -> set:
        """Extrait les mots-clés d'un texte (minuscules, hors ponctuation et stopwords)."""
        text = text.lower()
        # Supprime tout ce qui n'est ni une lettre ni un espace (ex : "3.0").
        text = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", text)
        words = text.split()
        return set(w for w in words if w not in self._stop_words and len(w) > 2)

    @staticmethod
    def match(keywords_offre: set, keywords_cv: set) -> dict:
        """Compare les mots-clés offre/CV → ``{score, matching, missing}`` (score 0–100)."""
        matching = keywords_offre & keywords_cv
        missing = keywords_offre - keywords_cv
        score = round(len(matching) / len(keywords_offre) * 100, 1) if keywords_offre else 0
        return {
            "score": score,
            "matching": matching,
            "missing": missing,
        }
