"""Système ATS par mots-clés via le modèle ml6team (BERT, token-classification).

L'extraction des mots-clés passe par le modèle HF
``ml6team/keyphrase-extraction-kbir-inspec`` (limite 512 tokens : le texte est
découpé en chunks de 400 mots). Le score d'adéquation suit la même logique que la
baseline : proportion des mots-clés de l'offre retrouvés dans le CV.
"""

from transformers import pipeline

from ats_system.config import ML6_KEYWORD_MODEL

MODEL_NAME = ML6_KEYWORD_MODEL
# BERT : max 512 tokens (~400 mots) ; on découpe le texte pour couvrir le document entier.
CHUNK_SIZE = 400


class Ml6KeywordMatcher:
    """Score d'adéquation offre/CV par intersection de mots-clés (modèle ml6team)."""

    def __init__(self) -> None:
        self.model = None

    def import_model(self) -> None:
        """Charge la pipeline d'extraction de keyphrases ml6team. À appeler avant usage."""
        self.model = pipeline(
            "token-classification",
            model=MODEL_NAME,
            aggregation_strategy="simple",
        )

    def extract_keywords(self, text: str) -> set:
        """Extrait les mots-clés d'un texte (chunks de 400 mots, dédup, minuscules)."""
        if self.model is None:
            raise RuntimeError("Appelez import_model() avant d'extraire des mots-clés.")

        words = text.split()
        chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

        keywords: set[str] = set()
        for chunk in chunks:
            for entity in self.model(chunk):
                if entity["entity_group"] == "KEY":
                    keywords.add(entity["word"].strip().lower())
        return keywords

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
