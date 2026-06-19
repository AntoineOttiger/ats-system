"""Système ATS par similarité cosinus d'embeddings de phrases.

L'offre et le CV sont encodés par le modèle ``all-MiniLM-L6-v2`` ; le score est leur
similarité cosinus ramenée sur une échelle 0–100.
"""

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingCosineScorer:
    """Score d'adéquation offre/CV par similarité cosinus d'embeddings."""

    def __init__(self) -> None:
        self.model: SentenceTransformer | None = None

    def import_model(self) -> None:
        """Charge le modèle d'embeddings de phrases. À appeler avant usage."""
        self.model = SentenceTransformer(MODEL_NAME)

    def score(self, offre: str, cv: str) -> float:
        """Similarité cosinus offre/CV, sur une échelle 0–100."""
        if self.model is None:
            raise RuntimeError("Appelez import_model() avant de calculer un score.")
        embeddings = self.model.encode([offre, cv], convert_to_tensor=True)
        sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        return round(sim * 100, 1)
