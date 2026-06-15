from sentence_transformers import SentenceTransformer


def import_model() -> SentenceTransformer:
    """Charge le modèle d'embeddings de phrases all-MiniLM-L6-v2."""
    return SentenceTransformer("all-MiniLM-L6-v2")
