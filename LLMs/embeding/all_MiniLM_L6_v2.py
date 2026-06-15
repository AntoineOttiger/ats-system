from sentence_transformers import SentenceTransformer


def import_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")
