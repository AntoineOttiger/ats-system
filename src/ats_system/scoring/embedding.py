from sentence_transformers import SentenceTransformer, util


def emb_cos_score(model: SentenceTransformer, offre: str, cv: str) -> float:
    embeddings = model.encode([offre, cv], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(score * 100, 1)
