from transformers import pipeline


def import_model():
    """Charge le modèle d'extraction de keyphrases ml6team (BERT, token-classification)."""
    return pipeline(
        "token-classification",
        model="ml6team/keyphrase-extraction-kbir-inspec",
        aggregation_strategy="simple",
    )


def infer_model(model, text: str) -> list[str]:
    # BERT-based model: max 512 tokens (~400 words). Split text into chunks to cover full documents.
    words = text.split()
    chunk_size = 400
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    keywords = []
    seen = set()
    for chunk in chunks:
        for entity in model(chunk):
            kw = entity["word"].strip()
            if entity["entity_group"] == "KEY" and kw.lower() not in seen:
                seen.add(kw.lower())
                keywords.append(kw)

    return keywords
