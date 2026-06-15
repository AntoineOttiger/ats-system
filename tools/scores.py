import re
import nltk
from nltk.corpus import stopwords
from LLMs.word_extractor.ml6team import infer_model

nltk.download('stopwords', quiet=True)

def keyword_match_score(offre: str, cv: str) -> dict:
    stop_words = set(stopwords.words('french')) | set(stopwords.words('english'))

    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^a-zàâäéèêëîïôùûüç\s]', ' ', text) # delete everything that is not a space or a letter (ex : "3.0")
        words = text.split()
        return set(w for w in words if w not in stop_words and len(w) > 2)

    words1 = tokenize(offre)
    words2 = tokenize(cv)

    matching = words1 & words2
    missing = words1 - words2

    score = round(len(matching) / len(words1) * 100, 1) if words1 else 0

    return {
        "score": score,
        "matching": matching,
        "missing": missing,
    }


def ml6_keyword_match_score(model, offre: str, cv: str) -> dict:
    keywords_offre = set(kw.lower() for kw in infer_model(model, offre))
    keywords_cv = set(kw.lower() for kw in infer_model(model, cv))

    matching = keywords_offre & keywords_cv
    missing = keywords_offre - keywords_cv

    score = round(len(matching) / len(keywords_offre) * 100, 1) if keywords_offre else 0

    return {
        "score": score,
        "matching": matching,
        "missing": missing,
    }