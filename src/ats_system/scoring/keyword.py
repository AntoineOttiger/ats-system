import re
import nltk
from nltk.corpus import stopwords
from ats_system.models.keyphrase_extractor import infer_model

nltk.download('stopwords', quiet=True)


def baseline_extract_keywords(text: str) -> set:
    stop_words = set(stopwords.words('french')) | set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'[^a-zàâäéèêëîïôùûüç\s]', ' ', text) # delete everything that is not a space or a letter (ex : "3.0")
    words = text.split()
    return set(w for w in words if w not in stop_words and len(w) > 2)


def ml6_extract_keywords(model, text: str) -> set:
    return set(kw.lower() for kw in infer_model(model, text))


def match_score(keywords_offre: set, keywords_cv: set) -> dict:
    matching = keywords_offre & keywords_cv
    missing = keywords_offre - keywords_cv

    score = round(len(matching) / len(keywords_offre) * 100, 1) if keywords_offre else 0

    return {
        "score": score,
        "matching": matching,
        "missing": missing,
    }
