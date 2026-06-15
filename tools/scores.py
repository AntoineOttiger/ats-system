import re
import nltk
from nltk.corpus import stopwords

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