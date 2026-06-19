"""Systèmes ATS de scoring/classement de CVs face à une offre d'emploi.

Chaque système est une classe autonome (un fichier par système) suivant la même
convention : ``import_model()`` charge le modèle (coûteux/facturé), puis les méthodes
d'inférence calculent les scores.
"""

from ats_system.systems.baseline_keyword_match import BaselineKeywordMatcher
from ats_system.systems.ml6_keyword_match import Ml6KeywordMatcher
from ats_system.systems.embedding_cosine import EmbeddingCosineScorer
from ats_system.systems.sliding_window_ranker import SlidingWindowCVRanker

__all__ = [
    "BaselineKeywordMatcher",
    "Ml6KeywordMatcher",
    "EmbeddingCosineScorer",
    "SlidingWindowCVRanker",
]
