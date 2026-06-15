"""Logique de scoring entre une offre et un CV."""

from ats_system.scoring.keyword import kw_match_score, ml6_kw_match_score
from ats_system.scoring.embedding import emb_cos_score

__all__ = ["kw_match_score", "ml6_kw_match_score", "emb_cos_score"]
