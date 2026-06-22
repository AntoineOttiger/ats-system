"""Adaptateurs unifiant les systèmes de ``systems/`` en signal de rang pour l'agent.

Le ``CVOptimizerAgent`` a besoin d'un seul service : « classe ce CV candidat parmi les CVs
concurrents d'un dataset, face à l'annonce » → un rang + un classement + une analyse. Mais les
systèmes de :mod:`ats_system.systems` exposent **trois interfaces incompatibles** :

- :class:`~ats_system.systems.SlidingWindowCVRanker` classe tous les CVs ensemble (jugement LLM
  holistique) et fournit déjà une justification par CV ;
- :class:`~ats_system.systems.BaselineKeywordMatcher` / :class:`~ats_system.systems.Ml6KeywordMatcher`
  scorent **chaque CV** par intersection de mots-clés (``{score, matching, missing}``) ;
- :class:`~ats_system.systems.EmbeddingCosineScorer` score **chaque CV** par similarité cosinus.

Ce module enveloppe chacun derrière l'interface commune :class:`DatasetRanker` (convention projet
``import_model()`` puis :meth:`~DatasetRanker.rank`), pour que l'agent reste agnostique au système
employé. Le choix se fait par nom via :func:`build_dataset_ranker` (cf. la constante
``CV_OPTIMIZER_RANKER`` de ``config.py``).
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from ats_system.systems import (
    BaselineKeywordMatcher,
    EmbeddingCosineScorer,
    HybridMl6SlidingWindowRanker,
    Ml6KeywordMatcher,
    SlidingWindowCVRanker,
)

if TYPE_CHECKING:
    from langchain_core.rate_limiters import BaseRateLimiter

    from ats_system.llm import TokensPerMinuteRateLimiter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface commune
# ---------------------------------------------------------------------------

@dataclass
class DatasetRankResult:
    """Résultat unifié du classement d'un candidat parmi les concurrents d'un dataset."""

    position: int            # rang du candidat (1 = meilleur)
    total: int               # nombre total de CVs classés (concurrents + candidat)
    ranking_ids: list[str]   # ids du meilleur au pire (pour afficher le classement complet)
    analysis: str            # feedback riche sur le candidat (contenu propre à chaque méthode)


@runtime_checkable
class DatasetRanker(Protocol):
    """Interface attendue par le ``CVOptimizerAgent`` pour son signal de rang.

    Convention projet : :meth:`import_model` charge le modèle (coûteux/facturé), puis
    :meth:`rank` fait l'inférence.
    """

    def import_model(self) -> None:
        """Charge le système sous-jacent. À appeler avant :meth:`rank`."""
        ...

    def rank(
        self,
        announcement: str,
        competitor_cvs: list[dict],
        candidate_id: str,
        candidate_text: str,
    ) -> DatasetRankResult:
        """Classe ``candidate_text`` parmi ``competitor_cvs`` face à ``announcement``.

        Args:
            announcement:    Texte complet de l'annonce visée.
            competitor_cvs:  CVs concurrents, dicts ``{"id", "content"}`` (cf. ``import_pdf``).
            candidate_id:    Identifiant à donner au CV candidat dans le classement.
            candidate_text:  Texte du CV candidat à évaluer.
        """
        ...


# ---------------------------------------------------------------------------
# Adaptateur fenêtre glissante (LLM, holistique)
# ---------------------------------------------------------------------------

class SlidingWindowDatasetRanker:
    """Enveloppe :class:`SlidingWindowCVRanker` (rang LLM holistique, coûteux/facturé)."""

    def __init__(
        self,
        window_size: int = 4,
        num_passes: int = 3,
        rate_limiter: Optional["BaseRateLimiter"] = None,
        tpm_limiter: Optional["TokensPerMinuteRateLimiter"] = None,
    ):
        self._ranker = SlidingWindowCVRanker(
            window_size=window_size,
            num_passes=num_passes,
            rate_limiter=rate_limiter,
            tpm_limiter=tpm_limiter,
        )

    def import_model(self) -> None:
        self._ranker.import_model()

    def rank(
        self,
        announcement: str,
        competitor_cvs: list[dict],
        candidate_id: str,
        candidate_text: str,
    ) -> DatasetRankResult:
        cvs = self._ranker.load_cvs(
            competitor_cvs + [{"id": candidate_id, "content": candidate_text}]
        )
        result = self._ranker.run_sliding_window_ranking(announcement, cvs)
        ranking = result.ranked_cvs

        ranking_ids = [cv.id for cv in ranking]
        position = ranking_ids.index(candidate_id) + 1
        analysis = result.justifications.get(candidate_id) or "(aucune)"
        return DatasetRankResult(
            position=position,
            total=len(ranking_ids),
            ranking_ids=ranking_ids,
            analysis=analysis,
        )


# ---------------------------------------------------------------------------
# Adaptateur hybride (présélection mots-clés ml6 puis affinage LLM)
# ---------------------------------------------------------------------------

class HybridDatasetRanker:
    """Enveloppe :class:`HybridMl6SlidingWindowRanker` (présélection ml6 puis affinage LLM).

    Coûteux/facturé (l'affinage par fenêtre glissante appelle le LLM), comme
    :class:`SlidingWindowDatasetRanker` ; l'analyse du candidat combine la justification LLM
    et son score mots-clés ml6.
    """

    def __init__(
        self,
        window_size: int = 4,
        num_passes: int = 3,
        rate_limiter: Optional["BaseRateLimiter"] = None,
        tpm_limiter: Optional["TokensPerMinuteRateLimiter"] = None,
    ):
        self._ranker = HybridMl6SlidingWindowRanker(
            window_size=window_size,
            num_passes=num_passes,
            rate_limiter=rate_limiter,
            tpm_limiter=tpm_limiter,
        )

    def import_model(self) -> None:
        self._ranker.import_model()

    def rank(
        self,
        announcement: str,
        competitor_cvs: list[dict],
        candidate_id: str,
        candidate_text: str,
    ) -> DatasetRankResult:
        all_cvs = competitor_cvs + [{"id": candidate_id, "content": candidate_text}]
        result = self._ranker.rank(announcement, all_cvs)

        ranking_ids = [cv_id for cv_id, _ in result.ranked]
        position = ranking_ids.index(candidate_id) + 1

        justification = result.justifications.get(candidate_id) or "(aucune)"
        ml6_score = result.ml6_scores.get(candidate_id)
        analysis = f"Score mots-clés ml6 : {ml6_score}/100.\nAnalyse LLM : {justification}"
        return DatasetRankResult(
            position=position,
            total=len(ranking_ids),
            ranking_ids=ranking_ids,
            analysis=analysis,
        )


# ---------------------------------------------------------------------------
# Adaptateurs basés sur un score par CV (mots-clés, embeddings)
# ---------------------------------------------------------------------------

# Nombre de mots-clés présents/manquants affichés dans l'analyse (échantillon trié).
_KEYWORD_SAMPLE = 15


class _KeywordDatasetRanker:
    """Base commune aux rankers mots-clés (baseline / ml6).

    Score chaque CV par proportion de mots-clés de l'annonce retrouvés, puis trie. L'analyse du
    candidat expose les mots-clés présents/manquants — un signal direct pour une réécriture
    honnête (alignement du vocabulaire sans inventer de qualification).
    """

    def __init__(self, matcher_cls: type) -> None:
        self._matcher_cls = matcher_cls
        self._matcher = None

    def import_model(self) -> None:
        self._matcher = self._matcher_cls()
        self._matcher.import_model()

    def rank(
        self,
        announcement: str,
        competitor_cvs: list[dict],
        candidate_id: str,
        candidate_text: str,
    ) -> DatasetRankResult:
        if self._matcher is None:
            raise RuntimeError("Appelez import_model() avant rank().")

        offre_kw = self._matcher.extract_keywords(announcement)
        all_cvs = competitor_cvs + [{"id": candidate_id, "content": candidate_text}]

        scored: list[tuple[str, float, dict]] = []
        for cv in all_cvs:
            result = self._matcher.match(offre_kw, self._matcher.extract_keywords(cv["content"]))
            scored.append((cv["id"], result["score"], result))

        # Tri par score décroissant (meilleur en premier).
        scored.sort(key=lambda item: item[1], reverse=True)
        ranking_ids = [cv_id for cv_id, _, _ in scored]
        position = ranking_ids.index(candidate_id) + 1

        candidate_match = next(res for cv_id, _, res in scored if cv_id == candidate_id)
        analysis = self._format_analysis(candidate_match)
        return DatasetRankResult(
            position=position,
            total=len(ranking_ids),
            ranking_ids=ranking_ids,
            analysis=analysis,
        )

    @staticmethod
    def _format_analysis(match: dict) -> str:
        matching = sorted(match["matching"])[:_KEYWORD_SAMPLE]
        missing = sorted(match["missing"])[:_KEYWORD_SAMPLE]
        return (
            f"Score mots-clés : {match['score']}/100 "
            f"({len(match['matching'])} présents, {len(match['missing'])} manquants).\n"
            f"Mots-clés de l'annonce présents : {', '.join(matching) or '(aucun)'}\n"
            f"Mots-clés de l'annonce manquants : {', '.join(missing) or '(aucun)'}"
        )


class EmbeddingDatasetRanker:
    """Enveloppe :class:`EmbeddingCosineScorer` : score chaque CV par similarité cosinus, puis trie."""

    def __init__(self) -> None:
        self._scorer: Optional[EmbeddingCosineScorer] = None

    def import_model(self) -> None:
        self._scorer = EmbeddingCosineScorer()
        self._scorer.import_model()

    def rank(
        self,
        announcement: str,
        competitor_cvs: list[dict],
        candidate_id: str,
        candidate_text: str,
    ) -> DatasetRankResult:
        if self._scorer is None:
            raise RuntimeError("Appelez import_model() avant rank().")

        all_cvs = competitor_cvs + [{"id": candidate_id, "content": candidate_text}]
        scored = [(cv["id"], self._scorer.score(announcement, cv["content"])) for cv in all_cvs]
        scored.sort(key=lambda item: item[1], reverse=True)

        ranking_ids = [cv_id for cv_id, _ in scored]
        position = ranking_ids.index(candidate_id) + 1
        candidate_score = next(score for cv_id, score in scored if cv_id == candidate_id)
        best_score = scored[0][1]

        analysis = (
            f"Similarité cosinus avec l'annonce : {candidate_score}/100.\n"
            f"Meilleur concurrent : {best_score}/100 "
            f"(écart : {round(best_score - candidate_score, 1)})."
        )
        return DatasetRankResult(
            position=position,
            total=len(ranking_ids),
            ranking_ids=ranking_ids,
            analysis=analysis,
        )


# ---------------------------------------------------------------------------
# Registre + factory
# ---------------------------------------------------------------------------

def _build_sliding_window(*, rate_limiter, tpm_limiter, window_size, num_passes) -> DatasetRanker:
    return SlidingWindowDatasetRanker(
        window_size=window_size,
        num_passes=num_passes,
        rate_limiter=rate_limiter,
        tpm_limiter=tpm_limiter,
    )


def _build_hybrid(*, rate_limiter, tpm_limiter, window_size, num_passes) -> DatasetRanker:
    return HybridDatasetRanker(
        window_size=window_size,
        num_passes=num_passes,
        rate_limiter=rate_limiter,
        tpm_limiter=tpm_limiter,
    )


# Mappe un nom de ranker → fabrique. Les fabriques acceptent les mêmes kwargs ; ceux qui ne
# concernent pas une méthode donnée (fenêtre glissante) sont simplement ignorés.
CV_OPTIMIZER_RANKERS = {
    "sliding_window": _build_sliding_window,
    "hybrid_ml6_sliding_window": _build_hybrid,
    "baseline_keyword": lambda **_: _KeywordDatasetRanker(BaselineKeywordMatcher),
    "ml6_keyword": lambda **_: _KeywordDatasetRanker(Ml6KeywordMatcher),
    "embedding_cosine": lambda **_: EmbeddingDatasetRanker(),
}


def build_dataset_ranker(
    name: str,
    *,
    rate_limiter: Optional["BaseRateLimiter"] = None,
    tpm_limiter: Optional["TokensPerMinuteRateLimiter"] = None,
    window_size: int = 4,
    num_passes: int = 3,
) -> DatasetRanker:
    """Construit le :class:`DatasetRanker` correspondant à ``name`` (clé de ``CV_OPTIMIZER_RANKERS``).

    Args:
        name:         Nom du ranker (cf. ``CV_OPTIMIZER_RANKER`` dans ``config.py``).
        rate_limiter: Limiteur RPS partagé, utilisé uniquement par les rankers LLM ; ignoré par les autres.
        tpm_limiter:  Limiteur TPM partagé, utilisé uniquement par les rankers LLM ; ignoré par les autres.
        window_size:  Taille de fenêtre (``sliding_window`` uniquement).
        num_passes:   Nombre de passes (``sliding_window`` uniquement).

    Raises:
        ValueError: si ``name`` n'est pas une clé connue.
    """
    try:
        factory = CV_OPTIMIZER_RANKERS[name]
    except KeyError:
        valides = ", ".join(sorted(CV_OPTIMIZER_RANKERS))
        raise ValueError(
            f"Ranker inconnu : {name!r}. Valeurs possibles : {valides}."
        ) from None
    return factory(rate_limiter=rate_limiter, tpm_limiter=tpm_limiter, window_size=window_size, num_passes=num_passes)
