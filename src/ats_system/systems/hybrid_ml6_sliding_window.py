"""Système ATS hybride : présélection mots-clés (ml6) puis affinage LLM (fenêtre glissante).

Pipeline en deux étapes, du moins coûteux au plus coûteux :

  1. **Première passe — ``Ml6KeywordMatcher``** : score local et gratuit de tous les
     CVs par intersection de mots-clés avec l'annonce. Sert de présélection rapide et
     fournit un ordre de départ pertinent pour l'étape suivante.
  2. **Affinage — ``SlidingWindowCVRanker``** : un LLM (modèle ``SLIDING_WINDOW_MODEL``)
     reclasse finement, par fenêtre glissante, tous les CVs (coûteux/facturé) en partant
     de l'ordre mots-clés de la première passe.

L'intérêt : donner au LLM un point de départ pertinent (ordre mots-clés) pour un
classement complet plus fiable. Convention projet : ``import_model()`` (chargement des
deux systèmes sous-jacents) puis ``rank()`` (inférence).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ats_system.config import (
    DEFAULT_ANNOUNCEMENT,
    DEFAULT_CV_CATEGORY,
    ML6_KEYWORD_MODEL,
    SLIDING_WINDOW_MODEL,
)
from ats_system.data import load_announcement, load_cvs
from ats_system.results_io import build_ranking, save_results, timestamped_run_dir
from ats_system.systems.ml6_keyword_match import Ml6KeywordMatcher
from ats_system.systems.sliding_window_ranker import RankingResult, SlidingWindowCVRanker

if TYPE_CHECKING:
    from langchain_core.rate_limiters import BaseRateLimiter

logger = logging.getLogger(__name__)

METHOD = "hybrid_ranking"


def _position_scored(order: list[str]) -> list[tuple[str, float]]:
    """Convertit un ordre de cv_ids en paires ``(cv_id, score)`` basées sur la position."""
    n = len(order)
    return [(cv_id, round((n - i) / n, 4)) for i, cv_id in enumerate(order)]


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------

@dataclass
class HybridRankingResult:
    """Résultat du classement hybride, avec l'historique complet des étapes.

    Attributs :
        ranked: classement final, paires ``(cv_id, score)`` triées du meilleur au pire.
                Le score est basé sur la position finale (rang 1 = score le plus élevé).
        justifications: justification LLM par ``cv_id``.
        ml6_ranking: classement de la première passe mots-clés, paires ``(cv_id, score)``
                     (score 0–100) triées décroissant.
        sliding_window: ``RankingResult`` brut de la fenêtre glissante (contient
                        l'historique par passe : ``pass_orders`` / ``pass_justifications``).
    """

    ranked: list[tuple[str, float]]
    justifications: dict[str, str]
    ml6_ranking: list[tuple[str, float]]
    sliding_window: Optional[RankingResult] = None
    # Scores mots-clés par cv_id (pratique pour l'affichage / la sauvegarde).
    ml6_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class HybridMl6SlidingWindowRanker:
    """Classement hybride : présélection ml6 (mots-clés) puis affinage fenêtre glissante (LLM)."""

    def __init__(
        self,
        window_size: int = 4,
        num_passes: int = 3,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        model: str = SLIDING_WINDOW_MODEL,
        rate_limiter: Optional["BaseRateLimiter"] = None,
    ):
        """
        Args:
            window_size:  Taille de la fenêtre glissante (CVs comparés par appel LLM).
            num_passes:   Nombre maximum de passes de la fenêtre glissante.
            max_tokens:   Tokens maximum par réponse LLM.
            api_key:      Clé API du fournisseur (à défaut : variable d'environnement).
            model:        Modèle de l'affinage (fournisseur déduit du préfixe). Défaut :
                          ``SLIDING_WINDOW_MODEL`` (cf. ``config.py``).
            rate_limiter: Limiteur de débit partagé transmis à la fenêtre glissante.
        """
        self.model = model
        self._ml6 = Ml6KeywordMatcher()
        self._ranker = SlidingWindowCVRanker(
            window_size=window_size,
            num_passes=num_passes,
            max_tokens=max_tokens,
            api_key=api_key,
            model=model,
            rate_limiter=rate_limiter,
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def import_model(self) -> None:
        """Charge les deux systèmes sous-jacents (ml6 puis client LLM). À appeler avant ``rank()``."""
        self._ml6.import_model()
        self._ranker.import_model()

    def rank(self, job_offer: str, cvs: list[dict]) -> HybridRankingResult:
        """Classe ``cvs`` face à ``job_offer`` : présélection mots-clés puis affinage LLM.

        Args:
            job_offer: Texte complet de l'annonce.
            cvs:       CVs à classer, dicts ``{"id", "content"}`` (cf. ``import_pdf``).

        Returns:
            Un ``HybridRankingResult`` (classement final + historique des étapes).
        """
        if len(cvs) < 2:
            raise ValueError("Il faut au moins 2 CVs à classer.")

        # --- Étape 1 : présélection mots-clés (ml6) ---
        logger.info("Première passe mots-clés (ml6) sur %d CVs...", len(cvs))
        offre_kw = self._ml6.extract_keywords(job_offer)
        cv_by_id = {cv["id"]: cv for cv in cvs}
        ml6_scored: list[tuple[str, float]] = [
            (cv["id"], self._ml6.match(offre_kw, self._ml6.extract_keywords(cv["content"]))["score"])
            for cv in cvs
        ]
        ml6_scored.sort(key=lambda x: x[1], reverse=True)
        ml6_scores = dict(ml6_scored)
        ordered_ids = [cv_id for cv_id, _ in ml6_scored]

        # --- Étape 2 : affinage LLM par fenêtre glissante sur tous les CVs ---
        logger.info("Affinage fenêtre glissante (LLM) sur les %d CVs...", len(ordered_ids))
        # On part de l'ordre mots-clés (point de départ pertinent pour le LLM).
        refined_cv_objs = self._ranker.load_cvs([cv_by_id[cid] for cid in ordered_ids])
        sw_result = self._ranker.run_sliding_window_ranking(job_offer, refined_cv_objs)

        # --- Étape 3 : recomposition du classement complet ---
        final_order = [cv.id for cv in sw_result.ranked_cvs]
        n = len(final_order)
        ranked = [(cid, round((n - i) / n, 4)) for i, cid in enumerate(final_order)]

        return HybridRankingResult(
            ranked=ranked,
            justifications=dict(sw_result.justifications),
            ml6_ranking=ml6_scored,
            sliding_window=sw_result,
            ml6_scores=ml6_scores,
        )

    def run(
        self,
        *,
        limit: Optional[int] = None,
        announcement: Path = DEFAULT_ANNOUNCEMENT,
        category: str = DEFAULT_CV_CATEGORY,
        save: bool = True,
    ) -> HybridRankingResult:
        """Pipeline complet : chargement, classement hybride et sauvegarde de l'historique.

        Si ``save``, écrit tout l'historique sous ``results/<METHOD>/<horodatage>/`` :
        présélection mots-clés (``ml6_keyword_match.json``), classement après chaque passe
        de la fenêtre glissante (``sliding_window_pass{N}.json``) et classement final
        (``hybrid_ranking.json``).

        Args:
            limit:        Nombre maximum de CVs à classer (``None``/``0`` = tous).
            announcement: PDF de l'annonce (défaut : annonce par défaut du projet).
            category:     Catégorie de CVs (sous-dossier de ``CV_DIR``).
            save:         Si vrai, écrit tout l'historique de classement.

        Returns:
            Le ``HybridRankingResult`` (classement final + historique des étapes).
        """
        print("Initialisation du système hybride (chargement des modèles)...")
        self.import_model()
        offre = load_announcement(announcement)
        cvs = load_cvs(category, limit)

        print(f"Classement de {len(cvs)} CVs (passe mots-clés puis appels LLM)...")
        result = self.rank(job_offer=offre["content"], cvs=cvs)

        self.display_results(result)

        if save:
            self._save_history(result, announcement, category, limit, len(cvs))
        return result

    def _save_history(
        self,
        result: HybridRankingResult,
        announcement: Path,
        category: str,
        limit: Optional[int],
        num_cvs: int,
    ) -> None:
        """Écrit prescan ml6 + passes de la fenêtre glissante + classement final."""
        run_dir = timestamped_run_dir(METHOD)
        base_params = {
            "announcement": Path(announcement).name,
            "category": category,
            "limit": limit if limit is not None else 0,
            "num_cvs": num_cvs,
        }
        window_size = self._ranker.window_size
        num_passes = self._ranker.num_passes

        # --- 1. Première passe : mots-clés ml6 (présélection) ---
        save_results(
            "ml6_keyword_match",
            build_ranking(result.ml6_ranking),
            {**base_params, "model": ML6_KEYWORD_MODEL, "stage": "preselection"},
            results_dir=run_dir,
            stamp_filename=False,
        )

        # --- 2. Chaque passe de la fenêtre glissante ---
        sw = result.sliding_window
        if sw is not None:
            for i, (order, justifs) in enumerate(zip(sw.pass_orders, sw.pass_justifications), 1):
                save_results(
                    f"sliding_window_pass{i}",
                    build_ranking(_position_scored(order), justifs),
                    {
                        **base_params,
                        "model": self.model,
                        "stage": "refinement",
                        "pass": i,
                        "window_size": window_size,
                    },
                    results_dir=run_dir,
                    stamp_filename=False,
                )

        # --- 3. Classement final ---
        out_path = save_results(
            METHOD,
            build_ranking(result.ranked, result.justifications),
            {
                **base_params,
                "model": self.model,
                "ml6_model": ML6_KEYWORD_MODEL,
                "window_size": window_size,
                "passes": num_passes,
            },
            extra={
                "passes": sw.passes if sw is not None else 0,
                "converged": sw.converged if sw is not None else False,
            },
            results_dir=run_dir,
            stamp_filename=False,
        )

        print(f"\nHistorique complet sauvegardé dans : {run_dir}")
        print(f"Classement final : {out_path.name}")

    def display_results(self, result: HybridRankingResult) -> None:
        """Affiche le classement final de façon lisible."""
        print("\n" + "=" * 60)
        print("CLASSEMENT HYBRIDE (ml6 -> fenêtre glissante)")
        if result.sliding_window is not None:
            print(
                f"Passes : {result.sliding_window.passes}  |  "
                f"Convergé : {result.sliding_window.converged}"
            )
        print("=" * 60)
        for rank, (cv_id, score) in enumerate(result.ranked, 1):
            ml6 = result.ml6_scores.get(cv_id, "-")
            justification = result.justifications.get(cv_id, "-")
            print(f"\n#{rank}  [{cv_id}]  score={score}  (ml6={ml6})")
            print(f"  Justification : {justification[:200]}{'...' if len(justification) > 200 else ''}")
        print("=" * 60)
