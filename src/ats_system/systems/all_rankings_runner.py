"""Orchestrateur : classe les CVs via les quatre méthodes en un seul run.

Lance, sur le même jeu de données (annonce + CVs chargés une fois), les quatre systèmes
ATS du projet et regroupe leurs sorties dans un unique dossier horodaté
``results/all_rankings/<horodatage>/`` (un JSON par méthode, schéma commun de
``ats_system.results_io``) :

  - ``baseline_keyword_match`` : mots-clés (regex + stopwords FR/EN) ;
  - ``ml6_keyword_match``       : mots-clés (modèle ml6team) ;
  - ``embedding_cosine``        : similarité cosinus (all-MiniLM-L6-v2) ;
  - ``sliding_window_ranking``  : classement LLM par fenêtre glissante (Claude ou Mistral).

Convention du projet : ``import_model()`` (chargement des quatre systèmes) puis ``run()``.
"""

import logging
from pathlib import Path
from typing import Optional

from ats_system.config import DEFAULT_ANNOUNCEMENT, DEFAULT_CV_DIR, ML6_KEYWORD_MODEL, SLIDING_WINDOW_MODEL
from ats_system.data import load_announcement, load_cvs
from ats_system.results_io import build_ranking, save_results, timestamped_run_dir
from ats_system.systems.baseline_keyword_match import BaselineKeywordMatcher
from ats_system.systems.embedding_cosine import EmbeddingCosineScorer
from ats_system.systems.embedding_cosine import MODEL_NAME as EMBEDDING_MODEL
from ats_system.systems.ml6_keyword_match import Ml6KeywordMatcher
from ats_system.systems.sliding_window_ranker import SlidingWindowCVRanker

logger = logging.getLogger(__name__)

METHOD = "all_rankings"


class AllRankingsRunner:
    """Lance les quatre méthodes de classement et sauvegarde leurs résultats ensemble."""

    def __init__(
        self,
        window_size: int = 4,
        num_passes: int = 10,
        model: str = SLIDING_WINDOW_MODEL,
    ):
        """
        Args:
            window_size: Fenêtre glissante : CVs comparés par appel LLM.
            num_passes:  Fenêtre glissante : nombre maximum de passes.
            model:       Modèle de la fenêtre glissante (fournisseur déduit du préfixe).
        """
        self.window_size = window_size
        self.num_passes = num_passes
        self.model = model
        self._baseline = BaselineKeywordMatcher()
        self._ml6 = Ml6KeywordMatcher()
        self._embedding = EmbeddingCosineScorer()
        self._sliding = SlidingWindowCVRanker(
            window_size=window_size, num_passes=num_passes, model=model
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def import_model(self) -> None:
        """Charge les quatre systèmes sous-jacents. À appeler avant ``run()``."""
        self._baseline.import_model()
        self._ml6.import_model()
        self._embedding.import_model()
        self._sliding.import_model()

    def run(
        self,
        *,
        limit: Optional[int] = None,
        announcement: Path = DEFAULT_ANNOUNCEMENT,
        cv_dir: Path = DEFAULT_CV_DIR,
        save: bool = True,
    ) -> dict:
        """Pipeline complet : charge les données une fois, lance les 4 méthodes, sauvegarde.

        Args:
            limit:        Nombre maximum de CVs à classer (``None``/``0`` = tous).
            announcement: PDF de l'annonce (défaut : annonce par défaut du projet).
            cv_dir:       Dossier contenant les CVs PDF.
            save:         Si vrai, écrit un JSON par méthode sous
                          ``results/all_rankings/<horodatage>/``.

        Returns:
            Dict des résultats par méthode : ``{"baseline", "ml6", "embedding"}`` (listes
            ``(cv_id, score)``) et ``"sliding_window"`` (``RankingResult``).
        """
        print("Chargement des modèles...")
        self.import_model()

        print("Extraction du texte de l'annonce...")
        offre = load_announcement(announcement)
        offre_text = offre["content"]
        cvs = load_cvs(cv_dir, limit)
        print(f"{len(cvs)} CVs chargés.")

        run_dir = timestamped_run_dir(METHOD) if save else None
        base_params = {
            "announcement": Path(announcement).name,
            "cv_dir": str(cv_dir),
            "limit": limit if limit is not None else 0,
            "num_cvs": len(cvs),
        }

        # --- 1. Mots-clés baseline ---
        print("\n[1/4] Mots-clés baseline...")
        baseline_scored = self._baseline.score_cvs(offre_text, cvs)
        if save:
            self._save(run_dir, "baseline_keyword_match", build_ranking(baseline_scored), dict(base_params))

        # --- 2. Mots-clés ml6team ---
        print("[2/4] Mots-clés ml6team...")
        ml6_scored = self._ml6.score_cvs(offre_text, cvs)
        if save:
            self._save(run_dir, "ml6_keyword_match", build_ranking(ml6_scored), {**base_params, "model": ML6_KEYWORD_MODEL})

        # --- 3. Embeddings ---
        print("[3/4] Embeddings...")
        emb_scored = self._embedding.score_cvs(offre_text, cvs)
        if save:
            self._save(run_dir, "embedding_cosine", build_ranking(emb_scored), {**base_params, "model": EMBEDDING_MODEL})

        # --- 4. Fenêtre glissante (LLM) ---
        print("[4/4] Fenêtre glissante (appels LLM)...")
        cv_objs = self._sliding.load_cvs(cvs)
        sw_result = self._sliding.run_sliding_window_ranking(job_offer=offre_text, cvs=cv_objs)
        if save:
            sw_scored = [(cv.id, sw_result.scores[cv.id]) for cv in sw_result.ranked_cvs]
            self._save(
                run_dir,
                "sliding_window_ranking",
                build_ranking(sw_scored, sw_result.justifications),
                {**base_params, "model": self.model, "window_size": self.window_size, "passes": self.num_passes},
                extra={"passes": sw_result.passes, "converged": sw_result.converged},
            )
            print(f"\nTous les classements sauvegardés dans : {run_dir}")

        return {
            "baseline": baseline_scored,
            "ml6": ml6_scored,
            "embedding": emb_scored,
            "sliding_window": sw_result,
        }

    # ------------------------------------------------------------------
    # Helper interne
    # ------------------------------------------------------------------

    @staticmethod
    def _save(run_dir: Path, method: str, ranking: list[dict], params: dict, extra: Optional[dict] = None) -> None:
        """Sauvegarde une méthode dans le dossier de run partagé (noms de fichier propres)."""
        out = save_results(method, ranking, params, extra=extra, results_dir=run_dir, stamp_filename=False)
        print(f"  -> {out.name}")
