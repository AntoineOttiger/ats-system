"""Système ATS par mots-clés baseline (regex + stopwords FR/EN).

Extraction naïve des mots-clés (minuscules, suppression de la ponctuation et des
stopwords), puis score d'adéquation = proportion des mots-clés de l'offre retrouvés
dans le CV.
"""

import re
from pathlib import Path
from typing import Optional

import nltk
from nltk.corpus import stopwords

from ats_system.config import DEFAULT_ANNOUNCEMENT, DEFAULT_CV_CATEGORY
from ats_system.data import load_announcement, load_cvs
from ats_system.results_io import build_ranking, save_results, timestamped_run_dir

METHOD = "baseline_keyword_match"


class BaselineKeywordMatcher:
    """Score d'adéquation offre/CV par intersection de mots-clés (baseline)."""

    def __init__(self) -> None:
        self._stop_words: set[str] = set()

    def import_model(self) -> None:
        """Télécharge les stopwords nltk et pré-calcule le set FR/EN. À appeler avant usage."""
        nltk.download("stopwords", quiet=True)
        self._stop_words = set(stopwords.words("french")) | set(stopwords.words("english"))

    def extract_keywords(self, text: str) -> set:
        """Extrait les mots-clés d'un texte (minuscules, hors ponctuation et stopwords)."""
        text = text.lower()
        # Supprime tout ce qui n'est ni une lettre ni un espace (ex : "3.0").
        text = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", text)
        words = text.split()
        return set(w for w in words if w not in self._stop_words and len(w) > 2)

    @staticmethod
    def match(keywords_offre: set, keywords_cv: set) -> dict:
        """Compare les mots-clés offre/CV → ``{score, matching, missing}`` (score 0–100)."""
        matching = keywords_offre & keywords_cv
        missing = keywords_offre - keywords_cv
        score = round(len(matching) / len(keywords_offre) * 100, 1) if keywords_offre else 0
        return {
            "score": score,
            "matching": matching,
            "missing": missing,
        }

    def score_cvs(self, offre_text: str, cvs: list[dict]) -> list[tuple[str, float]]:
        """Score chaque CV face à l'offre → paires ``(cv_id, score)`` triées décroissant."""
        keywords_offre = self.extract_keywords(offre_text)
        scored = [
            (cv["id"], self.match(keywords_offre, self.extract_keywords(cv["content"]))["score"])
            for cv in cvs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def run(
        self,
        *,
        limit: Optional[int] = None,
        announcement: Path = DEFAULT_ANNOUNCEMENT,
        category: str = DEFAULT_CV_CATEGORY,
        save: bool = True,
    ) -> list[tuple[str, float]]:
        """Pipeline complet : chargement du modèle, des données, scoring et sauvegarde.

        Args:
            limit:        Nombre maximum de CVs à traiter (``None``/``0`` = tous).
            announcement: PDF de l'annonce (défaut : annonce par défaut du projet).
            category:     Catégorie de CVs (sous-dossier de ``CV_DIR``).
            save:         Si vrai, écrit le classement sous ``results/<METHOD>/<horodatage>/``.

        Returns:
            Le classement, paires ``(cv_id, score)`` triées décroissant.
        """
        self.import_model()
        offre = load_announcement(announcement)
        cvs = load_cvs(category, limit)
        scored = self.score_cvs(offre["content"], cvs)

        for cv_id, score in scored:
            print(f"{score:5.1f}%  {cv_id}")

        if save:
            params = {
                "announcement": Path(announcement).name,
                "category": category,
                "limit": limit if limit is not None else 0,
                "num_cvs": len(cvs),
            }
            out = save_results(
                METHOD,
                build_ranking(scored),
                params,
                results_dir=timestamped_run_dir(METHOD),
                stamp_filename=False,
            )
            print(f"\nRésultats sauvegardés dans : {out}")
        return scored
