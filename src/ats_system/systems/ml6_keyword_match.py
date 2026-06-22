"""Système ATS par mots-clés via le modèle ml6team (BERT, token-classification).

L'extraction des mots-clés passe par le modèle HF
``ml6team/keyphrase-extraction-kbir-inspec`` (limite 512 tokens : le texte est
découpé en chunks de 400 mots). Le score d'adéquation suit la même logique que la
baseline : proportion des mots-clés de l'offre retrouvés dans le CV.
"""

from pathlib import Path
from typing import Optional

from transformers import pipeline

from ats_system.config import DEFAULT_ANNOUNCEMENT, DEFAULT_CV_DIR, ML6_KEYWORD_MODEL
from ats_system.data import load_announcement, load_cvs
from ats_system.results_io import build_ranking, save_results, timestamped_run_dir

MODEL_NAME = ML6_KEYWORD_MODEL
# BERT : max 512 tokens (~400 mots) ; on découpe le texte pour couvrir le document entier.
CHUNK_SIZE = 400
METHOD = "ml6_keyword_match"


class Ml6KeywordMatcher:
    """Score d'adéquation offre/CV par intersection de mots-clés (modèle ml6team)."""

    def __init__(self) -> None:
        self.model = None

    def import_model(self) -> None:
        """Charge la pipeline d'extraction de keyphrases ml6team. À appeler avant usage."""
        self.model = pipeline(
            "token-classification",
            model=MODEL_NAME,
            aggregation_strategy="simple",
        )

    def extract_keywords(self, text: str) -> set:
        """Extrait les mots-clés d'un texte (chunks de 400 mots, dédup, minuscules)."""
        if self.model is None:
            raise RuntimeError("Appelez import_model() avant d'extraire des mots-clés.")

        words = text.split()
        chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

        keywords: set[str] = set()
        for chunk in chunks:
            for entity in self.model(chunk):
                if entity["entity_group"] == "KEY":
                    keywords.add(entity["word"].strip().lower())
        return keywords

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
        cv_dir: Path = DEFAULT_CV_DIR,
        save: bool = True,
    ) -> list[tuple[str, float]]:
        """Pipeline complet : chargement du modèle, des données, scoring et sauvegarde.

        Args:
            limit:        Nombre maximum de CVs à traiter (``None``/``0`` = tous).
            announcement: PDF de l'annonce (défaut : annonce par défaut du projet).
            cv_dir:       Dossier contenant les CVs PDF.
            save:         Si vrai, écrit le classement sous ``results/<METHOD>/<horodatage>/``.

        Returns:
            Le classement, paires ``(cv_id, score)`` triées décroissant.
        """
        print("Chargement du modèle...")
        self.import_model()
        offre = load_announcement(announcement)
        cvs = load_cvs(cv_dir, limit)
        scored = self.score_cvs(offre["content"], cvs)

        for cv_id, score in scored:
            print(f"{score:5.1f}%  {cv_id}")

        if save:
            params = {
                "announcement": Path(announcement).name,
                "cv_dir": str(cv_dir),
                "model": MODEL_NAME,
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
