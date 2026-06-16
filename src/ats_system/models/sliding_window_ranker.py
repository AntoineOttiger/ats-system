"""Classement de CVs par fenêtre glissante (inspiré de RankGPT).

Chaque fenêtre contient un petit sous-ensemble de CVs que le LLM reclasse
localement. Plusieurs passes sont effectuées jusqu'à stabilisation du classement.
Le modèle Claude est appelé via le SDK officiel ``anthropic`` ; la clé API est
chargée depuis un fichier ``.env`` (variable ``ANTHROPIC_API_KEY``) — jamais
codée en dur.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class CV:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RankingResult:
    ranked_cvs: list[CV]
    scores: dict[str, float]          # cv_id -> score basé sur la position finale
    justifications: dict[str, str]    # cv_id -> dernière justification du LLM
    passes: int                       # nombre de passes effectuées
    converged: bool


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class SlidingWindowCVRanker:
    """Classe des CVs face à une offre d'emploi par fenêtre glissante.

    Algorithme (façon RankGPT) :
      1. Partir d'un ordre arbitraire des CVs.
      2. Faire glisser une fenêtre de ``window_size`` CVs sur la liste.
      3. Demander au LLM de reclasser les CVs de la fenêtre face à l'offre.
      4. Réinjecter le résultat local dans l'ordre global.
      5. Répéter pendant ``num_passes`` passes (ou jusqu'à convergence).
    """

    MODEL = "claude-opus-4-8"

    def __init__(
        self,
        window_size: int = 4,
        step_size: Optional[int] = None,
        num_passes: int = 3,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            window_size: Nombre de CVs comparés à chaque appel LLM (3-5 recommandé).
            step_size:   Décalage de la fenêtre à chaque étape. Défaut : window_size // 2.
            num_passes:  Nombre maximum de passes complètes sur la liste.
            max_tokens:  Nombre maximum de tokens par réponse du LLM.
            api_key:     Clé API Anthropic. À défaut, la variable d'environnement
                         ``ANTHROPIC_API_KEY`` (chargée depuis ``.env``) est utilisée.
        """
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else max(1, window_size // 2)
        self.num_passes = num_passes
        self.max_tokens = max_tokens
        self._api_key = api_key
        self.client: Optional[anthropic.Anthropic] = None

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def import_model(self) -> None:
        """Initialise le client Anthropic. À appeler avant tout classement.

        La clé API est lue depuis l'argument ``api_key`` ou, à défaut, depuis la
        variable d'environnement ``ANTHROPIC_API_KEY`` chargée d'un fichier ``.env``.
        """
        load_dotenv()
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError(
                "Aucune clé API trouvée. Renseignez ANTHROPIC_API_KEY dans un fichier .env "
                "(voir .env.example) ou passez api_key=."
            )
        self.client = anthropic.Anthropic(api_key=key)
        logger.info("Client Anthropic initialisé (modèle : %s)", self.MODEL)

    def load_cvs(self, cvs: list[dict]) -> list[CV]:
        """Convertit des dicts bruts en objets CV.

        Format de dict attendu (compatible ``import_pdf``) :
            {"id": "12472574.pdf", "content": "Texte du CV", "metadata": {...}}
        """
        return [
            CV(
                id=c["id"],
                content=c["content"],
                metadata=c.get("metadata", {}),
            )
            for c in cvs
        ]

    def run_sliding_window_ranking(
        self,
        job_offer: str,
        cvs: list[CV],
    ) -> RankingResult:
        """Point d'entrée principal. Exécute l'algorithme de fenêtre glissante.

        Args:
            job_offer: Texte complet de l'offre d'emploi.
            cvs:       Liste des objets CV à classer.

        Returns:
            Un ``RankingResult`` avec la liste ordonnée finale et ses métadonnées.
        """
        if self.client is None:
            raise RuntimeError("Appelez import_model() avant de lancer le ranker.")
        if len(cvs) < 2:
            raise ValueError("Il faut au moins 2 CVs à classer.")

        ranked = list(cvs)           # copie de travail, mutée en place à chaque passe
        justifications: dict[str, str] = {}
        prev_order: list[str] = []
        converged = False

        for pass_idx in range(self.num_passes):
            logger.info("Passe %d/%d", pass_idx + 1, self.num_passes)
            ranked = self._single_pass(job_offer, ranked, justifications)

            current_order = [cv.id for cv in ranked]
            if current_order == prev_order:
                logger.info("Classement convergé après %d passe(s).", pass_idx + 1)
                converged = True
                break
            prev_order = current_order

        # Score basé sur la position (rang 1 = score le plus élevé)
        n = len(ranked)
        scores = {cv.id: round((n - i) / n, 4) for i, cv in enumerate(ranked)}

        return RankingResult(
            ranked_cvs=ranked,
            scores=scores,
            justifications=justifications,
            passes=pass_idx + 1,
            converged=converged,
        )

    def display_results(self, result: RankingResult) -> None:
        """Affiche le classement de façon lisible."""
        print("\n" + "=" * 60)
        print("RÉSULTATS DU CLASSEMENT")
        print(f"Passes : {result.passes}  |  Convergé : {result.converged}")
        print("=" * 60)
        for rank, cv in enumerate(result.ranked_cvs, 1):
            score = result.scores[cv.id]
            justification = result.justifications.get(cv.id, "-")
            print(f"\n#{rank}  [{cv.id}]  score={score}")
            print(f"  Justification : {justification[:200]}{'...' if len(justification) > 200 else ''}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _single_pass(
        self,
        job_offer: str,
        ranked: list[CV],
        justifications: dict[str, str],
    ) -> list[CV]:
        """Une passe complète de la fenêtre glissante sur la liste classée."""
        n = len(ranked)
        start = 0

        while start < n:
            end = min(start + self.window_size, n)
            window = ranked[start:end]

            if len(window) >= 2:
                new_order, window_justifications = self._rank_window(job_offer, window)
                # Réinsère la fenêtre reclassée dans la liste
                ranked[start:end] = new_order
                justifications.update(window_justifications)
                logger.debug(
                    "  Fenêtre [%d:%d] -> %s",
                    start, end, [cv.id for cv in new_order]
                )

            start += self.step_size

        return ranked

    def _rank_window(
        self,
        job_offer: str,
        window: list[CV],
    ) -> tuple[list[CV], dict[str, str]]:
        """Demande au LLM de classer une petite fenêtre de CVs face à l'offre.

        Retourne la fenêtre réordonnée et les justifications par CV.
        """
        prompt = self._build_prompt(job_offer, window)

        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )

        # La réponse peut contenir des blocs thinking : on isole le bloc texte.
        raw = next((block.text for block in response.content if block.type == "text"), "")
        return self._parse_response(raw, window)

    def _build_prompt(self, job_offer: str, window: list[CV]) -> str:
        cv_blocks = "\n\n".join(
            f"--- CV [{cv.id}] ---\n{cv.content}" for cv in window
        )
        ids = [cv.id for cv in window]

        return f"""You are an expert technical recruiter. Your task is to rank the following CVs from MOST to LEAST suitable for the job offer below.

## Job Offer
{job_offer}

## CVs to rank
{cv_blocks}

## Instructions
- Rank the CVs strictly from best to worst fit for this specific job offer.
- Consider: required skills match, relevant experience, education level, domain alignment.
- Return ONLY a valid JSON object. No markdown, no explanation outside the JSON.

## Required JSON format
{{
  "ranking": ["{ids[0]}", ...],  // ordered list of CV IDs, best first
  "justifications": {{
    "{ids[0]}": "one sentence explaining this CV's fit",
    ...
  }}
}}
"""

    def _parse_response(
        self,
        raw: str,
        window: list[CV],
    ) -> tuple[list[CV], dict[str, str]]:
        """Parse la réponse JSON du LLM et retourne la fenêtre réordonnée."""
        window_map = {cv.id: cv for cv in window}

        # Isole l'objet JSON même s'il est entouré de texte ou de fences markdown.
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            logger.warning("Aucun JSON trouvé dans la réponse du LLM. Ordre original conservé.")
            logger.debug("Réponse brute : %s", raw)
            return window, {}

        try:
            data = json.loads(raw[start:end + 1])
            ranking: list[str] = data["ranking"]
            justifications: dict[str, str] = data.get("justifications", {})

            # Ne garde que les IDs réellement présents dans la fenêtre (sécurité)
            valid_ids = [rid for rid in ranking if rid in window_map]
            # Ajoute en fin tout ID manquant (ne devrait pas arriver, mais défensif)
            missing = [cv.id for cv in window if cv.id not in valid_ids]
            ordered_ids = valid_ids + missing

            return [window_map[rid] for rid in ordered_ids], justifications

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Échec du parsing de la réponse LLM (%s). Ordre original conservé.", e)
            logger.debug("Réponse brute : %s", raw)
            return window, {}
