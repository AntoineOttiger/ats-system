"""Agent d'optimisation de CV face à une annonce (LangChain / LangGraph + Mistral).

Mission de l'agent : réécrire un CV pour qu'il **remonte dans le classement** face aux
autres candidats d'un dataset synthétique, sans inventer de qualifications — uniquement en
reformulant et en mettant en valeur la substance réelle du candidat face à l'annonce.

Le signal de feedback de l'agent n'est pas un score mots-clés isolé mais le **rang
compétitif** du CV parmi les autres CVs du dataset, jugé holistiquement par un LLM via
``SlidingWindowCVRanker`` (classement par fenêtre glissante face à l'annonce). L'agent reçoit
l'annonce complète et **décide seul** comment réécrire le CV — il n'y a pas d'outil de
vocabulaire imposé. L'agent est un ReAct préfabriqué
(``langgraph.prebuilt.create_react_agent``) tournant sous Mistral
(``langchain_mistralai.ChatMistralAI``). La clé API est lue depuis ``.env`` (variable
``MISTRAL_API_KEY``) — jamais codée en dur.

⚠️ Coût : chaque appel de l'outil ``rank_cv_in_dataset`` relance un classement LLM **complet**
de tout le dataset (plusieurs requêtes au modèle du ranker, cf. ``SLIDING_WINDOW_MODEL``).
"""

import json
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent

from ats_system.config import CV_OPTIMIZER_MODEL, LLM_REQUESTS_PER_SECOND
from ats_system.data import import_pdf
from ats_system.systems import SlidingWindowCVRanker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt système
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Tu es un expert en optimisation de CV pour les systèmes ATS (Applicant \
Tracking Systems).

## Mission
On te confie une annonce et le CV d'un candidat. Ta mission est de RÉÉCRIRE ce CV pour qu'il \
REMONTE dans le classement face aux autres candidats du dataset, en l'alignant au mieux avec \
l'annonce.

## Règle absolue
INTERDICTION d'inventer des compétences, expériences, diplômes ou résultats. Tu ne fais que \
REFORMULER et METTRE EN VALEUR la substance réelle déjà présente dans le CV face à l'annonce. \
Le CV doit rester crédible et honnête.

## Méthode (boucle outillée)
1. Analyse l'annonce et le CV : repère les éléments du CV pertinents pour l'annonce et la \
meilleure façon de les formuler.
2. Appelle `rank_cv_in_dataset` sur le CV courant pour mesurer son rang parmi les autres \
candidats et lire l'analyse du recruteur.
3. Réécris le CV pour mieux valoriser les éléments pertinents (sans rien inventer).
4. Re-mesure avec `rank_cv_in_dataset` : le rang doit s'améliorer. Itère jusqu'à \
stabilisation (plus de gain de rang).

## Sortie finale
Quand le rang est stabilisé, renvoie UNIQUEMENT le texte complet du CV optimisé, sans \
commentaire ni explication autour."""


class CVOptimizerAgent:
    """Agent ReAct (Mistral) qui optimise un CV pour grimper dans le classement d'un dataset.

    Le rang du CV est jugé par un LLM via :class:`SlidingWindowCVRanker` (classement par
    fenêtre glissante face à l'annonce), et non par un comptage de mots-clés.

    Convention du projet : le chargement coûteux/facturé (LLM Mistral, ranker fenêtre
    glissante, lecture et cache des CVs concurrents du dataset) est isolé dans
    :func:`import_model` ; l'inférence se fait via :func:`stream` / :func:`optimize`.
    """

    def __init__(
        self,
        dataset_dir: Path,
        announcement_text: str,
        model: str = CV_OPTIMIZER_MODEL,
        temperature: float = 0.4,
        max_iterations: int = 20,
        window_size: int = 4,
        num_passes: int = 3,
        api_key: Optional[str] = None,
        requests_per_second: float = LLM_REQUESTS_PER_SECOND,
    ):
        """
        Args:
            dataset_dir:         Dossier d'un dataset synthétique (``synthetic_cvs_<...>/``)
                                 contenant les CVs concurrents et un ``manifest.json``.
            announcement_text:   Texte complet de l'annonce visée.
            model:               Identifiant du modèle Mistral. Défaut : ``CV_OPTIMIZER_MODEL``.
            temperature:         Température d'échantillonnage du modèle.
            max_iterations:      Limite de récursion du graphe (``recursion_limit``) — borne le
                                 nombre d'étapes (appels LLM + outils) avant arrêt forcé.
            window_size:         Taille de fenêtre du ``SlidingWindowCVRanker`` (signal de rang).
            num_passes:          Nombre de passes du ``SlidingWindowCVRanker``.
            api_key:             Clé API Mistral. À défaut, ``MISTRAL_API_KEY`` (chargée de ``.env``).
            requests_per_second: Débit cible des requêtes LLM (limiteur partagé entre l'agent et
                                 le ranker) pour éviter les 429. Défaut : ``LLM_REQUESTS_PER_SECOND``.
        """
        self.dataset_dir = Path(dataset_dir)
        self.announcement_text = announcement_text
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.window_size = window_size
        self.num_passes = num_passes
        self._api_key = api_key
        self.requests_per_second = requests_per_second

        self._llm: Optional[ChatMistralAI] = None
        self._ranker: Optional[SlidingWindowCVRanker] = None
        self._rate_limiter: Optional[InMemoryRateLimiter] = None
        self._agent = None
        # Cache rempli par import_model() : dicts {"id", "content"} des CVs concurrents.
        self._competitor_cvs: list[dict] = []

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def import_model(self) -> None:
        """Charge le LLM Mistral, le ranker fenêtre glissante, les CVs concurrents et l'agent.

        Le ranker est chargé une fois (``import_model``) ; en revanche, la boucle de feedback
        relance un classement LLM **complet** du dataset à chaque appel d'outil (coûteux).
        Les textes des CVs concurrents sont lus et mis en cache une seule fois ici.
        """
        load_dotenv()
        key = self._api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise EnvironmentError(
                "Aucune clé API trouvée. Renseignez MISTRAL_API_KEY dans un fichier .env "
                "(voir .env.example) ou passez api_key=."
            )

        # Limiteur de débit partagé : un seul token bucket gouverne TOUS les appels
        # Mistral du processus (modèle de l'agent + ranker), garantissant un plafond
        # global en req/s même quand les appels s'alternent. max_bucket_size=1 empêche
        # l'accumulation de tokens et donc les bursts.
        self._rate_limiter = InMemoryRateLimiter(
            requests_per_second=self.requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )

        self._llm = ChatMistralAI(
            model=self.model,
            temperature=self.temperature,
            api_key=key,
            rate_limiter=self._rate_limiter,
            max_retries=5,
        )

        # Ranker fenêtre glissante : signal de rang jugé par LLM (cf. SLIDING_WINDOW_MODEL).
        # Il partage le même limiteur de débit que l'agent.
        self._ranker = SlidingWindowCVRanker(
            window_size=self.window_size,
            num_passes=self.num_passes,
            rate_limiter=self._rate_limiter,
        )
        self._ranker.import_model()

        # Textes des CVs concurrents (tous sauf le « à optimiser ») — lus une fois.
        self._competitor_cvs = self._load_dataset_competitors()
        logger.info("CVs concurrents chargés : %d", len(self._competitor_cvs))

        # Outils (closures sur le cache) + agent ReAct.
        self._agent = create_react_agent(
            self._llm,
            tools=self._build_tools(),
            prompt=SYSTEM_PROMPT,
        )

    def stream(self, cv_text: str) -> Iterator[dict]:
        """Exécute l'agent en streamant ses étapes (pour afficher ses « pensées »).

        Args:
            cv_text: Texte du CV initial à optimiser.

        Yields:
            Les mises à jour ``stream_mode="updates"`` du graphe LangGraph : un dict
            ``{nom_du_nœud: {"messages": [...]}}`` par étape (raisonnement, appels d'outils,
            observations).
        """
        if self._agent is None:
            raise RuntimeError("Appelez import_model() avant de lancer l'agent.")

        initial = HumanMessage(
            content=(
                f"## Annonce visée\n{self.announcement_text}\n\n"
                f"## CV à optimiser\n{cv_text}"
            )
        )
        yield from self._agent.stream(
            {"messages": [initial]},
            config={"recursion_limit": self.max_iterations},
            stream_mode="updates",
        )

    def optimize(self, cv_text: str) -> str:
        """Exécute l'agent et retourne le texte du CV optimisé final.

        Args:
            cv_text: Texte du CV initial à optimiser.

        Returns:
            Le contenu du dernier message de l'assistant (le CV optimisé).
        """
        if self._agent is None:
            raise RuntimeError("Appelez import_model() avant de lancer l'agent.")

        initial = HumanMessage(
            content=(
                f"## Annonce visée\n{self.announcement_text}\n\n"
                f"## CV à optimiser\n{cv_text}"
            )
        )
        result = self._agent.invoke(
            {"messages": [initial]},
            config={"recursion_limit": self.max_iterations},
        )
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.content:
                return message.content if isinstance(message.content, str) else str(message.content)
        return ""

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _load_dataset_competitors(self) -> list[dict]:
        """Lit les CVs concurrents du dataset (dicts ``{"id", "content"}`` pour ``load_cvs``).

        Le CV « à optimiser » (entrée ``optimize: true`` du manifest) est exclu : c'est le
        candidat dont l'agent fait justement varier le rang.
        """
        manifest_path = self.dataset_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        competitors: list[dict] = []
        for entry in manifest["cvs"]:
            if entry.get("optimize"):
                continue
            cv_path = self.dataset_dir / entry["file"]
            competitors.append({"id": entry["file"], "content": import_pdf(str(cv_path))["content"]})
        return competitors

    def _build_tools(self) -> list:
        """Construit les outils LangChain (closures sur le cache du dataset)."""

        @tool
        def rank_cv_in_dataset(cv_text: str) -> str:
            """Mesure le rang du CV fourni parmi les candidats du dataset, face à l'annonce.

            Insère le CV parmi les concurrents et lance un classement par fenêtre glissante
            (jugement LLM holistique, pas un comptage de mots-clés). Renvoie son rang, le
            classement complet et l'analyse du recruteur sur ce CV — le signal le plus utile
            pour la réécriture.

            ⚠️ Appel coûteux : relance un classement LLM complet du dataset à chaque fois.

            Args:
                cv_text: Texte complet du CV à évaluer.
            """
            candidate_id = "CV_optimisé"
            cvs = self._ranker.load_cvs(
                self._competitor_cvs + [{"id": candidate_id, "content": cv_text}]
            )
            result = self._ranker.run_sliding_window_ranking(self.announcement_text, cvs)
            ranking = result.ranked_cvs

            position = next(i for i, cv in enumerate(ranking, 1) if cv.id == candidate_id)
            total = len(ranking)
            classement = "\n".join(f"  #{i}  {cv.id}" for i, cv in enumerate(ranking, 1))
            analyse = result.justifications.get(candidate_id, "(aucune)")
            return (
                f"Rang du CV optimisé : {position}/{total}\n"
                f"Classement complet (du meilleur au pire) :\n{classement}\n\n"
                f"Analyse du recruteur sur ce CV : {analyse}"
            )

        return [rank_cv_in_dataset]
