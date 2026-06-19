"""Agent d'optimisation de CV face à une annonce (LangChain / LangGraph + Mistral).

Mission de l'agent : réécrire un CV pour qu'il **remonte dans le classement** face aux
autres candidats d'un dataset synthétique, sans inventer de qualifications — uniquement en
reformulant la substance réelle du candidat avec le vocabulaire attendu par l'annonce.

Cas d'usage cible : le CV « à optimiser » (``to_optimize``) produit par le
``SyntheticCVGenerator`` — un candidat excellent sur le fond mais au vocabulaire non aligné,
que les méthodes mots-clés sous-évaluent à tort.

Le signal de feedback de l'agent n'est pas un score isolé mais le **rang compétitif** du CV
parmi les autres CVs du dataset, calculé via ``Ml6KeywordMatcher`` face à l'annonce. L'agent
est un ReAct préfabriqué (``langgraph.prebuilt.create_react_agent``) tournant sous Mistral
(``langchain_mistralai.ChatMistralAI``). La clé API est lue depuis ``.env`` (variable
``MISTRAL_API_KEY``) — jamais codée en dur.
"""

import json
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent

from ats_system.config import CV_OPTIMIZER_MODEL
from ats_system.data import import_pdf
from ats_system.systems import Ml6KeywordMatcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt système
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Tu es un expert en optimisation de CV pour les systèmes ATS (Applicant \
Tracking Systems).

## Mission
On te confie le CV d'un candidat qui, SUR LE FOND, correspond très bien à l'annonce, mais \
dont la rédaction n'emploie pas le vocabulaire attendu : les filtres par mots-clés le \
sous-évaluent donc à tort. Ta mission est de RÉÉCRIRE ce CV pour qu'il REMONTE dans le \
classement face aux autres candidats du dataset.

## Règle absolue
INTERDICTION d'inventer des compétences, expériences, diplômes ou résultats. Tu ne fais que \
REFORMULER la substance réelle déjà présente dans le CV en employant les termes, intitulés \
de poste et technologies attendus par l'annonce (synonymes explicités, vocabulaire aligné). \
Le CV doit rester crédible et honnête.

## Méthode (boucle outillée)
1. Appelle `list_job_keywords` pour connaître le vocabulaire cible de l'annonce.
2. Appelle `rank_cv_in_dataset` sur le CV courant pour mesurer son rang et lire les \
mots-clés MANQUANTS.
3. Réécris le CV en incorporant honnêtement les mots-clés manquants pertinents.
4. Re-mesure avec `rank_cv_in_dataset` : le rang doit s'améliorer. Itère jusqu'à \
stabilisation (plus de gain de rang) ou épuisement des mots-clés légitimement intégrables.

## Sortie finale
Quand le rang est stabilisé, renvoie UNIQUEMENT le texte complet du CV optimisé, sans \
commentaire ni explication autour."""


class CVOptimizerAgent:
    """Agent ReAct (Mistral) qui optimise un CV pour grimper dans le classement d'un dataset.

    Convention du projet : le chargement coûteux/facturé (LLM Mistral, modèle ml6, extraction
    et cache des mots-clés du dataset) est isolé dans :func:`import_model` ; l'inférence se
    fait via :func:`stream` / :func:`optimize`.
    """

    def __init__(
        self,
        dataset_dir: Path,
        announcement_text: str,
        model: str = CV_OPTIMIZER_MODEL,
        temperature: float = 0.4,
        max_iterations: int = 20,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            dataset_dir:       Dossier d'un dataset synthétique (``synthetic_cvs_<...>/``)
                               contenant les CVs concurrents et un ``manifest.json``.
            announcement_text: Texte complet de l'annonce visée.
            model:             Identifiant du modèle Mistral. Défaut : ``CV_OPTIMIZER_MODEL``.
            temperature:       Température d'échantillonnage du modèle.
            max_iterations:    Limite de récursion du graphe (``recursion_limit``) — borne le
                               nombre d'étapes (appels LLM + outils) avant arrêt forcé.
            api_key:           Clé API Mistral. À défaut, ``MISTRAL_API_KEY`` (chargée de ``.env``).
        """
        self.dataset_dir = Path(dataset_dir)
        self.announcement_text = announcement_text
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self._api_key = api_key

        self._llm: Optional[ChatMistralAI] = None
        self._matcher: Optional[Ml6KeywordMatcher] = None
        self._agent = None
        # Cache rempli par import_model() :
        self._job_keywords: set[str] = set()
        self._dataset_scores: list[tuple[str, float]] = []  # (nom_fichier, score) des concurrents

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def import_model(self) -> None:
        """Charge le LLM, le matcher ml6, met en cache les mots-clés du dataset et bâtit l'agent.

        Coûteux : une extraction ml6 par CV concurrent + l'annonce. Fait une seule fois ;
        la boucle de feedback ne refait ensuite qu'une extraction (sur le brouillon courant)
        par appel d'outil.
        """
        load_dotenv()
        key = self._api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise EnvironmentError(
                "Aucune clé API trouvée. Renseignez MISTRAL_API_KEY dans un fichier .env "
                "(voir .env.example) ou passez api_key=."
            )

        self._llm = ChatMistralAI(model=self.model, temperature=self.temperature, api_key=key)

        # Modèle ml6 partagé entre l'annonce et tous les CVs.
        self._matcher = Ml6KeywordMatcher()
        self._matcher.import_model()

        # Mots-clés de l'annonce (vocabulaire cible) — extraits une fois.
        self._job_keywords = self._matcher.extract_keywords(self.announcement_text)
        logger.info("Mots-clés de l'annonce extraits : %d", len(self._job_keywords))

        # Score ml6 de chaque CV concurrent (tous sauf le « à optimiser »).
        self._dataset_scores = self._score_dataset_competitors()
        logger.info("CVs concurrents scorés : %d", len(self._dataset_scores))

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

        initial = HumanMessage(content=f"Voici le CV à optimiser :\n\n{cv_text}")
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

        initial = HumanMessage(content=f"Voici le CV à optimiser :\n\n{cv_text}")
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

    def _score_dataset_competitors(self) -> list[tuple[str, float]]:
        """Charge les CVs concurrents du dataset et calcule leur score ml6 face à l'annonce.

        Le CV « à optimiser » (entrée ``optimize: true`` du manifest) est exclu : c'est le
        candidat dont l'agent fait justement varier le rang.
        """
        manifest_path = self.dataset_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        scores: list[tuple[str, float]] = []
        for entry in manifest["cvs"]:
            if entry.get("optimize"):
                continue
            cv_path = self.dataset_dir / entry["file"]
            cv_text = import_pdf(str(cv_path))["content"]
            keywords = self._matcher.extract_keywords(cv_text)
            score = self._matcher.match(self._job_keywords, keywords)["score"]
            scores.append((entry["file"], score))
        return scores

    def _build_tools(self) -> list:
        """Construit les outils LangChain (closures sur le cache du dataset)."""

        @tool
        def list_job_keywords() -> str:
            """Liste les mots-clés (vocabulaire cible) attendus par l'annonce.

            À utiliser pour savoir quels termes incorporer honnêtement dans le CV.
            """
            return "Mots-clés attendus par l'annonce :\n" + ", ".join(sorted(self._job_keywords))

        @tool
        def rank_cv_in_dataset(cv_text: str) -> str:
            """Mesure le rang du CV fourni parmi les candidats du dataset, face à l'annonce.

            Calcule le score mots-clés (ml6) du CV, l'insère dans le classement des autres
            candidats et renvoie : son rang, son score, les scores des concurrents et la liste
            des mots-clés MANQUANTS (ceux de l'annonce absents du CV) — le signal le plus utile
            pour la réécriture.

            Args:
                cv_text: Texte complet du CV à évaluer.
            """
            cv_keywords = self._matcher.extract_keywords(cv_text)
            match = self._matcher.match(self._job_keywords, cv_keywords)
            cv_score = match["score"]

            # Classement : concurrents + candidat courant, du meilleur au pire.
            ranking = sorted(
                self._dataset_scores + [("CV_optimisé", cv_score)],
                key=lambda pair: pair[1],
                reverse=True,
            )
            position = next(i for i, (name, _) in enumerate(ranking, 1) if name == "CV_optimisé")
            total = len(ranking)

            competitors = "\n".join(
                f"  #{i}  {name}  (score {score})" for i, (name, score) in enumerate(ranking, 1)
            )
            missing = ", ".join(sorted(match["missing"])) or "(aucun)"
            return (
                f"Rang du CV optimisé : {position}/{total}  (score {cv_score})\n"
                f"Classement complet :\n{competitors}\n\n"
                f"Mots-clés MANQUANTS à intégrer honnêtement : {missing}"
            )

        return [list_job_keywords, rank_cv_in_dataset]
