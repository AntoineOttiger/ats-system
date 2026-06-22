"""Optimiseur de CV « one-shot » face à une annonce (un seul appel LLM, sans boucle).

Point de comparaison (baseline) face à l'agent adversarial
(:class:`~ats_system.agents.cv_optimizer_agent.CVOptimizerAgent`) : là où l'agent ReAct
**itère** sur le feedback de rang d'un ranker jusqu'à stabilisation, cet optimiseur réécrit le
CV « à optimiser » d'un dataset synthétique en **un unique appel LLM**, **à l'aveugle** —
il ne voit que l'annonce et le CV, jamais les CVs concurrents ni le rang.

Pour rendre la comparaison chiffrée possible, le rang du CV est tout de même **mesuré avant
et après** la réécriture (via le ranker choisi par ``CV_OPTIMIZER_RANKER``), de façon
symétrique aux champs ``rang_initial`` / ``rang_final`` du méta de l'agent. Ces mesures ne
sont **pas** transmises au LLM : elles ne servent qu'au rapport.

Convention projet : le chargement coûteux/facturé (client LLM, ranker, cache des CVs
concurrents) est isolé dans :func:`import_model` ; l'inférence se fait via :func:`optimize`.

⚠️ Coût : avec le ranker ``sliding_window``, chaque mesure de rang relance un classement LLM
**complet** du dataset (deux classements ici : avant et après). Les rankers mots-clés /
embeddings sont locaux et gratuits.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter

from ats_system.agents.dataset_loading import find_optimize_cv, load_competitors
from ats_system.agents.dataset_rankers import DatasetRanker, build_dataset_ranker
from ats_system.config import CV_OPTIMIZER_MODEL, CV_OPTIMIZER_RANKER, LLM_REQUESTS_PER_SECOND, LLM_TOKENS_PER_MINUTE
from ats_system.data import write_text_pdf
from ats_system.llm import LLMClient, TokensPerMinuteRateLimiter
from ats_system.results_io import timestamped_run_dir

logger = logging.getLogger(__name__)

METHOD = "cv_optimizer_oneshot"


# ---------------------------------------------------------------------------
# Prompt « aveugle » (un seul message ; mêmes garde-fous que l'agent pour une
# comparaison équitable : interdiction d'inventer, sortie = CV seul).
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """Tu es un expert en optimisation de CV pour les systèmes ATS (Applicant \
Tracking Systems).

## Mission
On te confie une annonce et le CV d'un candidat. Réécris ce CV pour qu'il s'aligne au mieux \
avec l'annonce et maximise ses chances face à un système ATS.

## Règle absolue
INTERDICTION d'inventer des compétences, expériences, diplômes ou résultats. Tu ne fais que \
REFORMULER et METTRE EN VALEUR la substance réelle déjà présente dans le CV face à l'annonce. \
Le CV doit rester crédible et honnête.

## Annonce visée
{announcement}

## CV à optimiser
{cv}

## Sortie
Renvoie UNIQUEMENT le texte complet du CV optimisé, sans commentaire ni explication autour."""


class OneShotCVOptimizer:
    """Optimiseur de CV en un seul appel LLM (baseline non itérative face à l'agent ReAct).

    Le rang du CV (avant/après réécriture) est mesuré par le ranker choisi via
    ``CV_OPTIMIZER_RANKER`` (cf. :mod:`ats_system.agents.dataset_rankers`), uniquement pour le
    rapport — il n'est jamais montré au LLM, qui réécrit à l'aveugle.
    """

    def __init__(
        self,
        dataset_dir: Path,
        announcement_text: str,
        announcement_name: Optional[str] = None,
        model: str = CV_OPTIMIZER_MODEL,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        ranker_name: str = CV_OPTIMIZER_RANKER,
        window_size: int = 4,
        num_passes: int = 3,
        api_key: Optional[str] = None,
        requests_per_second: float = LLM_REQUESTS_PER_SECOND,
        tokens_per_minute: int = LLM_TOKENS_PER_MINUTE,
    ):
        """
        Args:
            dataset_dir:         Dossier d'un dataset synthétique (``synthetic_cvs_<...>/``)
                                 contenant les CVs concurrents et un ``manifest.json``.
            announcement_text:   Texte complet de l'annonce visée.
            announcement_name:   Nom de l'annonce (pour le méta de sauvegarde). Optionnel.
            model:               Identifiant du modèle (fournisseur déduit du préfixe, cf.
                                 :func:`ats_system.llm.detect_provider`). Défaut :
                                 ``CV_OPTIMIZER_MODEL``.
            temperature:         Température d'échantillonnage du modèle.
            max_tokens:          Nombre maximum de tokens du CV réécrit.
            ranker_name:         Méthode de classement utilisée pour mesurer le rang avant/après
                                 (cf. ``CV_OPTIMIZER_RANKERS``).
            window_size:         Taille de fenêtre (ranker ``sliding_window`` uniquement).
            num_passes:          Nombre de passes (ranker ``sliding_window`` uniquement).
            api_key:             Clé API. À défaut, la variable d'environnement du fournisseur
                                 (chargée de ``.env``).
            requests_per_second: Débit cible des requêtes LLM (limiteur RPS partagé) pour éviter les 429.
                                 Défaut : ``LLM_REQUESTS_PER_SECOND``.
            tokens_per_minute:   Quota en tokens/minute (limiteur TPM partagé) pour éviter les 429 TPM.
                                 Défaut : ``LLM_TOKENS_PER_MINUTE``.
        """
        self.dataset_dir = Path(dataset_dir)
        self.announcement_text = announcement_text
        self.announcement_name = announcement_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ranker_name = ranker_name
        self.window_size = window_size
        self.num_passes = num_passes
        self._api_key = api_key
        self.requests_per_second = requests_per_second
        self.tokens_per_minute = tokens_per_minute

        self._llm: Optional[LLMClient] = None
        self._ranker: Optional[DatasetRanker] = None
        self._rate_limiter: Optional[InMemoryRateLimiter] = None
        self._tpm_limiter: Optional[TokensPerMinuteRateLimiter] = None
        # Cache rempli par import_model() : dicts {"id", "content"} des CVs concurrents.
        self._competitor_cvs: list[dict] = []

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def import_model(self) -> None:
        """Charge le client LLM, le ranker et le cache des CVs concurrents.

        Un unique ``InMemoryRateLimiter`` (token bucket) est partagé entre le client LLM et le
        ranker (utile au seul ranker LLM ``sliding_window``), garantissant un plafond global de
        débit même quand les appels s'alternent.
        """
        load_dotenv()

        # Limiteur de débit partagé entre le client LLM et le ranker (cf. CVOptimizerAgent).
        self._rate_limiter = InMemoryRateLimiter(
            requests_per_second=self.requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
        self._tpm_limiter = TokensPerMinuteRateLimiter(self.tokens_per_minute)

        self._llm = LLMClient(
            self.model,
            api_key=self._api_key,
            rate_limiter=self._rate_limiter,
            tpm_limiter=self._tpm_limiter,
        )
        self._llm.import_model()

        # Ranker du dataset : sert à mesurer le rang avant/après (jamais montré au LLM).
        self._ranker = build_dataset_ranker(
            self.ranker_name,
            rate_limiter=self._rate_limiter,
            tpm_limiter=self._tpm_limiter,
            window_size=self.window_size,
            num_passes=self.num_passes,
        )
        self._ranker.import_model()

        # Textes des CVs concurrents (tous sauf le « à optimiser ») — lus une fois.
        self._competitor_cvs = load_competitors(self.dataset_dir)
        logger.info("CVs concurrents chargés : %d", len(self._competitor_cvs))

    def optimize(self, cv_text: str) -> str:
        """Réécrit le CV en un seul appel LLM, à l'aveugle (annonce + CV uniquement).

        Args:
            cv_text: Texte du CV initial à optimiser.

        Returns:
            Le texte du CV optimisé renvoyé par le modèle.
        """
        if self._llm is None:
            raise RuntimeError("Appelez import_model() avant optimize().")
        prompt = PROMPT_TEMPLATE.format(announcement=self.announcement_text, cv=cv_text)
        return self._llm.complete(prompt, self.max_tokens, self.temperature)

    def run(self, save: bool = True) -> str:
        """Pipeline complet : localise le CV à optimiser, mesure le rang, réécrit, re-mesure, sauvegarde.

        Le CV « à optimiser » du dataset (entrée ``optimize: true`` du manifest) est repéré
        automatiquement. Son rang est mesuré avant la réécriture, puis le CV est réécrit en un
        seul appel LLM, puis le rang est re-mesuré. Si ``save``, écrit un PDF du CV optimisé +
        un ``meta.json`` (dataset, annonce, modèle, ranker, rang initial/final) sous
        ``results/cv_optimizer_oneshot/<horodatage>/``.

        Args:
            save: Si vrai, écrit le PDF du CV optimisé et le méta JSON.

        Returns:
            Le texte du CV optimisé.
        """
        if self._llm is None or self._ranker is None:
            raise RuntimeError("Appelez import_model() avant run().")

        cv_file, cv_text = find_optimize_cv(self.dataset_dir)
        print(f"CV à optimiser : {cv_file}")

        # Rang initial (CV original) — mesure pour le rapport, non transmise au LLM.
        initial = self._ranker.rank(self.announcement_text, self._competitor_cvs, "CV_original", cv_text)
        print(f"Rang initial : {initial.position}/{initial.total}")

        print("\nRéécriture du CV (un seul appel LLM, à l'aveugle)...")
        optimized_text = self.optimize(cv_text)

        # Rang final (CV réécrit).
        final = self._ranker.rank(self.announcement_text, self._competitor_cvs, "CV_optimisé", optimized_text)
        print(f"Rang final   : {final.position}/{final.total}")

        print("\n" + "=" * 70)
        print("CV OPTIMISÉ")
        print("=" * 70)
        print(optimized_text)

        if save:
            self._save_run(cv_file, optimized_text, initial.position, final.position, final.total)
        return optimized_text

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _save_run(
        self,
        cv_file: str,
        optimized_text: str,
        rang_initial: int,
        rang_final: int,
        total: int,
    ) -> Path:
        """Écrit le PDF du CV optimisé + un ``meta.json`` sous ``results/cv_optimizer_oneshot/<horodatage>/``."""
        run_dir = timestamped_run_dir(METHOD)
        write_text_pdf(optimized_text, run_dir / "cv_optimise.pdf")

        meta = {
            "agent": "one_shot_cv_optimizer",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": self.dataset_dir.name,
            "announcement": self.announcement_name,
            "model": self.model,
            "ranker": self.ranker_name,
            "cv_optimise_source": cv_file,
            "num_competitors": len(self._competitor_cvs),
            "rang_initial": rang_initial,
            "rang_final": rang_final,
            "total": total,
            "cv_optimise_text": optimized_text,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nRésultats sauvegardés dans : {run_dir}")
        return run_dir
