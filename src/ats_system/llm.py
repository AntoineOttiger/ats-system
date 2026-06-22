"""Abstraction multi-fournisseurs (Anthropic / Mistral) pour les complétions texte.

Un même prompt → une réponse texte, quel que soit le fournisseur. Le fournisseur
n'est jamais passé explicitement : il est **déduit du préfixe du nom de modèle**
(``claude-*`` → Anthropic, sinon Mistral). Cela permet de basculer un système ATS
d'un fournisseur à l'autre en changeant uniquement la constante de modèle dans
``config.py``.

La clé API est lue depuis un fichier ``.env`` selon le fournisseur
(``ANTHROPIC_API_KEY`` ou ``MISTRAL_API_KEY``) — jamais codée en dur. Conforme à la
convention du projet : le chargement (``import_model``) est séparé de l'inférence
(``complete``).
"""

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langchain_core.rate_limiters import BaseRateLimiter

logger = logging.getLogger(__name__)


PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_MISTRAL = "mistral"


class TokensPerMinuteRateLimiter:
    """Limiteur de débit en tokens par minute (token bucket, thread-safe).

    Utilise la même logique que ``InMemoryRateLimiter`` de ``langchain_core``
    mais la granularité est le token, pas la requête.
    """

    def __init__(self, tokens_per_minute: int, check_every_n_seconds: float = 0.1):
        self.tokens_per_minute = tokens_per_minute
        self._tokens_per_second = tokens_per_minute / 60.0
        self._check_every_n_seconds = check_every_n_seconds
        self._available = float(tokens_per_minute)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n_tokens: int, blocking: bool = True) -> bool:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._last_refill = now
                self._available = min(
                    float(self.tokens_per_minute),
                    self._available + elapsed * self._tokens_per_second,
                )
                if self._available >= n_tokens:
                    self._available -= n_tokens
                    return True
                if not blocking:
                    return False
            time.sleep(self._check_every_n_seconds)

# Préfixes de noms de modèles Mistral connus (cf. catalogue Mistral).
_MISTRAL_PREFIXES = ("mistral", "ministral", "magistral", "codestral", "pixtral", "open-")

# Variable d'environnement portant la clé API, par fournisseur.
_API_KEY_ENV = {
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    PROVIDER_MISTRAL: "MISTRAL_API_KEY",
}


def detect_provider(model: str) -> str:
    """Déduit le fournisseur (``anthropic`` / ``mistral``) du préfixe du nom de modèle.

    Args:
        model: Identifiant du modèle (ex. ``"claude-haiku-4-5"``, ``"mistral-small-latest"``).

    Returns:
        ``PROVIDER_ANTHROPIC`` ou ``PROVIDER_MISTRAL``.

    Raises:
        ValueError: si le préfixe n'est reconnu par aucun fournisseur.
    """
    name = model.lower()
    if name.startswith("claude"):
        return PROVIDER_ANTHROPIC
    if name.startswith(_MISTRAL_PREFIXES):
        return PROVIDER_MISTRAL
    raise ValueError(
        f"Fournisseur indéterminé pour le modèle {model!r}. "
        f"Les noms doivent commencer par 'claude' (Anthropic) ou par un préfixe "
        f"Mistral connu ({', '.join(_MISTRAL_PREFIXES)})."
    )


class LLMClient:
    """Client de complétion texte unifié Anthropic / Mistral.

    Le fournisseur est déterminé à la construction à partir du nom de modèle. Le
    client SDK sous-jacent n'est instancié qu'à l'appel de :func:`import_model`
    (chargement coûteux/facturé séparé de l'inférence).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        rate_limiter: Optional["BaseRateLimiter"] = None,
        tpm_limiter: Optional["TokensPerMinuteRateLimiter"] = None,
        max_retries: int = 5,
    ):
        """
        Args:
            model:        Identifiant du modèle. Le fournisseur en est déduit
                          (cf. :func:`detect_provider`).
            api_key:      Clé API explicite. À défaut, la variable d'environnement
                          propre au fournisseur (chargée depuis ``.env``) est utilisée.
            rate_limiter: Limiteur de débit en requêtes/seconde (``BaseRateLimiter`` de
                          ``langchain_core``), **appliqué uniquement aux modèles Mistral**.
                          Si fourni, ``acquire(blocking=True)`` est appelé avant chaque
                          requête. Ignoré pour Claude.
            tpm_limiter:  Limiteur de débit en tokens/minute (:class:`TokensPerMinuteRateLimiter`),
                          **appliqué uniquement aux modèles Mistral**. Si fourni,
                          ``acquire(n_tokens)`` est appelé avant chaque requête avec une
                          estimation du prompt (``len(prompt) // 4``). Ignoré pour Claude.
            max_retries:  Nombre maximum de tentatives en cas d'erreur de rate limit
                          (429), avec backoff exponentiel.
        """
        self.model = model
        self.provider = detect_provider(model)
        self._api_key = api_key
        self._rate_limiter = rate_limiter
        self._tpm_limiter = tpm_limiter
        self._max_retries = max_retries
        self.client = None

    def import_model(self) -> None:
        """Instancie le client SDK du fournisseur. À appeler avant :func:`complete`.

        La clé API est lue depuis l'argument ``api_key`` ou, à défaut, depuis la
        variable d'environnement du fournisseur (``ANTHROPIC_API_KEY`` ou
        ``MISTRAL_API_KEY``), chargée d'un fichier ``.env``.
        """
        load_dotenv()
        env_var = _API_KEY_ENV[self.provider]
        key = self._api_key or os.environ.get(env_var)
        if not key:
            raise EnvironmentError(
                f"Aucune clé API trouvée. Renseignez {env_var} dans un fichier .env "
                "(voir .env.example) ou passez api_key=."
            )

        if self.provider == PROVIDER_ANTHROPIC:
            import anthropic

            self.client = anthropic.Anthropic(api_key=key)
        else:
            from mistralai.client import Mistral

            self.client = Mistral(api_key=key)

        logger.info("Client %s initialisé (modèle : %s)", self.provider, self.model)

    def complete(self, prompt: str, max_tokens: int, temperature: Optional[float] = None) -> str:
        """Envoie un prompt unique et retourne la réponse texte du modèle.

        Args:
            prompt:      Texte du message utilisateur.
            max_tokens:  Nombre maximum de tokens de la réponse.
            temperature: Température d'échantillonnage. Si ``None``, la valeur par
                         défaut du SDK est utilisée (non transmise).

        Returns:
            Le texte de la réponse (chaîne, éventuellement vide).
        """
        if self.client is None:
            raise RuntimeError("Appelez import_model() avant complete().")

        messages = [{"role": "user", "content": prompt}]
        n_tokens_est = len(prompt) // 4

        if self.provider == PROVIDER_ANTHROPIC:
            kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            # Pas de thinking : on isole le premier bloc texte (les modèles avec
            # thinking adaptatif peuvent renvoyer des blocs en amont).
            response = self._call_with_retry(self.client.messages.create, n_tokens=n_tokens_est, **kwargs)
            return next((block.text for block in response.content if block.type == "text"), "")

        # Mistral
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self._call_with_retry(self.client.chat.complete, n_tokens=n_tokens_est, **kwargs)
        return (response.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Détecte une erreur de type rate limit (429), quel que soit le SDK."""
        if getattr(exc, "status_code", None) == 429:
            return True
        text = str(exc).lower()
        return "429" in text or "rate limit" in text

    def _call_with_retry(self, func, *, n_tokens: int = 0, **kwargs):
        """Appelle ``func(**kwargs)`` avec limitation de débit et retry sur 429.

        Respecte les limiteurs de débit partagés (RPS puis TPM) avant chaque tentative,
        puis réessaie avec un backoff exponentiel si l'API répond par un rate limit.
        """
        # Les limiteurs de débit ne visent que Mistral (free tier) ; Claude n'est pas throttlé.
        throttled = self.provider == PROVIDER_MISTRAL
        rps_throttled = throttled and self._rate_limiter is not None
        tpm_throttled = throttled and self._tpm_limiter is not None

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            if rps_throttled:
                self._rate_limiter.acquire(blocking=True)
            if tpm_throttled:
                self._tpm_limiter.acquire(n_tokens)
            try:
                return func(**kwargs)
            except Exception as exc:  # noqa: BLE001 — on relève si ce n'est pas un 429
                if not self._is_rate_limit_error(exc):
                    raise
                last_exc = exc
                wait = 2.0 ** attempt
                logger.warning(
                    "Rate limit atteint (tentative %d/%d). Nouvel essai dans %.1fs.",
                    attempt + 1, self._max_retries, wait,
                )
                time.sleep(wait)
        # Toutes les tentatives ont échoué sur un rate limit.
        raise last_exc
