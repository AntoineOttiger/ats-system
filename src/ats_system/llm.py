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
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_MISTRAL = "mistral"

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

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Args:
            model:   Identifiant du modèle. Le fournisseur en est déduit
                     (cf. :func:`detect_provider`).
            api_key: Clé API explicite. À défaut, la variable d'environnement
                     propre au fournisseur (chargée depuis ``.env``) est utilisée.
        """
        self.model = model
        self.provider = detect_provider(model)
        self._api_key = api_key
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

        if self.provider == PROVIDER_ANTHROPIC:
            kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            # Pas de thinking : on isole le premier bloc texte (les modèles avec
            # thinking adaptatif peuvent renvoyer des blocs en amont).
            response = self.client.messages.create(**kwargs)
            return next((block.text for block in response.content if block.type == "text"), "")

        # Mistral
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.chat.complete(**kwargs)
        return (response.choices[0].message.content or "").strip()
