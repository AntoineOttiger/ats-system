"""Sauvegarde des résultats de classement de CVs.

Schéma commun à toutes les méthodes (mots-clés, embeddings, fenêtre glissante) :
un JSON horodaté par run, jamais écrasé, contenant un classement ordonné du
meilleur au pire candidat. Les champs spécifiques à une méthode (``passes``,
``converged`` pour la fenêtre glissante…) sont ajoutés via ``extra``.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ats_system.config import RESULTS_DIR


def build_ranking(
    scored: list[tuple[str, float]],
    justifications: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Construit la liste ``ranking`` au schéma commun à partir de paires (cv_id, score).

    Args:
        scored:         paires ``(cv_id, score)`` déjà triées du meilleur au pire.
        justifications: justification par ``cv_id`` (optionnel ; méthodes LLM). Si
                        fourni, chaque entrée gagne un champ ``justification``.

    Returns:
        Liste d'entrées ``{rank, cv_id, score[, justification]}``.
    """
    ranking = []
    for rank, (cv_id, score) in enumerate(scored, 1):
        entry: dict = {"rank": rank, "cv_id": cv_id, "score": score}
        if justifications is not None:
            entry["justification"] = justifications.get(cv_id, "")
        ranking.append(entry)
    return ranking


def save_results(
    method: str,
    ranking: list[dict],
    params: dict,
    extra: Optional[dict] = None,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Sauvegarde un classement dans un JSON horodaté (jamais écrasé).

    Le fichier est nommé ``{method}_{timestamp}.json`` ; un suffixe incrémental
    est ajouté en cas de collision improbable, garantissant qu'aucun run n'en
    écrase un autre.

    Args:
        method:      identifiant de la méthode (préfixe de fichier + champ ``method``).
        ranking:     entrées ``{rank, cv_id, score, ...}`` déjà triées (cf. ``build_ranking``).
        params:      paramètres du run (annonce, catégorie, limit, modèle…).
        extra:       champs supplémentaires de premier niveau (ex. ``passes``, ``converged``).
        results_dir: dossier de sortie (défaut ``RESULTS_DIR``).

    Returns:
        Le chemin du fichier écrit.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"{method}_{stamp}.json"
    counter = 1
    while out_path.exists():
        out_path = results_dir / f"{method}_{stamp}_{counter}.json"
        counter += 1

    payload = {
        "method": method,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        **(extra or {}),
        "ranking": ranking,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
