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


def timestamped_run_dir(name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Crée et renvoie un dossier de run horodaté ``results_dir / name / <horodatage>``.

    Permet à chaque point d'entrée (système, agent…) de regrouper ses sorties dans son
    propre dossier daté, ex. ``results/all_rankings/20260618-171737/``. Même nommage
    anti-collision que ``save_results`` : un suffixe incrémental est ajouté si le dossier
    existe déjà.

    Args:
        name:        sous-dossier propre au point d'entrée (ex. ``"all_rankings"``).
        results_dir: dossier racine des résultats (défaut ``RESULTS_DIR``).

    Returns:
        Le ``Path`` du dossier créé.
    """
    base = results_dir / name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / stamp
    counter = 1
    while run_dir.exists():
        run_dir = base / f"{stamp}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def save_results(
    method: str,
    ranking: list[dict],
    params: dict,
    extra: Optional[dict] = None,
    results_dir: Path = RESULTS_DIR,
    stamp_filename: bool = True,
) -> Path:
    """Sauvegarde un classement dans un JSON (jamais écrasé).

    Le fichier est nommé ``{method}_{timestamp}.json`` (ou ``{method}.json`` si
    ``stamp_filename`` est faux) ; un suffixe incrémental est ajouté en cas de collision,
    garantissant qu'aucun run n'en écrase un autre.

    Args:
        method:         identifiant de la méthode (préfixe de fichier + champ ``method``).
        ranking:        entrées ``{rank, cv_id, score, ...}`` déjà triées (cf. ``build_ranking``).
        params:         paramètres du run (annonce, catégorie, limit, modèle…).
        extra:          champs supplémentaires de premier niveau (ex. ``passes``, ``converged``).
        results_dir:    dossier de sortie (défaut ``RESULTS_DIR``).
        stamp_filename: horodate le nom de fichier. Mettre à ``False`` à l'intérieur d'un
                        dossier déjà horodaté (cf. ``timestamped_run_dir``) pour des noms
                        propres (``baseline_keyword_match.json``…).

    Returns:
        Le chemin du fichier écrit.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if stamp_filename:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = f"{method}_{stamp}"
    else:
        base_name = method
    out_path = results_dir / f"{base_name}.json"
    counter = 1
    while out_path.exists():
        out_path = results_dir / f"{base_name}_{counter}.json"
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
