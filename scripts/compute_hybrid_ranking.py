"""Classe les CVs ENGINEERING face à l'annonce par défaut via le système hybride.

Pipeline : présélection mots-clés (``Ml6KeywordMatcher``) puis affinage LLM par
fenêtre glissante (``SlidingWindowCVRanker``) sur tous les CVs, en partant de l'ordre
mots-clés.

Chaque run crée un dossier horodaté ``results/hybrid_ranking_<timestamp>/`` contenant
tout l'historique de classement (schéma commun de ``ats_system.results_io``) :
  - ``ml6_keyword_match_*.json``       : classement de la première passe mots-clés ;
  - ``sliding_window_pass{N}_*.json``  : classement après chaque passe de la fenêtre
                                         glissante ;
  - ``hybrid_ranking_*.json``          : classement final.
"""

import argparse
from datetime import datetime

from ats_system.config import (
    CV_DIR,
    DEFAULT_ANNOUNCEMENT,
    DEFAULT_CV_CATEGORY,
    ML6_KEYWORD_MODEL,
    RESULTS_DIR,
    SLIDING_WINDOW_MODEL,
)
from ats_system.data import import_pdf
from ats_system.results_io import build_ranking, save_results
from ats_system.systems import HybridMl6SlidingWindowRanker

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def _position_scored(order: list[str]) -> list[tuple[str, float]]:
    """Convertit un ordre de cv_ids en paires (cv_id, score) basées sur la position."""
    n = len(order)
    return [(cv_id, round((n - i) / n, 4)) for i, cv_id in enumerate(order)]


def main():
    parser = argparse.ArgumentParser(
        description="Classe les CVs ENGINEERING face à l'annonce par défaut via le système hybride "
        "(présélection mots-clés ml6 puis affinage LLM par fenêtre glissante)."
    )
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument("--window-size", type=int, default=4, help="Fenêtre glissante : CVs par appel LLM")
    parser.add_argument("--passes", type=int, default=3, help="Fenêtre glissante : nombre maximum de passes")
    parser.add_argument(
        "--model", type=str, default=SLIDING_WINDOW_MODEL,
        help="Modèle de l'affinage LLM (fournisseur déduit du préfixe : Claude ou Mistral)",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    print("Initialisation du système hybride (chargement des modèles)...")
    ranker = HybridMl6SlidingWindowRanker(
        window_size=args.window_size,
        num_passes=args.passes,
        model=args.model,
    )
    ranker.import_model()

    print("Extraction du texte de l'annonce...")
    offre = import_pdf(str(DEFAULT_ANNOUNCEMENT))

    cv_files = sorted(CATEGORY_DIR.glob("*.pdf"))
    if limit is not None:
        cv_files = cv_files[:limit]

    print(f"Chargement de {len(cv_files)} CVs...")
    cvs = [import_pdf(str(cv_path)) for cv_path in cv_files]

    print("Classement en cours (passe mots-clés puis appels LLM)...")
    result = ranker.rank(job_offer=offre["content"], cvs=cvs)

    ranker.display_results(result)

    # Dossier de run : tout l'historique de classement dedans.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / f"hybrid_ranking_{stamp}"

    base_params = {
        "announcement": DEFAULT_ANNOUNCEMENT.name,
        "category": DEFAULT_CV_CATEGORY,
        "limit": args.limit,
        "num_cvs": len(cvs),
    }

    # --- 1. Première passe : mots-clés ml6 (sur tous les CVs) ---
    save_results(
        "ml6_keyword_match",
        build_ranking(result.ml6_ranking),
        {**base_params, "model": ML6_KEYWORD_MODEL, "stage": "preselection"},
        results_dir=run_dir,
    )

    # --- 2. Chaque passe de la fenêtre glissante ---
    sw = result.sliding_window
    if sw is not None:
        for i, (order, justifs) in enumerate(zip(sw.pass_orders, sw.pass_justifications), 1):
            save_results(
                f"sliding_window_pass{i}",
                build_ranking(_position_scored(order), justifs),
                {
                    **base_params,
                    "model": args.model,
                    "stage": "refinement",
                    "pass": i,
                    "window_size": args.window_size,
                },
                results_dir=run_dir,
            )

    # --- 3. Classement final ---
    out_path = save_results(
        "hybrid_ranking",
        build_ranking(result.ranked, result.justifications),
        {
            **base_params,
            "model": args.model,
            "ml6_model": ML6_KEYWORD_MODEL,
            "window_size": args.window_size,
            "passes": args.passes,
        },
        extra={
            "passes": sw.passes if sw is not None else 0,
            "converged": sw.converged if sw is not None else False,
        },
        results_dir=run_dir,
    )

    print(f"\nHistorique complet sauvegardé dans : {run_dir}")
    print(f"Classement final : {out_path.name}")


if __name__ == "__main__":
    main()
