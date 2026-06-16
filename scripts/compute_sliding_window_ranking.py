import argparse
import json
from datetime import datetime
from pathlib import Path

from ats_system.config import DEFAULT_ANNOUNCEMENT, CV_DIR, DEFAULT_CV_CATEGORY, RESULTS_DIR
from ats_system.data import import_pdf
from ats_system.models.sliding_window_ranker import RankingResult, SlidingWindowCVRanker

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def save_results(result: RankingResult, params: dict) -> Path:
    """Sauvegarde le classement dans un JSON horodaté (jamais écrasé).

    Le nom inclut un timestamp à la seconde ; un suffixe incrémental est ajouté
    en cas de collision improbable, garantissant qu'aucun run n'en écrase un autre.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"sliding_window_ranking_{stamp}.json"
    counter = 1
    while out_path.exists():
        out_path = RESULTS_DIR / f"sliding_window_ranking_{stamp}_{counter}.json"
        counter += 1

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "passes": result.passes,
        "converged": result.converged,
        "ranking": [
            {
                "rank": rank,
                "cv_id": cv.id,
                "score": result.scores[cv.id],
                "justification": result.justifications.get(cv.id, ""),
            }
            for rank, cv in enumerate(result.ranked_cvs, 1)
        ],
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Classe les CVs ENGINEERING face à l'annonce par défaut via le sliding window ranker (LLM Claude)."
    )
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument("--window-size", type=int, default=4, help="Nombre de CVs comparés par appel LLM")
    parser.add_argument("--passes", type=int, default=3, help="Nombre maximum de passes")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    print("Initialisation du ranker...")
    ranker = SlidingWindowCVRanker(window_size=args.window_size, num_passes=args.passes)
    ranker.import_model()

    print("Extraction du texte de l'annonce...")
    offre = import_pdf(str(DEFAULT_ANNOUNCEMENT))

    cv_files = sorted(CATEGORY_DIR.glob("*.pdf"))
    if limit is not None:
        cv_files = cv_files[:limit]

    print(f"Chargement de {len(cv_files)} CVs...")
    cvs = ranker.load_cvs([import_pdf(str(cv_path)) for cv_path in cv_files])

    print("Classement en cours (appels LLM)...")
    result = ranker.run_sliding_window_ranking(job_offer=offre["content"], cvs=cvs)

    ranker.display_results(result)

    params = {
        "announcement": DEFAULT_ANNOUNCEMENT.name,
        "category": DEFAULT_CV_CATEGORY,
        "limit": args.limit,
        "window_size": args.window_size,
        "passes": args.passes,
        "num_cvs": len(cvs),
    }
    out_path = save_results(result, params)
    print(f"\nRésultats sauvegardés dans : {out_path}")


if __name__ == "__main__":
    main()
