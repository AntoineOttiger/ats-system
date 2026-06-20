import argparse

from ats_system.config import DEFAULT_ANNOUNCEMENT, CV_DIR, DEFAULT_CV_CATEGORY, SLIDING_WINDOW_MODEL
from ats_system.data import import_pdf
from ats_system.systems import SlidingWindowCVRanker
from ats_system.results_io import build_ranking, save_results

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def main():
    parser = argparse.ArgumentParser(
        description="Classe les CVs ENGINEERING face à l'annonce par défaut via le sliding window ranker (LLM, modèle défini dans config.py)."
    )
    parser.add_argument("--limit", type=int, default=5, help="Nombre maximum de CVs à traiter (0 = tous)")
    parser.add_argument("--window-size", type=int, default=4, help="Nombre de CVs comparés par appel LLM")
    parser.add_argument("--passes", type=int, default=3, help="Nombre maximum de passes")
    parser.add_argument(
        "--model", type=str, default=SLIDING_WINDOW_MODEL,
        help="Identifiant du modèle (fournisseur déduit du préfixe : Claude ou Mistral)",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    print("Initialisation du ranker...")
    ranker = SlidingWindowCVRanker(window_size=args.window_size, num_passes=args.passes, model=args.model)
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
        "model": ranker.model,
        "limit": args.limit,
        "window_size": args.window_size,
        "passes": args.passes,
        "num_cvs": len(cvs),
    }
    scored = [(cv.id, result.scores[cv.id]) for cv in result.ranked_cvs]
    ranking = build_ranking(scored, result.justifications)
    out_path = save_results(
        "sliding_window_ranking",
        ranking,
        params,
        extra={"passes": result.passes, "converged": result.converged},
    )
    print(f"\nRésultats sauvegardés dans : {out_path}")


if __name__ == "__main__":
    main()
