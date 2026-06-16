import argparse

from ats_system.config import DEFAULT_ANNOUNCEMENT, CV_DIR, DEFAULT_CV_CATEGORY
from ats_system.data import import_pdf
from ats_system.models.sliding_window_ranker import SlidingWindowCVRanker

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


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


if __name__ == "__main__":
    main()
