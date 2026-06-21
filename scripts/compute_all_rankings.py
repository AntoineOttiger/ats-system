"""Classe les CVs ENGINEERING face à l'annonce par défaut via les quatre méthodes.

Méthodes comparées :
  - baseline_keyword_match : mots-clés (regex + stopwords FR/EN)
  - ml6_keyword_match       : mots-clés (modèle ml6team)
  - embedding_cosine        : similarité cosinus (all-MiniLM-L6-v2)
  - sliding_window_ranking  : classement LLM par fenêtre glissante (Claude ou Mistral)

Chaque run crée un dossier horodaté ``results/all_rankings_<timestamp>/`` contenant
un fichier JSON par méthode (schéma commun de ``ats_system.results_io``).
"""

import argparse
from datetime import datetime

from ats_system.config import DEFAULT_ANNOUNCEMENT, CV_DIR, DEFAULT_CV_CATEGORY, ML6_KEYWORD_MODEL, RESULTS_DIR
from ats_system.data import import_pdf
from ats_system.results_io import build_ranking, save_results
from ats_system.systems import (
    BaselineKeywordMatcher,
    EmbeddingCosineScorer,
    Ml6KeywordMatcher,
    SlidingWindowCVRanker,
)

CATEGORY_DIR = CV_DIR / DEFAULT_CV_CATEGORY


def rank_baseline(matcher: BaselineKeywordMatcher, offre_text: str, cvs: list[dict]) -> list[tuple[str, float]]:
    """Classement par mots-clés baseline ; retourne (cv_id, score) trié décroissant."""
    keywords_offre = matcher.extract_keywords(offre_text)
    scored = [
        (cv["id"], matcher.match(keywords_offre, matcher.extract_keywords(cv["content"]))["score"])
        for cv in cvs
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def rank_ml6(matcher: Ml6KeywordMatcher, offre_text: str, cvs: list[dict]) -> list[tuple[str, float]]:
    """Classement par mots-clés ml6team ; retourne (cv_id, score) trié décroissant."""
    keywords_offre = matcher.extract_keywords(offre_text)
    scored = [
        (cv["id"], matcher.match(keywords_offre, matcher.extract_keywords(cv["content"]))["score"])
        for cv in cvs
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def rank_embedding(scorer: EmbeddingCosineScorer, offre_text: str, cvs: list[dict]) -> list[tuple[str, float]]:
    """Classement par similarité cosinus ; retourne (cv_id, score) trié décroissant."""
    scored = [(cv["id"], scorer.score(offre_text, cv["content"])) for cv in cvs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def main():
    parser = argparse.ArgumentParser(
        description="Classe les CVs ENGINEERING face à l'annonce par défaut via les quatre méthodes."
    )
    parser.add_argument("--limit", type=int, default=30, help="Nombre maximum de CVs à classer (0 = tous)")
    parser.add_argument("--window-size", type=int, default=4, help="Fenêtre glissante : CVs par appel LLM")
    parser.add_argument("--passes", type=int, default=10, help="Fenêtre glissante : nombre maximum de passes")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    print("Extraction du texte de l'annonce...")
    offre = import_pdf(str(DEFAULT_ANNOUNCEMENT))
    offre_text = offre["content"]

    cv_files = sorted(CATEGORY_DIR.glob("*.pdf"))
    if limit is not None:
        cv_files = cv_files[:limit]

    print(f"Chargement de {len(cv_files)} CVs...")
    cvs = [import_pdf(str(cv_path)) for cv_path in cv_files]

    # Dossier de run : un fichier par méthode dedans.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / f"all_rankings_{stamp}"

    base_params = {
        "announcement": DEFAULT_ANNOUNCEMENT.name,
        "category": DEFAULT_CV_CATEGORY,
        "limit": args.limit,
        "num_cvs": len(cvs),
    }

    # --- 1. Mots-clés baseline ---
    print("\n[1/4] Mots-clés baseline...")
    baseline_matcher = BaselineKeywordMatcher()
    baseline_matcher.import_model()
    scored = rank_baseline(baseline_matcher, offre_text, cvs)
    out = save_results("baseline_keyword_match", build_ranking(scored), dict(base_params), results_dir=run_dir)
    print(f"  -> {out}")

    # --- 2. Mots-clés ml6team ---
    print("\n[2/4] Mots-clés ml6team (chargement du modèle)...")
    ml6_matcher = Ml6KeywordMatcher()
    ml6_matcher.import_model()
    scored = rank_ml6(ml6_matcher, offre_text, cvs)
    params = {**base_params, "model": ML6_KEYWORD_MODEL}
    out = save_results("ml6_keyword_match", build_ranking(scored), params, results_dir=run_dir)
    print(f"  -> {out}")

    # --- 3. Embeddings ---
    print("\n[3/4] Embeddings (chargement du modèle)...")
    emb_scorer = EmbeddingCosineScorer()
    emb_scorer.import_model()
    scored = rank_embedding(emb_scorer, offre_text, cvs)
    params = {**base_params, "model": "all-MiniLM-L6-v2"}
    out = save_results("embedding_cosine", build_ranking(scored), params, results_dir=run_dir)
    print(f"  -> {out}")

    # --- 4. Fenêtre glissante (LLM) ---
    print("\n[4/4] Fenêtre glissante (appels LLM)...")
    ranker = SlidingWindowCVRanker(window_size=args.window_size, num_passes=args.passes)
    ranker.import_model()
    cv_objs = ranker.load_cvs(cvs)
    result = ranker.run_sliding_window_ranking(job_offer=offre_text, cvs=cv_objs)
    sw_scored = [(cv.id, result.scores[cv.id]) for cv in result.ranked_cvs]
    params = {
        **base_params,
        "model": ranker.model,
        "window_size": args.window_size,
        "passes": args.passes,
    }
    out = save_results(
        "sliding_window_ranking",
        build_ranking(sw_scored, result.justifications),
        params,
        extra={"passes": result.passes, "converged": result.converged},
        results_dir=run_dir,
    )
    print(f"  -> {out}")

    print(f"\nTous les classements sauvegardés dans : {run_dir}")


if __name__ == "__main__":
    main()
