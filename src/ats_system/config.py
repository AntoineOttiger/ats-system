"""Chemins, constantes et settings centralisés du projet."""

from pathlib import Path

# Racine du projet (src/ats_system/config.py -> remonte de 3 niveaux)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Données
DATA_DIR = PROJECT_ROOT / "data"
ANNOUNCEMENTS_DIR = DATA_DIR / "announcement"
CV_DIR = DATA_DIR / "cv"
GENERATED_DATA_DIR = DATA_DIR / "generated_data"

# Sorties des scripts (résultats de classement, scores, etc.)
RESULTS_DIR = PROJECT_ROOT / "results"

# Valeurs par défaut utilisées par les scripts et les tests
DEFAULT_ANNOUNCEMENT = ANNOUNCEMENTS_DIR / "mechanical_engineer_job_posting_2016.pdf"
DEFAULT_CV_CATEGORY = "ENGINEERING"
DEFAULT_CV = CV_DIR / DEFAULT_CV_CATEGORY / "12472574.pdf"

# Modèle utilisé par le SlidingWindowCVRanker (classement par fenêtre glissante).
# Le fournisseur (Claude ou Mistral) est déduit du préfixe du nom — il suffit de
# changer cette valeur pour basculer (ex. "mistral-small-latest").
SLIDING_WINDOW_MODEL = "mistral-small-latest"

# Modèle utilisé par le SyntheticCVGenerator (génération de CVs synthétiques).
# Idem : fournisseur déduit du préfixe (ex. "claude-haiku-4-5" pour passer sur Claude).
CV_GENERATOR_MODEL = "mistral-small-latest"

# Modèle Mistral utilisé par le CVOptimizerAgent (agent d'optimisation de CV, LangGraph)
CV_OPTIMIZER_MODEL = "mistral-small-latest"

# Débit cible par défaut des appels LLM (requêtes/seconde) pour éviter les 429 de
# l'API. Calibré sur le free tier Mistral (~1 req/s) ; à augmenter sur un tier payant.
LLM_REQUESTS_PER_SECOND = 1.0
