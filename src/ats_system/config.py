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

# Modèle Claude utilisé par le SlidingWindowCVRanker (classement par fenêtre glissante)
SLIDING_WINDOW_MODEL = "claude-haiku-4-5"

# Modèle Mistral utilisé par le SyntheticCVGenerator (génération de CVs synthétiques)
CV_GENERATOR_MODEL = "mistral-small-latest"

# Modèle Mistral utilisé par le CVOptimizerAgent (agent d'optimisation de CV, LangGraph)
CV_OPTIMIZER_MODEL = "mistral-small-latest"
