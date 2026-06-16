"""Chemins, constantes et settings centralisés du projet."""

from pathlib import Path

# Racine du projet (src/ats_system/config.py -> remonte de 3 niveaux)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Données
DATA_DIR = PROJECT_ROOT / "data"
ANNOUNCEMENTS_DIR = DATA_DIR / "announcements"
CV_DIR = DATA_DIR / "cv"

# Sorties des scripts (résultats de classement, scores, etc.)
RESULTS_DIR = PROJECT_ROOT / "results"

# Valeurs par défaut utilisées par les scripts et les tests
DEFAULT_ANNOUNCEMENT = ANNOUNCEMENTS_DIR / "mechanical_engineer_job_posting_2016.pdf"
DEFAULT_CV_CATEGORY = "ENGINEERING"
DEFAULT_CV = CV_DIR / DEFAULT_CV_CATEGORY / "12472574.pdf"
