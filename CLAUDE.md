# Projet : ats-system

## Description

Ce projet consiste à implémenter un système ATS (Applicant Tracking System).

Le projet suit un **layout `src/`** : `ats_system` est le seul package importable.
Il est installé en mode éditable (`pip install -e ".[dev]"`), donc les imports
se font en absolu (`from ats_system... import ...`) depuis les scripts et les tests.

## Structure du projet

### `pyproject.toml`

Dépendances, packaging (setuptools, layout `src/`) et configuration des outils
(`pytest`, `ruff`).

### `src/ats_system/` — le package importable

#### `config.py`

Chemins et constantes centralisés (basés sur `pathlib.Path`) :

- `PROJECT_ROOT`, `DATA_DIR`, `ANNOUNCEMENTS_DIR`, `CV_DIR`
- `DEFAULT_ANNOUNCEMENT` — `mechanical_engineer_job_posting_2016.pdf`
- `DEFAULT_CV_CATEGORY` (`"ENGINEERING"`), `DEFAULT_CV`

#### `data/` — I/O

- `data/pdf_loader.py` → `import_pdf(file_path: str) -> dict` — Extrait un fichier PDF (via `pypdf`) et le retourne sous la forme `{"id": nom_du_fichier, "content": texte}`. Réexporté par `ats_system.data`.

#### `scoring/` — logique de scoring

- `scoring/keyword.py`

  - `baseline_extract_keywords(text: str) -> set` — Extrait les mots-clés d'un texte (mise en minuscules, suppression de la ponctuation/chiffres, stopwords FR+EN supprimés, mots de ≤ 2 caractères ignorés).
  - `ml6_extract_keywords(model, text: str) -> set` — Extrait les keyphrases d'un texte via le LLM ml6team. `model` doit être chargé via `keyphrase_extractor.import_model()`.
  - `match_score(keywords_offre: set, keywords_cv: set) -> dict` — Calcule la correspondance entre deux ensembles de mots-clés (offre vs CV). Retourne `{"score": float, "matching": set, "missing": set}`.
- `scoring/embedding.py`

  - `emb_cos_score(model: SentenceTransformer, offre: str, cv: str) -> float` — Similarité cosinus entre l'embedding de l'offre et celui du CV (score 0–100). `model` chargé via `embedding_model.import_model()`.

  Ces fonctions sont réexportées par `ats_system.scoring`.

#### `models/` — wrappers LLM

- `models/keyphrase_extractor.py`
  - `import_model() -> pipeline` — Charge le modèle HuggingFace `ml6team/keyphrase-extraction-kbir-inspec` (token classification, basé sur BERT).
  - `infer_model(model, text: str) -> list[str]` — Extrait les keyphrases d'un texte. Découpe en chunks de 400 mots (limite 512 tokens de BERT). Déduplique.
- `models/embedding_model.py`
  - `import_model() -> SentenceTransformer` — Charge le modèle d'embeddings `all-MiniLM-L6-v2`.

### `scripts/` — points d'entrée exécutables

Lancés via `python scripts/<nom>.py` (le package étant installé en éditable).

- `scripts/compute_kw_match_scores.py` — Calcule le score keyword baseline (`baseline_extract_keywords` + `match_score`) de tous les CVs `ENGINEERING` contre l'annonce par défaut, et affiche les résultats triés du meilleur au plus bas score.
- `scripts/compute_ml6_kw_match_scores.py` — Même logique avec `ml6_extract_keywords` + `match_score`. Argument `--limit N` (défaut : 5 ; `0` = tous).
- `scripts/compute_emb_scores.py` — Même logique avec `emb_cos_score`. Argument `--limit N` (défaut : 5 ; `0` = tous).
- `scripts/count_tokens.py` — Compte le nombre de tokens d'un document PDF (tokenizer BERT `ml6team`). Arguments `--doc` (défaut : `DEFAULT_CV`) et `--model`. Affiche mots, tokens et nombre de chunks de 512 tokens nécessaires.
- `scripts/count_tokens_stats.py` — Calcule moyenne, écart type, min et max du nombre de tokens de tous les CVs du dossier `DEFAULT_CV_CATEGORY`. Arguments `--limit N` (défaut : 0 = tous) et `--model`.

### `notebooks/`

Notebooks Jupyter d'exploration.

### `tests/`

- `tests/test_import_pdf.py` — Teste `import_pdf` (assertions pytest).
- `tests/test_kw_match_score.py` — Teste `baseline_extract_keywords` + `match_score` entre l'annonce et un CV (paramétrable via `--offre` et `--cv`).
- `tests/test_ml6team_extractor.py` — Teste `import_model` et `infer_model` sur un CV ENGINEERING.
- `tests/test_ml6_kw_match_score.py` — Teste `ml6_extract_keywords` + `match_score` entre l'annonce et un CV (paramétrable via `--offre` et `--cv`).
- `tests/test_emb_score.py` — Teste `emb_cos_score` entre l'annonce et un CV (paramétrable via `--offre` et `--cv`).

### `data/` — données (non versionné)

- `data/announcements/` — Annonces d'emploi au format PDF
  - `mechanical_engineer_job_posting_2016.pdf` (seule annonce actuelle)
- `data/cv/` — CVs au format PDF, organisés par catégorie métier :
  ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION,
  BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT,
  DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR,
  INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER

### `ats_syst/`

Environnement virtuel Python. Ne pas versionner.

### `notes/`

> ⚠️ **Ne pas modifier.** Ce dossier contient les notes personnelles de l'utilisateur. Aucune lecture, modification ou suppression ne doit être effectuée sur son contenu.
