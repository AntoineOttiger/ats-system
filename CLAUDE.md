# Projet : ats-system

Système ATS (Applicant Tracking System) : scoring/classement de CVs (PDF) face à
une offre d'emploi, via trois approches comparées — mots-clés, embeddings, et un
LLM (Claude) en fenêtre glissante.

## Développement

- **Layout `src/`** : `ats_system` est le seul package importable. Installé en
  éditable, donc les imports sont **absolus** partout (`from ats_system… import …`),
  y compris dans `scripts/` et `tests/`.
- **Python** ≥ 3.10.
- **Installation** : `pip install -e ".[dev]"`
- **Tests** : `pytest`. ⚠️ Seul `tests/test_import_pdf.py` est un vrai test pytest
  (fonctions `test_*` + `assert`). Les autres fichiers de `tests/` sont en fait des
  **scripts CLI exploratoires** (`argparse` + `main()`, lancés via
  `python tests/<nom>.py`), non collectés par pytest.
- **Lint / format** : `ruff` (`line-length = 120`, `src = ["src", "scripts", "tests"]`).
- **Secrets** : copier `.env.example` → `.env` (ignoré par git) et renseigner
  `ANTHROPIC_API_KEY` (requis pour `SlidingWindowCVRanker`). Chargé via `python-dotenv`.

## Conventions

- Commentaires et docstrings en **français**.
- Type hints sur les signatures publiques.
- Constantes de chemins centralisées dans `config.py` — ne pas coder de chemins en dur.
- Chargement des modèles (HF, embeddings, Anthropic) volontairement séparé de
  l'inférence (`import_model()` puis `infer_model()` / fonctions de scoring) ; le
  chargement est coûteux (modèles HF) ou facturé (Claude).

## Cartographie

> Les signatures détaillées vivent dans le code — voici les responsabilités et les
> points d'entrée publics.

### `src/ats_system/`

- **`config.py`** — chemins et constantes (`PROJECT_ROOT`, `DATA_DIR`,
  `ANNOUNCEMENTS_DIR`, `CV_DIR`, `DEFAULT_ANNOUNCEMENT`,
  `DEFAULT_CV_CATEGORY` = `"ENGINEERING"`, `DEFAULT_CV`).
- **`data/pdf_loader.py`** — `import_pdf()` : PDF → `{"id", "content"}` (via `pypdf`).
  Réexporté par `ats_system.data`.
- **`scoring/`** (réexporté par `ats_system.scoring`) :
  - `keyword.py` — `baseline_extract_keywords()` (regex + stopwords FR/EN),
    `ml6_extract_keywords()` (via le modèle ml6team), `match_score()`
    (intersection offre/CV → `{"score", "matching", "missing"}`).
  - `embedding.py` — `emb_cos_score()` : similarité cosinus offre/CV (0–100).
- **`models/`** (wrappers de modèles) :
  - `keyphrase_extractor.py` — modèle HF `ml6team/keyphrase-extraction-kbir-inspec`
    (BERT, token classification ; chunks de 400 mots pour la limite 512 tokens).
  - `embedding_model.py` — `SentenceTransformer` `all-MiniLM-L6-v2`.
  - `sliding_window_ranker.py` — `SlidingWindowCVRanker` : classement par fenêtre
    glissante (inspiré de RankGPT) via Claude (`claude-opus-4-8`, SDK `anthropic`).
    Entrées : `import_model()`, `load_cvs()`, `run_sliding_window_ranking()` →
    `RankingResult`, `display_results()`.

### `scripts/` — points d'entrée (`python scripts/<nom>.py`)

Comparent les approches de scoring sur les CVs `ENGINEERING` vs l'annonce par défaut
(arg `--limit N`, défaut 5 ; `0` = tous) :

- `compute_kw_match_scores.py` — baseline mots-clés.
- `compute_ml6_kw_match_scores.py` — mots-clés ml6team.
- `compute_emb_scores.py` — embeddings.
- `compute_sliding_window_ranking.py` — classement Claude (`--window-size`, `--passes` ;
  nécessite `ANTHROPIC_API_KEY`).
- `count_tokens.py` / `count_tokens_stats.py` — statistiques de tokens (tokenizer ml6team).

### `data/` — données (non versionné)

- `data/announcements/` — annonces PDF (actuellement `mechanical_engineer_job_posting_2016.pdf`).
- `data/cv/<CATÉGORIE>/` — CVs PDF par métier (ACCOUNTANT, …, ENGINEERING, …, TEACHER).

## Garde-fous

- **`notes/`** — ⚠️ notes personnelles de l'utilisateur. **Ne pas lire, modifier ou supprimer.**
- **`ats_syst/`** — environnement virtuel. Ne pas versionner.
