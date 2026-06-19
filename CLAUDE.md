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
  `ANNOUNCEMENTS_DIR`, `CV_DIR`, `RESULTS_DIR`, `DEFAULT_ANNOUNCEMENT`,
  `DEFAULT_CV_CATEGORY` = `"ENGINEERING"`, `DEFAULT_CV`,
  `SLIDING_WINDOW_MODEL` = `"claude-haiku-4-5"` — modèle Claude du ranker).
- **`data/pdf_loader.py`** — `import_pdf()` : PDF → `{"id", "content"}` (via `pypdf`).
  Réexporté par `ats_system.data`.
- **`results_io.py`** — schéma commun de sauvegarde des classements :
  `build_ranking()` (paires `(cv_id, score)` triées → entrées
  `{rank, cv_id, score[, justification]}`) et `save_results()` (JSON horodaté
  sous `RESULTS_DIR`, jamais écrasé ; champs spécifiques via `extra`).
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
    glissante (inspiré de RankGPT) via Claude (modèle `SLIDING_WINDOW_MODEL` de
    `config.py`, SDK `anthropic`). Entrées : `import_model()`, `load_cvs()`,
    `run_sliding_window_ranking()` → `RankingResult`, `display_results()`.

### `scripts/` — points d'entrée (`python scripts/<nom>.py`)

Comparent les approches de scoring sur les CVs `ENGINEERING` vs l'annonce par défaut
(arg `--limit N`, défaut 5 ; `0` = tous) :

- `compute_kw_match_scores.py` — baseline mots-clés.
- `compute_ml6_kw_match_scores.py` — mots-clés ml6team.
- `compute_emb_scores.py` — embeddings.
- `compute_sliding_window_ranking.py` — classement Claude (`--window-size`, `--passes` ;
  nécessite `ANTHROPIC_API_KEY`).
- `compute_all_rankings.py` — lance les **quatre** méthodes (baseline, ml6, embeddings,
  fenêtre glissante) sur le même run et écrit un JSON par méthode dans un dossier
  horodaté `results/all_rankings_<timestamp>/` (`--limit`, `--window-size`, `--passes`).
- `count_tokens.py` / `count_tokens_stats.py` — statistiques de tokens (tokenizer ml6team).

### `notebooks/` — analyses exploratoires (`jupyter`)

- `ranking_similarity_rbo.ipynb` — compare les classements d'un run `all_rankings_*`
  via le **Rank-Based Overlap** (RBO, Webber et al. 2010), la fenêtre glissante (Claude)
  servant de référence.
- `token_distribution.ipynb` — distribution du nombre de tokens des CVs.

### `data/` — données (non versionné)

- `data/announcements/` — annonces PDF (actuellement `mechanical_engineer_job_posting_2016.pdf`).
- `data/cv/<CATÉGORIE>/` — CVs PDF par métier (ACCOUNTANT, …, ENGINEERING, …, TEACHER).

### `results/` — sorties de classement (JSON horodatés via `results_io.save_results`)

## Garde-fous

- **`notes/`** — ⚠️ notes personnelles de l'utilisateur. **Ne pas lire, modifier ou supprimer.**
- **`ats_syst/`** — environnement virtuel. Ne pas versionner.
