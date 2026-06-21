# Projet : ats-system

Système ATS (Applicant Tracking System) : scoring/classement de CVs (PDF) face à
une offre d'emploi, via trois approches comparées — mots-clés, embeddings, et un
LLM (Claude **ou** Mistral, fournisseur déduit du nom de modèle) en fenêtre glissante.

## Skills disponibles

Détails par module dans `.claude/skills/` (à lire à la demande) :

- `systems.md` — les systèmes ATS (mots-clés baseline/ml6, embeddings, fenêtre glissante, hybride, orchestrateur).
- `generators.md` — génération de CVs synthétiques PDF (vérité-terrain + CV « à optimiser »).
- `agents.md` — agent d'optimisation de CV (ReAct) et couche d'adaptation des rankers.
- `scripts.md` — points d'entrée CLI (`python scripts/<nom>.py`).

**Ne consulte un skill que si la tâche courante le nécessite. Lis le skill
correspondant avant de modifier du code dans ce module.**

## Maintenance automatique

Un hook `PostToolUse` (`.claude/settings.json` → `.claude/scripts/sync_skills.py`)
surveille les modifications de code (events `Edit`, `Write`, `Bash`) et **détecte quand
un skill est désynchronisé** du module qu'il documente. Le mapping dossier → skill est
extrait dynamiquement des lignes « → détails : `.claude/skills/<nom>.md` » de ce fichier.

**Si le hook retourne une erreur** (code de sortie ≠ 0, message `[sync_skills] …`) :

1. Lire `.claude/.stale_skills` (un nom de skill périmé par ligne).
2. Pour chaque skill listé, relire le code modifié dans le dossier correspondant.
3. Mettre à jour le skill (`.claude/skills/<nom>.md`) pour refléter les **vrais
   changements** du code (signatures, comportements, points d'entrée).
4. Supprimer `.claude/.stale_skills` une fois la resynchronisation faite.

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
- **Secrets** : copier `.env.example` → `.env` (ignoré par git) et renseigner la clé
  du fournisseur effectivement utilisé. La clé requise dépend des modèles configurés
  dans `config.py` (fournisseur déduit du préfixe, cf. `llm.py`) : `MISTRAL_API_KEY`
  pour les modèles Mistral, `ANTHROPIC_API_KEY` pour les modèles Claude. Avec les
  valeurs par défaut (tout en `mistral-*`), seule `MISTRAL_API_KEY` est nécessaire.
  Chargé via `python-dotenv`.

## Conventions

- Commentaires et docstrings en **français**.
- Type hints sur les signatures publiques.
- Constantes de chemins centralisées dans `config.py` — ne pas coder de chemins en dur.
- Chargement des modèles (HF, embeddings, clients LLM) volontairement séparé de
  l'inférence (`import_model()` puis les méthodes de scoring/extraction de la classe) ;
  le chargement est coûteux (modèles HF) ou facturé (Claude / Mistral).
- **Méthode `run()`** : chaque système / orchestrateur / agent expose une méthode
  `run(*, limit=None, announcement=DEFAULT_ANNOUNCEMENT, category=DEFAULT_CV_CATEGORY,
  save=True)` (l'agent : `run(save=True)`) qui enchaîne `import_model()` → chargement
  des données (`ats_system.data.loaders`) → exécution → affichage → **sauvegarde**. C'est
  le point d'entrée des scripts (un seul appel). Les méthodes granulaires restent exposées
  et inchangées (réutilisées par `dataset_rankers.py`). `save=False` exécute sans rien écrire.
- **Sorties par script** : chaque `run()` écrit dans son propre dossier horodaté
  `results/<nom>/<YYYYMMDD-HHMMSS>/` (via `results_io.timestamped_run_dir`), avec des noms
  de fichier propres (`save_results(..., stamp_filename=False)`).

## Cartographie

> Les signatures détaillées vivent dans le code — voici les responsabilités et les
> points d'entrée publics.

### `src/ats_system/`

- **`config.py`** — chemins et constantes (`PROJECT_ROOT`, `DATA_DIR`,
  `ANNOUNCEMENTS_DIR`, `CV_DIR`, `RESULTS_DIR`, `DEFAULT_ANNOUNCEMENT`,
  `DEFAULT_CV_CATEGORY` = `"ENGINEERING"`, `DEFAULT_CV`,
  `SLIDING_WINDOW_MODEL` = `"mistral-small-latest"` — modèle du ranker,
  `ML6_KEYWORD_MODEL` = `"ml6team/keyphrase-extraction-kbir-inspec"` — modèle HF des
  systèmes/scripts mots-clés ml6 (extraction + tokenizer `count_tokens`),
  `GENERATED_DATA_DIR`, `CV_GENERATOR_MODEL` = `"mistral-small-latest"` — modèle
  du générateur de CVs, `CV_OPTIMIZER_MODEL` = `"mistral-small-latest"` —
  modèle Mistral de l'agent d'optimisation, `CV_OPTIMIZER_RANKER` = `"sliding_window"` —
  méthode de classement du dataset utilisée par l'outil de feedback de l'agent
  d'optimisation, parmi les clés de `CV_OPTIMIZER_RANKERS` : `"sliding_window"`,
  `"baseline_keyword"`, `"ml6_keyword"`, `"embedding_cosine"`, `LLM_REQUESTS_PER_SECOND` =
  `1.0` — débit cible par défaut des appels LLM (req/s) pour éviter les 429 ; calibré sur
  le free tier Mistral (~1 req/s), à augmenter sur un tier payant). Pour `SLIDING_WINDOW_MODEL` et
  `CV_GENERATOR_MODEL`, le **fournisseur** (Claude ou Mistral) est déduit du préfixe
  du nom de modèle (cf. `llm.py`) : changer la valeur suffit à basculer (ex.
  `"claude-haiku-4-5"` pour repasser le ranker sur Claude).
- **`llm.py`** — abstraction multi-fournisseurs partagée par le ranker et le
  générateur. `detect_provider()` (préfixe `claude-*` → Anthropic, sinon Mistral) et
  `LLMClient` (`import_model()` charge le bon SDK / la bonne clé, `complete()` fait une
  complétion texte simple et renvoie le texte, quel que soit le fournisseur). `LLMClient`
  accepte un `rate_limiter` optionnel (`BaseRateLimiter` de `langchain_core`, **appliqué
  aux seuls modèles Mistral** : `acquire()` avant chaque requête — partager une même
  instance entre clients plafonne le débit global) et `max_retries` (retry sur 429 avec
  backoff exponentiel). Ce limiteur est propagé via `SlidingWindowCVRanker`,
  `HybridMl6SlidingWindowRanker` et les `DatasetRanker` (cf. `dataset_rankers.py`).
- **`data/`** (réexporté par `ats_system.data`) — I/O des données :
  - `pdf_loader.py` — `import_pdf()` : PDF → `{"id", "content"}` (via `pypdf`).
  - `loaders.py` — `load_announcement(path)` et `load_cvs(category, limit)` : chargement
    par défaut de l'annonce et des CVs d'une catégorie (mutualise le glob `*.pdf` jusque-là
    dupliqué dans les scripts ; utilisé par les méthodes `run()`).
  - `pdf_writer.py` — `write_text_pdf(text, path)` : écrit un texte en PDF (Helvetica,
    latin-1). Mutualisé entre le générateur de CVs et l'agent (PDF du CV optimisé).
- **`results_io.py`** — schéma commun de sauvegarde des classements :
  `build_ranking()` (paires `(cv_id, score)` triées → entrées
  `{rank, cv_id, score[, justification]}`), `save_results()` (JSON sous `RESULTS_DIR`,
  jamais écrasé ; champs spécifiques via `extra` ; `stamp_filename=False` pour un nom de
  fichier sans horodatage dans un dossier déjà daté) et `timestamped_run_dir(name)` (crée
  `results/<name>/<horodatage>/`).
- **`systems/`** (réexporté par `ats_system.systems`) — un système ATS = une classe
  par fichier (convention `import_model()` puis inférence + `run()` ; mots-clés/embeddings
  exposent aussi `score_cvs()`). Baseline/ml6 mots-clés, embeddings cosinus, fenêtre
  glissante LLM, hybride, et l'orchestrateur `AllRankingsRunner`.
  → détails : `.claude/skills/systems.md`
- **`generators/`** (réexporté par `ats_system.generators`) — `SyntheticCVGenerator` :
  génère des CVs synthétiques PDF face à une annonce (niveaux de profil + CV « à optimiser »).
  → détails : `.claude/skills/generators.md`
- **`agents/`** (réexporté par `ats_system.agents`) — agents LangChain / LangGraph :
  `CVOptimizerAgent` (ReAct Mistral, réécrit un CV via le feedback de rang) et
  `dataset_rankers.py` (couche d'adaptation `DatasetRanker` autour des systèmes).
  → détails : `.claude/skills/agents.md`

### `scripts/` — points d'entrée (`python scripts/<nom>.py`)

Scripts minces : parsing d'arguments puis un seul appel `run()` (toute la logique vit dans
les classes).
→ détails : `.claude/skills/scripts.md`

### `notebooks/` — analyses exploratoires (`jupyter`)

- `ranking_similarity_rbo.ipynb` — compare les classements d'un run
  `results/all_rankings/<timestamp>/` via le **Rank-Based Overlap** (RBO, Webber et al.
  2010), la fenêtre glissante (LLM) servant de référence.
- `token_distribution.ipynb` — distribution du nombre de tokens des CVs.

### `data/` — données (non versionné)

- `data/announcement/` — annonces PDF (actuellement `mechanical_engineer_job_posting_2016.pdf`).
- `data/cv/<CATÉGORIE>/` — CVs PDF par métier (ACCOUNTANT, …, ENGINEERING, …, TEACHER).
- `data/generated_data/synthetic_cvs_<timestamp>/` — CVs synthétiques générés
  (`SyntheticCVGenerator`) + `manifest.json` de vérité-terrain.

### `results/` — sorties de classement

Un sous-dossier horodaté par script/run : `results/<nom>/<timestamp>/` (cf.
`results_io.timestamped_run_dir` + `save_results`). Ex. `results/all_rankings/<ts>/`,
`results/hybrid_ranking/<ts>/`, `results/cv_optimizer/<ts>/`.

## Garde-fous

- **`notes/`** — ⚠️ notes personnelles de l'utilisateur. **Ne pas lire, modifier ou supprimer.**
- **`ats_syst/`** — environnement virtuel. Ne pas versionner.
