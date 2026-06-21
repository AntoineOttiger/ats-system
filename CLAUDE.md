# Projet : ats-system

Système ATS (Applicant Tracking System) : scoring/classement de CVs (PDF) face à
une offre d'emploi, via trois approches comparées — mots-clés, embeddings, et un
LLM (Claude **ou** Mistral, fournisseur déduit du nom de modèle) en fenêtre glissante.

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
  `"baseline_keyword"`, `"ml6_keyword"`, `"embedding_cosine"`). Pour `SLIDING_WINDOW_MODEL` et
  `CV_GENERATOR_MODEL`, le **fournisseur** (Claude ou Mistral) est déduit du préfixe
  du nom de modèle (cf. `llm.py`) : changer la valeur suffit à basculer (ex.
  `"claude-haiku-4-5"` pour repasser le ranker sur Claude).
- **`llm.py`** — abstraction multi-fournisseurs partagée par le ranker et le
  générateur. `detect_provider()` (préfixe `claude-*` → Anthropic, sinon Mistral) et
  `LLMClient` (`import_model()` charge le bon SDK / la bonne clé, `complete()` fait une
  complétion texte simple et renvoie le texte, quel que soit le fournisseur).
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
  par fichier, convention commune `import_model()` (chargement) puis inférence, plus une
  méthode `run()` (pipeline complet + sauvegarde, cf. Conventions). Les systèmes mots-clés
  / embeddings exposent en plus `score_cvs(offre_text, cvs)` (boucle de scoring → paires
  `(cv_id, score)` triées, réutilisée par `run()` et `AllRankingsRunner`) :
  - `baseline_keyword_match.py` — `BaselineKeywordMatcher` : mots-clés baseline
    (regex + stopwords FR/EN). `import_model()` (stopwords nltk),
    `extract_keywords()`, `match()` (statique → `{"score", "matching", "missing"}`),
    `score_cvs()`, `run()` → `results/baseline_keyword_match/<ts>/`.
  - `ml6_keyword_match.py` — `Ml6KeywordMatcher` : mots-clés via le modèle HF
    `ml6team/keyphrase-extraction-kbir-inspec` (BERT, token classification ; chunks
    de 400 mots pour la limite 512 tokens). `import_model()`, `extract_keywords()`,
    `match()`, `score_cvs()`, `run()` → `results/ml6_keyword_match/<ts>/`.
  - `embedding_cosine.py` — `EmbeddingCosineScorer` : similarité cosinus offre/CV
    (0–100) via `SentenceTransformer` `all-MiniLM-L6-v2`. `import_model()`, `score()`,
    `score_cvs()`, `run()` → `results/embedding_cosine/<ts>/`.
  - `sliding_window_ranker.py` — `SlidingWindowCVRanker` : classement par fenêtre
    glissante (inspiré de RankGPT) via un LLM (modèle `SLIDING_WINDOW_MODEL` de
    `config.py` ; fournisseur Claude **ou** Mistral déduit du préfixe, appelé via
    `LLMClient` de `llm.py`). Entrées : `import_model()`, `load_cvs()`,
    `run_sliding_window_ranking()` → `RankingResult`, `run()` →
    `results/sliding_window_ranking/<ts>/`, `display_results()`. Le `RankingResult`
    expose en plus l'historique par passe (`pass_orders`, `pass_justifications` : ordre
    des cv_ids + justifications à la fin de chaque passe).
  - `hybrid_ml6_sliding_window.py` — `HybridMl6SlidingWindowRanker` : système hybride
    deux étapes. (1) Présélection mots-clés via `Ml6KeywordMatcher` (local, gratuit) sur
    tous les CVs, fournissant l'ordre de départ ; (2) affinage LLM par fenêtre glissante
    (`SlidingWindowCVRanker`, modèle `SLIDING_WINDOW_MODEL`) sur **tous** les CVs, en
    partant de cet ordre mots-clés. `import_model()` (charge les deux systèmes) puis
    `rank()` → `HybridRankingResult` (classement final + historique : classement
    mots-clés, `RankingResult` de la fenêtre glissante avec son historique par passe),
    `run()` (écrit tout l'historique dans `results/hybrid_ranking/<ts>/` :
    `ml6_keyword_match.json`, `sliding_window_pass{N}.json`, `hybrid_ranking.json`),
    `display_results()`.
  - `all_rankings_runner.py` — `AllRankingsRunner` : orchestrateur qui charge les données
    **une fois** et lance les **quatre** systèmes ci-dessus (en réutilisant leur
    `score_cvs()` / `run_sliding_window_ranking()`). `import_model()` (charge les quatre)
    puis `run()` → un JSON par méthode dans un même `results/all_rankings/<ts>/`.
- **`generators/`** (réexporté par `ats_system.generators`) :
  - `synthetic_cv_generator.py` — `SyntheticCVGenerator` : génère des CVs
    synthétiques PDF face à une annonce via un LLM (modèle `CV_GENERATOR_MODEL` ;
    fournisseur Mistral **ou** Claude déduit du préfixe, appelé via `LLMClient` de
    `llm.py`). La proximité CV/annonce est pilotée par des niveaux de profil
    discrets (`PROFILE_LEVELS` : `perfect`, `strong`, `partial`, `unrelated`, avec
    `rank` de vérité-terrain). `generate_cvs()` produit en plus, par défaut, **un CV
    « à optimiser »** (`to_optimize`, `rank` 0, marqué `optimize: true` dans le
    manifest) : candidat excellent sur le fond mais au vocabulaire volontairement non
    aligné avec l'annonce (cas de test délibéré pour les méthodes mots-clés /
    embeddings). Sa consigne (`DEFAULT_OPTIMIZE_INSTRUCTION`) est personnalisable via
    `optimize_instruction=` ; désactivable via `include_optimize=False`. Entrées :
    `import_model()`, `generate_cv()`, `generate_cvs(..., save=True)` → si `save`, écrit les
    PDF (via `data.pdf_writer.write_text_pdf`) + un `manifest.json` sous
    `GENERATED_DATA_DIR/synthetic_cvs_<timestamp>/` et renvoie ce dossier ; si `save=False`,
    n'écrit rien et renvoie la liste des CVs générés en mémoire.

- **`agents/`** (réexporté par `ats_system.agents`) — agents LangChain / LangGraph :
  - `cv_optimizer_agent.py` — `CVOptimizerAgent` : agent ReAct (Mistral, modèle
    `CV_OPTIMIZER_MODEL`, stack `langchain` / `langgraph` / `langchain-mistralai`) dont la
    mission est de **réécrire un CV** pour le faire remonter dans le classement face aux
    autres CVs d'un dataset synthétique, **sans inventer** de qualifications (reformulation /
    mise en valeur de la substance réelle face à l'annonce). L'agent reçoit l'annonce complète
    et **décide seul** comment réécrire le CV. Le signal de feedback est le **rang compétitif**
    du CV dans le dataset, calculé par la méthode choisie via `CV_OPTIMIZER_RANKER` (cf.
    `dataset_rankers.py`) face à l'annonce. Unique outil exposé : `rank_cv_in_dataset(cv_text)`
    (rang + classement complet + analyse de ce CV). ⚠️ Avec le ranker par défaut
    `sliding_window`, chaque appel d'outil relance un classement LLM complet du dataset (coûteux) ;
    les rankers mots-clés/embeddings sont locaux et gratuits. Convention `import_model()` (charge
    le LLM Mistral, le ranker du dataset et met en cache les CVs concurrents) puis `stream()`
    (trace des « pensées ») / `optimize()` → texte du CV optimisé. `run(save=True)` localise
    seul le CV « à optimiser » du dataset, l'optimise (en affichant la trace) et, si `save`,
    écrit le PDF du CV optimisé (`cv_optimise.pdf`) + un `meta.json` (dataset, annonce, modèle,
    ranker, trace, rang initial/final) sous `results/cv_optimizer/<ts>/`. Nécessite `MISTRAL_API_KEY`
    (agent ; couvre aussi le ranker `sliding_window` tant que `SLIDING_WINDOW_MODEL` reste sur
    Mistral). Si `SLIDING_WINDOW_MODEL` est basculé sur Claude, `ANTHROPIC_API_KEY` est requise
    en plus (ranker).
  - `dataset_rankers.py` — couche d'adaptation : enveloppe chaque système de `systems/` derrière
    l'interface commune `DatasetRanker` (`import_model()` puis `rank()` → `DatasetRankResult` :
    rang du candidat parmi les concurrents + classement + analyse riche). Registre
    `CV_OPTIMIZER_RANKERS` + factory `build_dataset_ranker(name, ...)` pilotés par
    `CV_OPTIMIZER_RANKER`. Les rankers mots-clés/embeddings transforment des scores par CV en rang
    (tri décroissant) ; seul `sliding_window` (LLM) utilise le limiteur de débit.

### `scripts/` — points d'entrée (`python scripts/<nom>.py`)

Chaque script est désormais **mince** : il parse ses arguments puis fait un seul appel
`run()` (toute la logique vit dans les classes). Les scripts de classement comparent les
approches sur les CVs `ENGINEERING` vs l'annonce par défaut (arg `--limit N` ; `0` = tous)
et acceptent `--save/--no-save` (sauvegarde par défaut activée) :

- `compute_kw_match_scores.py` — baseline mots-clés.
- `compute_ml6_kw_match_scores.py` — mots-clés ml6team.
- `compute_emb_scores.py` — embeddings.
- `compute_sliding_window_ranking.py` — classement par fenêtre glissante LLM
  (`--window-size`, `--passes`, `--model` ; nécessite la clé du fournisseur de
  `SLIDING_WINDOW_MODEL`, par défaut `MISTRAL_API_KEY`).
- `compute_all_rankings.py` — `AllRankingsRunner` : lance les **quatre** méthodes
  (baseline, ml6, embeddings, fenêtre glissante) sur le même run et écrit un JSON par
  méthode dans `results/all_rankings/<timestamp>/` (`--limit`, `--window-size`, `--passes`).
- `compute_hybrid_ranking.py` — `HybridMl6SlidingWindowRanker` : présélection mots-clés
  (ml6) puis affinage LLM par fenêtre glissante (`--limit`, `--window-size`, `--passes`,
  `--model`). Écrit **tout l'historique** dans `results/hybrid_ranking/<timestamp>/` :
  `ml6_keyword_match.json` (présélection), `sliding_window_pass{N}.json` (après chaque
  passe) et `hybrid_ranking.json` (classement final). Nécessite la clé du fournisseur de
  `SLIDING_WINDOW_MODEL` (par défaut `MISTRAL_API_KEY`).
- `count_tokens.py` / `count_tokens_stats.py` — statistiques de tokens (tokenizer ml6team).
  Hors périmètre `run()` (ni système ni classement) : inchangés, n'écrivent rien.
- `generate_synthetic_cvs.py` — génère `--count N` CVs synthétiques PDF face à une
  annonce (`--announcement`, `--model`, `--temperature`) via `SyntheticCVGenerator`
  (nécessite `MISTRAL_API_KEY`). Génère aussi un CV « à optimiser » (désactivable via
  `--no-optimize`, consigne personnalisable via `--optimize-prompt`). `--save/--no-save`
  (défaut : écrit) → sorties sous `data/generated_data/synthetic_cvs_<timestamp>/`.
- `launch_cv_optimizer_agent.py` — lance le `CVOptimizerAgent` sur le CV « à optimiser »
  d'un dataset synthétique (`--dataset`, défaut : le plus récent ; `--announcement`,
  `--model`, `--max-iterations`, `--save/--no-save`). Via `agent.run()` : affiche les
  pensées et écrit `cv_optimise.pdf` + `meta.json` sous `results/cv_optimizer/<timestamp>/`.
  Nécessite `MISTRAL_API_KEY`.

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
