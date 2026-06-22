# Module `systems/`

Chemin : `src/ats_system/systems/` (réexporté par `ats_system.systems`).

Un système ATS = **une classe par fichier**, convention commune `import_model()`
(chargement) puis inférence, plus une méthode `run()` (pipeline complet + sauvegarde,
cf. Conventions de `CLAUDE.md`). Les systèmes mots-clés / embeddings exposent en plus
`score_cvs(offre_text, cvs)` (boucle de scoring → paires `(cv_id, score)` triées,
réutilisée par `run()` et `AllRankingsRunner`).

## Classes

- **`baseline_keyword_match.py`** — `BaselineKeywordMatcher` : mots-clés baseline
  (regex + stopwords FR/EN). `import_model()` (stopwords nltk), `extract_keywords()`,
  `match()` (statique → `{"score", "matching", "missing"}`), `score_cvs()`, `run()` →
  `results/baseline_keyword_match/<ts>/`.

- **`ml6_keyword_match.py`** — `Ml6KeywordMatcher` : mots-clés via le modèle HF
  `ml6team/keyphrase-extraction-kbir-inspec` (BERT, token classification ; chunks de
  400 mots pour la limite 512 tokens). `import_model()`, `extract_keywords()`, `match()`,
  `score_cvs()`, `run()` → `results/ml6_keyword_match/<ts>/`.

- **`embedding_cosine.py`** — `EmbeddingCosineScorer` : similarité cosinus offre/CV
  (0–100) via `SentenceTransformer` `all-MiniLM-L6-v2`. `import_model()`, `score()`,
  `score_cvs()`, `run()` → `results/embedding_cosine/<ts>/`.

- **`sliding_window_ranker.py`** — `SlidingWindowCVRanker` : classement par fenêtre
  glissante (inspiré de RankGPT) via un LLM (modèle `SLIDING_WINDOW_MODEL` de `config.py` ;
  fournisseur Claude **ou** Mistral déduit du préfixe, appelé via `LLMClient` de `llm.py`).
  Entrées : `import_model()`, `load_cvs()`, `run_sliding_window_ranking()` → `RankingResult`,
  `run()` → `results/sliding_window_ranking/<ts>/`, `display_results()`. Le `RankingResult`
  expose en plus l'historique par passe (`pass_orders`, `pass_justifications` : ordre des
  cv_ids + justifications à la fin de chaque passe).

- **`hybrid_ml6_sliding_window.py`** — `HybridMl6SlidingWindowRanker` : système hybride
  deux étapes. (1) Présélection mots-clés via `Ml6KeywordMatcher` (local, gratuit) sur tous
  les CVs, fournissant l'ordre de départ ; (2) affinage LLM par fenêtre glissante
  (`SlidingWindowCVRanker`, modèle `SLIDING_WINDOW_MODEL`) sur **tous** les CVs, en partant
  de cet ordre mots-clés. `import_model()` (charge les deux systèmes) puis `rank()` →
  `HybridRankingResult` (classement final + historique : classement mots-clés, `RankingResult`
  de la fenêtre glissante avec son historique par passe), `run()` (écrit tout l'historique
  dans `results/hybrid_ranking/<ts>/` : `ml6_keyword_match.json`, `sliding_window_pass{N}.json`,
  `hybrid_ranking.json`), `display_results()`.

- **`all_rankings_runner.py`** — `AllRankingsRunner` : orchestrateur qui charge les données
  **une fois** et lance les **quatre** systèmes ci-dessus (en réutilisant leur `score_cvs()` /
  `run_sliding_window_ranking()`). `import_model()` (charge les quatre) puis `run()` → un JSON
  par méthode dans un même `results/all_rankings/<ts>/`.

- **`hrflow_ranker.py`** — `HrflowCVRanker` : scoring via l'API HrFlow (parsing PDF +
  scoring cloud, SDK `hrflow` v4). Flow dans `run()` :
  (1) `_store_job(text)` → indexe l'offre via `job.storing.add_json` (texte dans `summary`,
  référence = hash MD5 pour upsert idempotent) → `job_key` ;
  (2) `_upload_cv(cv_path)` × N → `profile.parsing.add_file` (PDF binaire,
  `reference=cv_path.name`, `sync_parsing_indexing=1` pour indexation auto) ;
  (3) pause `index_wait` secondes (défaut 10 s, traitement asynchrone côté HrFlow) ;
  (4) `profile.scoring.list(source_keys, board_key, job_key, limit)` → scores, filtrés
  par référence pour ne garder que le batch courant, normalisés 0–1 → 0–100.
  `import_model()` charge `Hrflow` (import tardif) et lit `HRFLOW_API_KEY`,
  `HRFLOW_API_USER`, `HRFLOW_SOURCE_KEY`, `HRFLOW_BOARD_KEY` depuis `.env`.
  Charge les CVs depuis les **chemins PDF** directement (`CV_DIR / category / *.pdf`),
  pas via `load_cvs()` — nécessaire pour l'upload binaire.
  N'expose **pas** `score_cvs()` (incompatible avec l'approche fichier).
  `run()` → `results/hrflow_ranking/<ts>/`.
