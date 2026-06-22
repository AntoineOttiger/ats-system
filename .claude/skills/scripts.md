# Dossier `scripts/` — points d'entrée (`python scripts/<nom>.py`)

Chemin : `scripts/`.

Chaque script est **mince** : il parse ses arguments puis fait un seul appel `run()`
(toute la logique vit dans les classes). Les scripts de classement acceptent :
- `--limit N` (0 = tous)
- `--announcement PATH` (défaut : `DEFAULT_ANNOUNCEMENT` de `config.py`)
- `--cv-dir PATH` (défaut : `DEFAULT_CV_DIR` = `data/cv/ENGINEERING/`) — permet d'utiliser
  n'importe quel dossier de CVs PDF, y compris les datasets synthétiques
- `--save/--no-save` (sauvegarde activée par défaut)

## Scripts

- **`compute_kw_match_scores.py`** — baseline mots-clés (`--limit`, `--announcement`, `--cv-dir`).
- **`compute_ml6_kw_match_scores.py`** — mots-clés ml6team (`--limit`, `--announcement`, `--cv-dir`).
- **`compute_emb_scores.py`** — embeddings (`--limit`, `--announcement`, `--cv-dir`).
- **`compute_sliding_window_ranking.py`** — classement par fenêtre glissante LLM
  (`--limit`, `--announcement`, `--cv-dir`, `--window-size`, `--passes`, `--model` ;
  nécessite la clé du fournisseur de `SLIDING_WINDOW_MODEL`, par défaut `MISTRAL_API_KEY`).
- **`compute_all_rankings.py`** — `AllRankingsRunner` : lance les **quatre** méthodes
  (baseline, ml6, embeddings, fenêtre glissante) sur le même run et écrit un JSON par méthode
  dans `results/all_rankings/<timestamp>/` (`--limit`, `--announcement`, `--cv-dir`,
  `--window-size`, `--passes`).
- **`compute_hybrid_ranking.py`** — `HybridMl6SlidingWindowRanker` : présélection mots-clés
  (ml6) puis affinage LLM par fenêtre glissante (`--limit`, `--announcement`, `--cv-dir`,
  `--window-size`, `--passes`, `--model`). Écrit **tout l'historique** dans
  `results/hybrid_ranking/<timestamp>/` : `ml6_keyword_match.json` (présélection),
  `sliding_window_pass{N}.json` (après chaque passe) et `hybrid_ranking.json` (classement final).
  Nécessite la clé du fournisseur de `SLIDING_WINDOW_MODEL` (par défaut `MISTRAL_API_KEY`).
- **`count_tokens.py`** / **`count_tokens_stats.py`** — statistiques de tokens (tokenizer
  ml6team). Hors périmètre `run()` (ni système ni classement) : n'écrivent rien.
  `count_tokens_stats.py` utilise `DEFAULT_CV_DIR` (plus `DEFAULT_CV_CATEGORY`) depuis `config.py`.
- **`generate_synthetic_cvs.py`** — génère `--count N` CVs synthétiques PDF face à une annonce
  (`--announcement`, `--model`, `--temperature`) via `SyntheticCVGenerator` (nécessite
  `MISTRAL_API_KEY`). Génère aussi un CV « à optimiser » (désactivable via `--no-optimize`,
  consigne personnalisable via `--optimize-prompt`). `--profile-set default|top` : set de
  niveaux de profil à utiliser (`default` = hétérogène, `top` = 5 variantes top-niveau ;
  défaut : `default`). `--save/--no-save` (défaut : écrit) →
  sorties sous `data/generated_data/synthetic_cvs_<timestamp>/`.
- **`launch_cv_optimizer_agent.py`** — lance le `CVOptimizerAgent` sur le CV « à optimiser »
  d'un dataset synthétique (`--dataset`, défaut : le plus récent ; `--announcement`, `--model`,
  `--max-iterations`, `--save/--no-save`). `--ranker` choisit la méthode de classement du
  feedback de rang (choix : clés de `CV_OPTIMIZER_RANKERS` ; défaut : `CV_OPTIMIZER_RANKER` de
  `config.py`). Via `agent.run()` : affiche les pensées et écrit `cv_optimise.pdf` +
  `meta.json` sous `results/cv_optimizer/<timestamp>/`. Nécessite `MISTRAL_API_KEY`.
- **`launch_one_shot_cv_optimizer.py`** — lance l'`OneShotCVOptimizer` (baseline non itérative)
  sur le CV « à optimiser » d'un dataset synthétique : réécriture en **un seul appel LLM** à
  l'aveugle, avec mesure du rang avant/après (`--dataset`, défaut : le plus récent ;
  `--announcement`, `--model`, `--ranker` parmi les clés de `CV_OPTIMIZER_RANKERS`,
  `--save/--no-save` ; **pas** de `--max-iterations`). Via `optimizer.run()` : écrit
  `cv_optimise.pdf` + `meta.json` (avec `rang_initial`/`rang_final`) sous
  `results/cv_optimizer_oneshot/<timestamp>/`. Nécessite la clé du fournisseur du modèle
  (par défaut `MISTRAL_API_KEY`).
