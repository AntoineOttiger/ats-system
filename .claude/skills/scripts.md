# Dossier `scripts/` — points d'entrée (`python scripts/<nom>.py`)

Chemin : `scripts/`.

Chaque script est **mince** : il parse ses arguments puis fait un seul appel `run()`
(toute la logique vit dans les classes). Les scripts de classement comparent les approches
sur les CVs `ENGINEERING` vs l'annonce par défaut (arg `--limit N` ; `0` = tous) et
acceptent `--save/--no-save` (sauvegarde par défaut activée).

## Scripts

- **`compute_kw_match_scores.py`** — baseline mots-clés.
- **`compute_ml6_kw_match_scores.py`** — mots-clés ml6team.
- **`compute_emb_scores.py`** — embeddings.
- **`compute_sliding_window_ranking.py`** — classement par fenêtre glissante LLM
  (`--window-size`, `--passes`, `--model` ; nécessite la clé du fournisseur de
  `SLIDING_WINDOW_MODEL`, par défaut `MISTRAL_API_KEY`).
- **`compute_all_rankings.py`** — `AllRankingsRunner` : lance les **quatre** méthodes
  (baseline, ml6, embeddings, fenêtre glissante) sur le même run et écrit un JSON par méthode
  dans `results/all_rankings/<timestamp>/` (`--limit`, `--window-size`, `--passes`).
- **`compute_hybrid_ranking.py`** — `HybridMl6SlidingWindowRanker` : présélection mots-clés
  (ml6) puis affinage LLM par fenêtre glissante (`--limit`, `--window-size`, `--passes`,
  `--model`). Écrit **tout l'historique** dans `results/hybrid_ranking/<timestamp>/` :
  `ml6_keyword_match.json` (présélection), `sliding_window_pass{N}.json` (après chaque passe)
  et `hybrid_ranking.json` (classement final). Nécessite la clé du fournisseur de
  `SLIDING_WINDOW_MODEL` (par défaut `MISTRAL_API_KEY`).
- **`count_tokens.py`** / **`count_tokens_stats.py`** — statistiques de tokens (tokenizer
  ml6team). Hors périmètre `run()` (ni système ni classement) : inchangés, n'écrivent rien.
- **`generate_synthetic_cvs.py`** — génère `--count N` CVs synthétiques PDF face à une annonce
  (`--announcement`, `--model`, `--temperature`) via `SyntheticCVGenerator` (nécessite
  `MISTRAL_API_KEY`). Génère aussi un CV « à optimiser » (désactivable via `--no-optimize`,
  consigne personnalisable via `--optimize-prompt`). `--save/--no-save` (défaut : écrit) →
  sorties sous `data/generated_data/synthetic_cvs_<timestamp>/`.
- **`launch_cv_optimizer_agent.py`** — lance le `CVOptimizerAgent` sur le CV « à optimiser »
  d'un dataset synthétique (`--dataset`, défaut : le plus récent ; `--announcement`, `--model`,
  `--max-iterations`, `--save/--no-save`). Via `agent.run()` : affiche les pensées et écrit
  `cv_optimise.pdf` + `meta.json` sous `results/cv_optimizer/<timestamp>/`. Nécessite
  `MISTRAL_API_KEY`.
