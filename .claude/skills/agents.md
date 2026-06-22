# Module `agents/`

Chemin : `src/ats_system/agents/` (réexporté par `ats_system.agents`).
Agents LangChain / LangGraph.

## Classes

- **`cv_optimizer_agent.py`** — `CVOptimizerAgent` : agent ReAct (Mistral, modèle
  `CV_OPTIMIZER_MODEL`, stack `langchain` / `langgraph` / `langchain-mistralai`) dont la
  mission est de **réécrire un CV** pour le faire remonter dans le classement face aux
  autres CVs d'un dataset synthétique, **sans inventer** de qualifications (reformulation /
  mise en valeur de la substance réelle face à l'annonce). L'agent reçoit l'annonce complète
  et **décide seul** comment réécrire le CV. Le signal de feedback est le **rang compétitif**
  du CV dans le dataset, calculé par la méthode choisie via `CV_OPTIMIZER_RANKER` (cf.
  `dataset_rankers.py`) face à l'annonce — par défaut `ml6_keyword` (mots-clés ml6, local et
  gratuit).

  Unique outil exposé : `rank_cv_in_dataset(cv_text)` (rang + classement complet + analyse
  de ce CV). ⚠️ Avec le ranker `sliding_window`, chaque appel d'outil relance un classement
  LLM complet du dataset (coûteux) ; les rankers mots-clés/embeddings sont locaux et gratuits.

  Convention `import_model()` (charge le LLM Mistral, le ranker du dataset, construit un
  limiteur de débit partagé `InMemoryRateLimiter` — un seul token bucket plafonne **tous** les
  appels Mistral du processus, agent + ranker `sliding_window`, à `requests_per_second`
  (défaut `LLM_REQUESTS_PER_SECOND`) — et met en cache les CVs concurrents) puis `stream()`
  (trace des « pensées ») / `optimize()` → texte du CV optimisé.

  `run(save=True)` localise seul le CV « à optimiser » du dataset, l'optimise (en affichant la
  trace) et, si `save`, écrit le PDF du CV optimisé (`cv_optimise.pdf`) + un `meta.json`
  (dataset, annonce, modèle, ranker, trace, rang initial/final) sous `results/cv_optimizer/<ts>/`.

  Nécessite `MISTRAL_API_KEY` (agent ; couvre aussi le ranker `sliding_window` tant que
  `SLIDING_WINDOW_MODEL` reste sur Mistral). Si `SLIDING_WINDOW_MODEL` est basculé sur Claude,
  `ANTHROPIC_API_KEY` est requise en plus (ranker).

- **`one_shot_cv_optimizer.py`** — `OneShotCVOptimizer` : **baseline non itérative** face au
  `CVOptimizerAgent`. Réécrit le CV « à optimiser » d'un dataset synthétique en **un seul
  appel LLM**, **à l'aveugle** (prompt = annonce + CV uniquement, mêmes garde-fous que l'agent :
  interdiction d'inventer, sortie = CV seul), via `LLMClient` directement (pas de
  LangGraph/ReAct, fournisseur déduit du préfixe de `CV_OPTIMIZER_MODEL`). Le rang du CV est
  mesuré **avant et après** la réécriture par le ranker choisi (`CV_OPTIMIZER_RANKER`) — pour
  le rapport seulement, **jamais montré au LLM**. Convention `import_model()` (charge
  `LLMClient`, le ranker, construit un `InMemoryRateLimiter` partagé client+ranker, met en
  cache les CVs concurrents) puis `optimize(cv_text)` → texte du CV optimisé. `run(save=True)`
  localise seul le CV « à optimiser », mesure le rang initial, réécrit, re-mesure, et si
  `save` écrit `cv_optimise.pdf` + `meta.json` (dataset, annonce, modèle, ranker,
  `rang_initial`/`rang_final`/`total`, sans trace) sous `results/cv_optimizer_oneshot/<ts>/`.
  Nécessite la clé du fournisseur du modèle (par défaut `MISTRAL_API_KEY` ; plus
  `ANTHROPIC_API_KEY` si le ranker `sliding_window` tourne sur un modèle Claude).

- **`dataset_loading.py`** — helpers mutualisés (utilisés par `CVOptimizerAgent` et
  `OneShotCVOptimizer`) pour lire un dataset synthétique via son `manifest.json` :
  `find_optimize_cv(dataset_dir)` → `(nom_fichier, texte)` du CV `optimize: true`, et
  `load_competitors(dataset_dir)` → liste de dicts `{"id", "content"}` des autres CVs.

- **`dataset_rankers.py`** — couche d'adaptation : enveloppe chaque système de `systems/`
  derrière l'interface commune `DatasetRanker` (`import_model()` puis `rank()` →
  `DatasetRankResult` : rang du candidat parmi les concurrents + classement + analyse riche).
  Registre `CV_OPTIMIZER_RANKERS` + factory `build_dataset_ranker(name, ...)` pilotés par
  `CV_OPTIMIZER_RANKER`. Clés : `sliding_window` (LLM holistique), `hybrid_ml6_sliding_window`
  (présélection mots-clés ml6 puis affinage fenêtre glissante LLM — adaptateur
  `HybridDatasetRanker`, analyse = score ml6 + justification LLM), `baseline_keyword`,
  `ml6_keyword`, `embedding_cosine`. Les rankers mots-clés/embeddings transforment des scores
  par CV en rang (tri décroissant) ; `sliding_window` et `hybrid_ml6_sliding_window` (qui
  appellent le LLM) utilisent le limiteur de débit partagé.
