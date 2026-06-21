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

- **`dataset_rankers.py`** — couche d'adaptation : enveloppe chaque système de `systems/`
  derrière l'interface commune `DatasetRanker` (`import_model()` puis `rank()` →
  `DatasetRankResult` : rang du candidat parmi les concurrents + classement + analyse riche).
  Registre `CV_OPTIMIZER_RANKERS` + factory `build_dataset_ranker(name, ...)` pilotés par
  `CV_OPTIMIZER_RANKER`. Les rankers mots-clés/embeddings transforment des scores par CV en
  rang (tri décroissant) ; seul `sliding_window` (LLM) utilise le limiteur de débit.
