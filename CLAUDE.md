# ats-system

Scoring/classement de CVs (PDF) face à une offre, via 3 approches : mots-clés,
embeddings, LLM fenêtre glissante (Claude **ou** Mistral selon préfixe du modèle, cf. `llm.py`).

## Skills (`.claude/skills/`) — lire à la demande, **avant** de toucher au module

- `systems.md` → systèmes ATS — détails : `.claude/skills/systems.md`
- `generators.md` → génération CVs synthétiques — détails : `.claude/skills/generators.md`
- `agents.md` → agent d'optimisation + `DatasetRanker` — détails : `.claude/skills/agents.md`
- `scripts.md` → points d'entrée CLI — détails : `.claude/skills/scripts.md`

## Hook `PostToolUse` (`sync_skills.py`)

Sur `Edit|Write`, détecte si le **fichier édité** appartient à un module documenté (mapping
dossier→skill lu sur les lignes `→ détails : …` ci-dessus). **Non bloquant** : signale le
skill périmé **une seule fois** via un rappel `[sync_skills] …` (`additionalContext`) et
l'inscrit dans `.claude/.stale_skills`. **Quand ce rappel apparaît** : à l'occasion, relire
le code du module, mettre à jour `.claude/skills/<nom>.md`, puis supprimer `.stale_skills`.

## Dev

- Layout `src/`, package `ats_system` éditable → imports **absolus** partout. Python ≥ 3.10.
- `pip install -e ".[dev]"` · lint `ruff` (line-length 120).
- `pytest` ne collecte que `tests/test_import_pdf.py` ; les autres `tests/*.py` sont des
  scripts CLI (`python tests/<nom>.py`).
- Secrets : `.env.example` → `.env`. Clé selon préfixe modèle (`mistral-*`→`MISTRAL_API_KEY`,
  `claude-*`→`ANTHROPIC_API_KEY`) ; par défaut Mistral seul.

## Conventions

- Docstrings/commentaires **français**, type hints publics.
- Chemins/constantes dans `config.py` — jamais en dur. Changer un modèle = changer sa valeur (fournisseur suit le préfixe).
- Chargement modèles (`import_model()`) séparé de l'inférence.
- Point d'entrée unique `run(*, limit=None, announcement=DEFAULT_ANNOUNCEMENT, category=DEFAULT_CV_CATEGORY, save=True)`
  (agent : `run(save=True)`) : import → données → exécution → sauvegarde. `save=False` n'écrit rien.
- Sorties dans `results/<nom>/<YYYYMMDD-HHMMSS>/` (`timestamped_run_dir`).

## Structure

`src/ats_system/` : `config.py`, `llm.py`, `data/`, `results_io.py`, `systems/`,
`generators/`, `agents/` (3 derniers détaillés dans les skills). `scripts/` = args + un
`run()`. `data/`, `results/` non versionnés.

## Garde-fous

- **`notes/`** : personnel — ne pas lire/modifier/supprimer.
- **`ats_syst/`** : venv — ne pas versionner.
