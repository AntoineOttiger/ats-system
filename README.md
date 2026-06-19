# ats-system

Système ATS (Applicant Tracking System) : calcule la correspondance entre une
offre d'emploi et des CVs (PDF) selon plusieurs méthodes de scoring.

## Installation

```bash
python -m venv ats_syst
ats_syst\Scripts\activate        # Windows
pip install -e ".[dev]"
```

La méthode par fenêtre glissante (Claude) nécessite une clé API : copier
`.env.example` → `.env` et renseigner `ANTHROPIC_API_KEY`.

## Structure

```
src/ats_system/      # package importable
  config.py          # chemins & constantes centralisés
  data/              # I/O (chargement PDF)
  scoring/           # logique de scoring (keyword, embedding)
  models/            # wrappers de modèles (keyphrase_extractor, embedding_model, sliding_window_ranker)
  results_io.py      # schéma commun & sauvegarde JSON des classements
scripts/             # points d'entrée exécutables
notebooks/           # notebooks Jupyter d'exploration
tests/               # tests pytest
data/                # données : announcements/ (versionné), cv/ (non versionné)
results/             # sorties de classement (JSON horodatés)
notes/               # notes personnelles (ne pas modifier)
```

Les CVs (`data/cv/`) proviennent du
[Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data)
(Kaggle) et ne sont pas versionnés.

## Méthodes de scoring

- **keyword** : intersection de mots-clés (stopwords FR+EN supprimés).
- **ml6 keyword** : extraction de keyphrases via le LLM `ml6team/keyphrase-extraction-kbir-inspec`.
- **embedding** : similarité cosinus des embeddings `all-MiniLM-L6-v2`.
- **fenêtre glissante** : classement par un LLM Claude (inspiré de RankGPT) ; nécessite `ANTHROPIC_API_KEY`.

## Utilisation

```bash
python scripts/compute_kw_match_scores.py
python scripts/compute_ml6_kw_match_scores.py --limit 5
python scripts/compute_emb_scores.py --limit 0
python scripts/compute_sliding_window_ranking.py --window-size 4 --passes 10   # nécessite ANTHROPIC_API_KEY
```

Pour exécuter les quatre méthodes en un seul run (un JSON par méthode dans
`results/all_rankings_<timestamp>/`) :

```bash
python scripts/compute_all_rankings.py --limit 30
```

### Tokens

```bash
python scripts/count_tokens.py --doc data/cv/ENGINEERING/12472574.pdf   # tokens d'un document
python scripts/count_tokens_stats.py                                    # moyenne/écart type des CVs ENGINEERING
```

## Notebooks

Le dossier `notebooks/` contient des notebooks d'exploration (nécessite `pip install -e ".[dev]"`
pour `matplotlib` / `ipykernel`, et le kernel doit pointer sur le venv `ats_syst`) :

- `token_distribution.ipynb` — distribution du nombre de tokens des CVs `ENGINEERING` (stats + histogramme).
- `ranking_similarity_rbo.ipynb` — similarité entre les classements d'un run `all_rankings_*` via le
  Rank-Based Overlap (RBO), la fenêtre glissante (Claude) servant de référence.
