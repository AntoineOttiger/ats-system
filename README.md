# ats-system

Système ATS (Applicant Tracking System) : calcule la correspondance entre une
offre d'emploi et des CVs (PDF) selon plusieurs méthodes de scoring.

## Installation

```bash
python -m venv ats_syst
ats_syst\Scripts\activate        # Windows
pip install -e ".[dev]"
```

## Structure

```
src/ats_system/      # package importable
  config.py          # chemins & constantes centralisés
  data/              # I/O (chargement PDF)
  scoring/           # logique de scoring (keyword, embedding)
  models/            # wrappers LLM (keyphrase_extractor, embedding_model)
scripts/             # points d'entrée exécutables
notebooks/           # notebooks Jupyter d'exploration
tests/               # tests pytest
data/                # données (non versionné) : announcements/, cv/
notes/               # notes personnelles (ne pas modifier)
```

## Méthodes de scoring

- **keyword** : intersection de mots-clés (stopwords FR+EN supprimés).
- **ml6 keyword** : extraction de keyphrases via le LLM `ml6team/keyphrase-extraction-kbir-inspec`.
- **embedding** : similarité cosinus des embeddings `all-MiniLM-L6-v2`.

## Utilisation

```bash
python scripts/compute_kw_match_scores.py
python scripts/compute_ml6_kw_match_scores.py --limit 5
python scripts/compute_emb_scores.py --limit 0
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
