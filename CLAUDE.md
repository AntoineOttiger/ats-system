# Projet : ats-system

## Description

Ce projet consiste à implémenter un système ATS (Applicant Tracking System).

## Structure du projet

### `dataset/`

Contient les données du projet :

- `dataset/announcement/` — Annonces d'emploi au format PDF
  - `mechanical_engineer_job_posting_2016.pdf` (seule annonce actuelle)
- `dataset/cv/` — CVs au format PDF, organisés par catégorie métier :
  ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION,
  BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT,
  DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR,
  INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER

### `tools/`

Fonctions utilitaires réutilisables.

#### `tools/data_manager.py`
- `import_pdf(file_path: str) -> str` — Extrait et retourne le texte complet d'un fichier PDF (via `pypdf`).

#### `tools/scores.py`
- `keyword_match_score(offre: str, cv: str) -> dict` — Calcule la correspondance entre une offre et un CV en comparant les mots-clés (stopwords FR+EN supprimés). Retourne `{"score": float, "matching": set, "missing": set}`.

### `scr/`

Scripts exécutables.

- `scr/compute_keyword_match_scores.py` — Calcule le `keyword_match_score` de tous les CVs du dossier `ENGINEERING` contre l'annonce `mechanical_engineer_job_posting_2016.pdf`, et affiche les résultats triés du meilleur au plus bas score.

### `test/`

Scripts de test.

- `test/test_import_pdf.py` — Teste la fonction `import_pdf`.
- `test/test_keyword_match_score.py` — Teste `keyword_match_score` entre l'annonce et un CV (paramétrable via `--offre` et `--cv`).

### `ats_syst/`

Environnement virtuel Python. Ne pas versionner.

### `notes/`

> ⚠️ **Ne pas modifier.** Ce dossier contient les notes personnelles de l'utilisateur. Aucune lecture, modification ou suppression ne doit être effectuée sur son contenu.
