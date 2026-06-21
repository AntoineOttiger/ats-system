# Module `generators/`

Chemin : `src/ats_system/generators/` (réexporté par `ats_system.generators`).

## Classes

- **`synthetic_cv_generator.py`** — `SyntheticCVGenerator` : génère des CVs synthétiques
  PDF face à une annonce via un LLM (modèle `CV_GENERATOR_MODEL` ; fournisseur Mistral
  **ou** Claude déduit du préfixe, appelé via `LLMClient` de `llm.py`). La proximité
  CV/annonce est pilotée par des niveaux de profil discrets (`PROFILE_LEVELS` : `perfect`,
  `strong`, `partial`, `unrelated`, avec `rank` de vérité-terrain).

  `generate_cvs()` produit en plus, par défaut, **un CV « à optimiser »** (`to_optimize`,
  `rank` 0, marqué `optimize: true` dans le manifest) : candidat excellent sur le fond mais
  au vocabulaire volontairement non aligné avec l'annonce (cas de test délibéré pour les
  méthodes mots-clés / embeddings). Sa consigne (`DEFAULT_OPTIMIZE_INSTRUCTION`) est
  personnalisable via `optimize_instruction=` ; désactivable via `include_optimize=False`.

  Entrées : `import_model()`, `generate_cv()`, `generate_cvs(..., save=True)` → si `save`,
  écrit les PDF (via `data.pdf_writer.write_text_pdf`) + un `manifest.json` sous
  `GENERATED_DATA_DIR/synthetic_cvs_<timestamp>/` et renvoie ce dossier ; si `save=False`,
  n'écrit rien et renvoie la liste des CVs générés en mémoire.
