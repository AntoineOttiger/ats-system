# Module `generators/`

Chemin : `src/ats_system/generators/` (réexporté par `ats_system.generators`).

## Classes

- **`synthetic_cv_generator.py`** — `SyntheticCVGenerator` : génère des CVs synthétiques
  PDF face à une annonce via un LLM (modèle `CV_GENERATOR_MODEL` ; fournisseur Mistral
  **ou** Claude déduit du préfixe, appelé via `LLMClient` de `llm.py`). La proximité
  CV/annonce est pilotée par des **sets de niveaux de profil** sélectionnables via le
  paramètre `profile_set` de `generate_cvs()` :

  - `"default"` (défaut) — `PROFILE_LEVELS` : niveaux hétérogènes `perfect` (rank 0),
    `strong` (1), `partial` (2), `unrelated` (3).
  - `"top"` — `PROFILE_LEVELS_TOP` : 5 variantes top-niveau (toutes rank 0) —
    `top_technical`, `top_leadership`, `top_academic`, `top_veteran`, `top_innovator`.
    Produit un lot homogène de CVs excellents mais stylistiquement variés.

  Tous les sets sont indexés dans `PROFILE_SETS: dict[str, tuple[ProfileLevel, ...]]`.
  Le paramètre `levels` filtre toujours dans le set choisi.

  `generate_cvs()` produit en plus, par défaut, **un CV « à optimiser »** (`to_optimize`,
  `rank` 0, marqué `optimize: true` dans le manifest) : candidat excellent sur le fond mais
  au vocabulaire volontairement non aligné avec l'annonce (cas de test délibéré pour les
  méthodes mots-clés / embeddings). Sa consigne (`DEFAULT_OPTIMIZE_INSTRUCTION`) est
  personnalisable via `optimize_instruction=` ; désactivable via `include_optimize=False`.

  Signatures publiques :
  - `import_model()` — initialise le client LLM.
  - `generate_cv(announcement, level)` — génère un CV pour un `ProfileLevel` donné.
  - `generate_cvs(announcement, n, *, levels=None, profile_set="default", run_name=None,
    output_dir=GENERATED_DATA_DIR, include_optimize=True, optimize_instruction=None,
    save=True)` → si `save`, écrit les PDF (via `data.pdf_writer.write_text_pdf`) +
    un `manifest.json` (champ `params.profile_set` inclus) sous
    `GENERATED_DATA_DIR/synthetic_cvs_<timestamp>/` et renvoie ce dossier ; si
    `save=False`, n'écrit rien et renvoie la liste des CVs générés en mémoire.
