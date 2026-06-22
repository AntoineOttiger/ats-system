"""Génération de CVs synthétiques face à une annonce, via l'API Mistral.

À partir d'une offre d'emploi, le générateur produit des CVs PDF dont la
pertinence vis-à-vis de l'annonce est **connue à l'avance** : chaque CV est généré
pour un *niveau de profil* discret (du candidat idéal au profil hors-sujet). Cela
fournit une vérité-terrain réutilisable pour évaluer les méthodes de classement.

Le LLM est appelé via :class:`ats_system.llm.LLMClient` : le fournisseur (Mistral ou
Claude) est déduit du préfixe du modèle (cf. ``CV_GENERATOR_MODEL`` dans
``config.py``). La clé API est chargée depuis un fichier ``.env`` — jamais codée en
dur, comme pour le ``SlidingWindowCVRanker``.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ats_system.config import CV_GENERATOR_MODEL, GENERATED_DATA_DIR
from ats_system.data import write_text_pdf
from ats_system.llm import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Niveaux de profil — contrôlent la proximité CV / annonce
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProfileLevel:
    """Un niveau discret de correspondance entre le CV généré et l'annonce.

    ``rank`` donne l'ordre de pertinence attendu (0 = meilleur fit) : il sert de
    vérité-terrain pour évaluer les classements produits par les autres méthodes.
    ``instruction`` est la consigne (en français) injectée dans le prompt.
    """

    name: str
    rank: int
    instruction: str


# Ordonnés du meilleur au pire fit (rank croissant).
PROFILE_LEVELS: tuple[ProfileLevel, ...] = (
    ProfileLevel(
        name="perfect",
        rank=0,
        instruction=(
            "Generate the resume of the IDEAL candidate: they have every skill, "
            "experience and qualification required by the job posting, with a "
            "perfectly aligned career path and a level of experience at least "
            "equal to the one required."
        ),
    ),
    ProfileLevel(
        name="strong",
        rank=1,
        instruction=(
            "Generate the resume of a STRONG candidate: they master most of the "
            "required skills and work in the right field, but with a few gaps "
            "(a missing secondary skill or slightly less experience than "
            "requested)."
        ),
    ),
    ProfileLevel(
        name="partial",
        rank=2,
        instruction=(
            "Generate the resume of a PARTIALLY matching candidate: they come from "
            "a related field and possess only some of the required skills. Several "
            "key requirements of the job posting are not covered."
        ),
    ),
    ProfileLevel(
        name="unrelated",
        rank=3,
        instruction=(
            "Generate the resume of an UNRELATED candidate: a different occupation "
            "with no real connection to the job posting. The profile remains "
            "credible and realistic, but does not match the position."
        ),
    ),
)

PROFILE_LEVELS_BY_NAME: dict[str, ProfileLevel] = {lvl.name: lvl for lvl in PROFILE_LEVELS}

# Cinq variantes de candidats top-niveau (rank 0 pour toutes : vérité-terrain identique),
# avec des angles différents pour assurer la diversité stylistique du lot.
PROFILE_LEVELS_TOP: tuple[ProfileLevel, ...] = (
    ProfileLevel(
        name="top_technical",
        rank=0,
        instruction=(
            "Generate the resume of a TOP-TIER candidate with exceptional technical depth: "
            "they have mastered the exact technology stack required by the posting, can cite "
            "measurable achievements on recent technical projects (performance gains, system "
            "scale, delivery timelines) and hold relevant certifications or have contributed "
            "to respected open-source projects in the field."
        ),
    ),
    ProfileLevel(
        name="top_leadership",
        rank=0,
        instruction=(
            "Generate the resume of a TOP-TIER candidate who combines solid technical "
            "expertise with proven leadership: they satisfy all technical requirements of "
            "the posting AND have led cross-functional teams, mentored junior engineers, "
            "driven architectural decisions, and delivered results through others — not "
            "just individually."
        ),
    ),
    ProfileLevel(
        name="top_academic",
        rank=0,
        instruction=(
            "Generate the resume of a TOP-TIER candidate with an elite academic background: "
            "they graduated from a highly ranked institution (engineering school, research "
            "university) with honours, have publications or thesis work directly relevant "
            "to the position, and their early-career experience — internships and first "
            "roles — already demonstrates strong alignment with the job requirements."
        ),
    ),
    ProfileLevel(
        name="top_veteran",
        rank=0,
        instruction=(
            "Generate the resume of a TOP-TIER seasoned expert: a senior professional with "
            "15+ years of experience directly relevant to the posting, a track record of "
            "delivering large-scale projects, quantified impact (budget managed, headcount "
            "led, revenue influenced), and industry recognition (speaker, advisor, or "
            "committee member in their field)."
        ),
    ),
    ProfileLevel(
        name="top_innovator",
        rank=0,
        instruction=(
            "Generate the resume of a TOP-TIER high-visibility innovator: an ideal fit for "
            "the posting who also stands out through notable achievements beyond day-to-day "
            "work — patents filed, widely-used open-source tools authored, awards received, "
            "conference talks given, or significant media coverage of their technical work. "
            "Their profile reads as both highly competent and distinctively accomplished."
        ),
    ),
)

PROFILE_LEVELS_TOP_BY_NAME: dict[str, ProfileLevel] = {lvl.name: lvl for lvl in PROFILE_LEVELS_TOP}

# Tous les sets disponibles, indexés par nom.
PROFILE_SETS: dict[str, tuple[ProfileLevel, ...]] = {
    "default": PROFILE_LEVELS,
    "top": PROFILE_LEVELS_TOP,
}


# ---------------------------------------------------------------------------
# CV « à optimiser » — fit théorique fort, mais vocabulaire non aligné
# ---------------------------------------------------------------------------

# Rang de vérité-terrain du CV « à optimiser » : sur le fond, c'est un excellent
# candidat (rank 0). Le « défaut » est volontaire et porte uniquement sur le
# vocabulaire, pas sur l'adéquation réelle au poste.
OPTIMIZE_RANK = 0

# Consigne par défaut du CV « à optimiser » : candidat excellent sur le fond, mais
# CV rédigé sans reprendre les termes de l'annonce, de sorte que les méthodes
# mots-clés / embeddings le sous-évaluent à tort. C'est le CV « cobaye » destiné à
# être optimisé ensuite.
DEFAULT_OPTIMIZE_INSTRUCTION = (
    "Generate the resume of a candidate who, IN SUBSTANCE, is an EXCELLENT match for "
    "the job posting: they genuinely have every required skill, experience and "
    "qualification. HOWEVER, the resume is deliberately worded so that it does NOT "
    "use the vocabulary of the job posting: avoid its exact keywords, job titles, "
    "tool and technology names, and standard phrasing. Instead, describe the very "
    "same real competencies using synonyms, paraphrases, generic wording and "
    "alternative job titles. The candidate is truly a top fit, but a keyword- or "
    "embedding-based screening would wrongly underrate this resume because the terms "
    "do not match the posting. Keep the profile credible and realistic."
)


def _optimize_level(instruction: Optional[str] = None) -> ProfileLevel:
    """Construit le ``ProfileLevel`` du CV « à optimiser ».

    Args:
        instruction: Consigne personnalisée. À défaut, ``DEFAULT_OPTIMIZE_INSTRUCTION``.
    """
    return ProfileLevel(
        name="to_optimize",
        rank=OPTIMIZE_RANK,
        instruction=instruction or DEFAULT_OPTIMIZE_INSTRUCTION,
    )


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class SyntheticCVGenerator:
    """Génère des CVs synthétiques PDF face à une annonce, via l'API Mistral ou Claude.

    La proximité de chaque CV avec l'annonce est pilotée par un ``ProfileLevel``
    discret. La méthode principale :func:`generate_cvs` produit ``n`` CVs en cyclant
    sur les niveaux disponibles (distribution équilibrée), les écrit en PDF et
    enregistre un ``manifest.json`` décrivant la vérité-terrain.
    """

    def __init__(
        self,
        model: str = CV_GENERATOR_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        """
        Args:
            model:       Identifiant du modèle. Le fournisseur (Mistral ou Claude)
                         est déduit de son préfixe. Défaut : ``CV_GENERATOR_MODEL``
                         (défini dans ``config.py``).
            api_key:     Clé API du fournisseur. À défaut, la variable
                         d'environnement adéquate (``MISTRAL_API_KEY`` ou
                         ``ANTHROPIC_API_KEY``, chargée depuis ``.env``) est utilisée.
            max_tokens:  Nombre maximum de tokens par CV généré.
            temperature: Température d'échantillonnage du modèle.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key
        self._llm: Optional[LLMClient] = None

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def import_model(self) -> None:
        """Initialise le client LLM (Mistral ou Claude). À appeler avant toute génération.

        Le fournisseur est déduit du préfixe du modèle. La clé API est lue depuis
        l'argument ``api_key`` ou, à défaut, depuis la variable d'environnement du
        fournisseur, chargée d'un fichier ``.env``.
        """
        self._llm = LLMClient(self.model, api_key=self._api_key)
        self._llm.import_model()

    def generate_cv(self, announcement: str, level: ProfileLevel) -> str:
        """Génère le texte d'un CV pour un niveau de profil donné.

        Args:
            announcement: Texte complet de l'annonce.
            level:        Niveau de profil ciblé (cf. ``PROFILE_LEVELS``).

        Returns:
            Le texte du CV généré.
        """
        if self._llm is None:
            raise RuntimeError("Appelez import_model() avant de générer des CVs.")

        prompt = self._build_prompt(announcement, level)
        return self._llm.complete(prompt, self.max_tokens, self.temperature)

    def generate_cvs(
        self,
        announcement: str,
        n: int,
        levels: Optional[list[str]] = None,
        profile_set: str = "default",
        run_name: Optional[str] = None,
        output_dir: Path = GENERATED_DATA_DIR,
        include_optimize: bool = True,
        optimize_instruction: Optional[str] = None,
        save: bool = True,
    ):
        """Génère ``n`` CVs ; si ``save``, les écrit en PDF avec un manifest de vérité-terrain.

        Les ``n`` CVs se voient assigner un niveau de profil en **cyclant** sur la liste
        des niveaux retenus (distribution équilibrée). Si ``include_optimize`` est vrai,
        **un** CV « à optimiser » supplémentaire est généré : un candidat excellent sur le
        fond mais dont le vocabulaire ne reprend pas les termes de l'annonce (cas de test
        délibéré pour les méthodes mots-clés / embeddings).

        Args:
            announcement:         Texte complet de l'annonce.
            n:                    Nombre de CVs « niveau de profil » à générer.
            levels:               Sous-ensemble de noms de niveaux à utiliser dans le set
                                  choisi. Défaut : tous les niveaux du set.
            profile_set:          Set de niveaux à utiliser : ``"default"`` (hétérogène :
                                  perfect/strong/partial/unrelated) ou ``"top"`` (5 variantes
                                  top-niveau). Défaut : ``"default"``.
            run_name:             Nom de dossier forcé. Défaut : ``synthetic_cvs_<horodatage>``.
            output_dir:           Dossier parent des sorties. Défaut : ``GENERATED_DATA_DIR``.
            include_optimize:     Si vrai, génère en plus un CV « à optimiser ».
            optimize_instruction: Consigne personnalisée du CV « à optimiser ».
                                  Défaut : ``DEFAULT_OPTIMIZE_INSTRUCTION``.
            save:                 Si vrai, écrit les PDF + ``manifest.json`` et retourne le
                                  ``Path`` du dossier. Sinon, n'écrit rien et retourne la liste
                                  des CVs générés en mémoire (dicts ``{"file", "level",
                                  "level_rank", "text"[, "optimize"]}``).

        Returns:
            Le ``Path`` du dossier de sortie (si ``save``), sinon la liste des CVs générés.
        """
        if self._llm is None:
            raise RuntimeError("Appelez import_model() avant de générer des CVs.")
        if n < 1:
            raise ValueError("Il faut générer au moins 1 CV.")
        if profile_set not in PROFILE_SETS:
            raise ValueError(
                f"Set de profils inconnu : {profile_set!r}. "
                f"Valeurs possibles : {list(PROFILE_SETS)}."
            )

        available = {lvl.name: lvl for lvl in PROFILE_SETS[profile_set]}
        selected = self._resolve_levels(levels, available)
        run_dir = self._make_run_dir(output_dir, run_name) if save else None

        generated: list[dict] = []
        for i in range(n):
            level = selected[i % len(selected)]
            logger.info("Génération du CV %d/%d (niveau : %s)", i + 1, n, level.name)
            text = self.generate_cv(announcement, level)

            file_name = f"cv_{i + 1:03d}_{level.name}.pdf"
            generated.append(
                {"file": file_name, "level": level.name, "level_rank": level.rank, "text": text}
            )
            if save:
                write_text_pdf(text, run_dir / file_name)

        if include_optimize:
            opt_level = _optimize_level(optimize_instruction)
            logger.info("Génération du CV « à optimiser » (niveau : %s)", opt_level.name)
            text = self.generate_cv(announcement, opt_level)

            file_name = f"cv_{0:03d}_{opt_level.name}.pdf"
            generated.append(
                {
                    "file": file_name,
                    "level": opt_level.name,
                    "level_rank": opt_level.rank,
                    "text": text,
                    "optimize": True,
                }
            )
            if save:
                write_text_pdf(text, run_dir / file_name)

        if not save:
            logger.info("%d CVs générés (non sauvegardés)", len(generated))
            return generated

        manifest = {
            "generator": "synthetic_cv_generator",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "params": {
                "model": self.model,
                "count": n,
                "profile_set": profile_set,
                "levels": [lvl.name for lvl in selected],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "include_optimize": include_optimize,
                "optimize_instruction": (
                    _optimize_level(optimize_instruction).instruction if include_optimize else None
                ),
            },
            # Le manifest ne référence que les métadonnées des fichiers (pas le texte brut).
            "cvs": [{k: v for k, v in cv.items() if k != "text"} for cv in generated],
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info("%d CVs générés dans %s", len(generated), run_dir)
        return run_dir

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_levels(
        levels: Optional[list[str]], available: dict[str, ProfileLevel]
    ) -> list[ProfileLevel]:
        """Valide et résout une liste de noms de niveaux en objets ``ProfileLevel``."""
        if levels is None:
            return list(available.values())
        resolved = []
        for name in levels:
            if name not in available:
                raise ValueError(
                    f"Niveau inconnu : {name!r}. Valeurs possibles : {list(available)}."
                )
            resolved.append(available[name])
        if not resolved:
            raise ValueError("La liste de niveaux ne peut pas être vide.")
        return resolved

    @staticmethod
    def _make_run_dir(output_dir: Path, run_name: Optional[str]) -> Path:
        """Crée le dossier de sortie horodaté (jamais écrasé).

        Même logique de nommage que ``results_io.save_results`` : un horodatage
        ``%Y%m%d-%H%M%S`` avec suffixe incrémental anti-collision.
        """
        if run_name is not None:
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_dir / f"synthetic_cvs_{stamp}"
        counter = 1
        while run_dir.exists():
            run_dir = output_dir / f"synthetic_cvs_{stamp}_{counter}"
            counter += 1
        run_dir.mkdir(parents=True)
        return run_dir

    def _build_prompt(self, announcement: str, level: ProfileLevel) -> str:
        return f"""You are an expert resume writer. Based on the job posting below, \
write ONE realistic and credible fictional candidate resume, in English.

## Job posting
{announcement}

## Expected level of match
{level.instruction}

## Writing guidelines
- Invent fictional contact details (name, email, phone, city).
- Structure the resume with clear sections: Summary / Profile, Professional \
experience (with dates and descriptions), Education, Skills, Languages.
- Stay consistent with the level of match requested above.
- Do not add any comment or explanation outside the resume: return only the resume text."""

    # La conversion texte → PDF (police Helvetica, latin-1) vit désormais dans
    # ``ats_system.data.pdf_writer.write_text_pdf`` (mutualisée avec l'agent).
