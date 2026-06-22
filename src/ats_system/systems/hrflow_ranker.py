"""Système ATS via l'API HrFlow (indexation JSON + scoring de profils face à une offre).

Flow (SDK hrflow v4) :
  1. L'offre est indexée dans un board HrFlow via ``job.storing.add_json`` (texte dans
     ``sections``), avec un hash MD5 du texte comme référence pour l'upsert idempotent.
  2. Chaque CV PDF est lu localement (``import_pdf``) puis indexé via
     ``profile.storing.add_json`` (le texte brut dans ``text`` et ``info.summary``).
     L'indexation est synchrone — aucune pause nécessaire.
  3. L'API ``profile.scoring.list`` renvoie le score de correspondance de chaque
     profil (0–1, converti en 0–100 pour cohérence avec les autres systèmes).

Prérequis — variables d'environnement dans ``.env`` :
  HRFLOW_API_KEY, HRFLOW_API_USER, HRFLOW_SOURCE_KEY, HRFLOW_BOARD_KEY
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ats_system.config import CV_DIR, DEFAULT_ANNOUNCEMENT, DEFAULT_CV_CATEGORY
from ats_system.data import load_announcement
from ats_system.data.pdf_loader import import_pdf
from ats_system.results_io import build_ranking, save_results, timestamped_run_dir

METHOD = "hrflow_ranking"

_CONSENT = {
    "owner": {
        "parsing": False,
        "revealing": False,
        "embedding": True,
        "searching": True,
        "scoring": True,
        "reasoning": False,
    },
    "controller": {
        "parsing": False,
        "revealing": False,
        "embedding": True,
        "searching": True,
        "scoring": True,
        "reasoning": False,
    },
}


class HrflowCVRanker:
    """Score CV/offre via l'API HrFlow (indexation JSON + scoring cloud)."""

    def __init__(self) -> None:
        self._client = None
        self._source_key: str = ""
        self._board_key: str = ""

    def import_model(self) -> None:
        """Charge les credentials HrFlow et initialise le client. À appeler avant usage."""
        from hrflow import Hrflow  # import tardif : package optionnel

        load_dotenv()
        api_key = os.environ["HRFLOW_API_KEY"]
        api_user = os.environ["HRFLOW_API_USER"]
        self._source_key = os.environ["HRFLOW_SOURCE_KEY"]
        self._board_key = os.environ["HRFLOW_BOARD_KEY"]
        self._client = Hrflow(api_secret=api_key, api_user=api_user)

    def _store_job(self, text: str) -> str:
        """Indexe l'offre dans le board HrFlow → retourne le job_key.

        Utilise un hash MD5 du texte comme référence (upsert idempotent).
        Si la référence existe déjà, récupère le job existant via ``list``.
        """
        reference = "ats-job-" + hashlib.md5(text.encode()).hexdigest()[:12]
        job_json = {
            "reference": reference,
            "name": "ATS Scoring Job",
            "location": {"text": ""},
            "sections": [{"name": "description", "title": "Description", "description": text}],
        }
        response = self._client.job.storing.add_json(self._board_key, job_json)
        code = response.get("code")

        if code == 400 and "Already used" in (response.get("message") or ""):
            list_resp = self._client.job.storing.list([self._board_key], reference=reference, page=1, limit=1)
            if list_resp.get("code") != 200:
                raise RuntimeError(f"HrFlow job list — code {list_resp.get('code')} : {list_resp.get('message')}")
            jobs = list_resp.get("data", [])
            if not jobs:
                raise RuntimeError("HrFlow : offre déjà référencée mais introuvable via list")
            return jobs[0]["key"]

        if code not in (200, 201):
            raise RuntimeError(f"HrFlow job storing — code {code} : {response.get('message')}")
        return response["data"]["key"]

    def _index_profile(self, cv_path: Path) -> None:
        """Lit le PDF localement et indexe le profil via ``profile.storing.add_json``.

        La référence est le nom du fichier PDF, ce qui permet l'idempotence entre
        exécutions. Si la référence existe déjà, l'appel est ignoré silencieusement.
        """
        text = import_pdf(cv_path)["content"]
        profile_json = {
            "reference": cv_path.name,
            "text": text,
            "consent_algorithmic": _CONSENT,
            "info": {
                "first_name": cv_path.stem,
                "last_name": "",
                "full_name": cv_path.stem,
                "email": f"{cv_path.stem}@ats-system.local",
                "phone": "",
                "location": {"text": ""},
                "summary": text[:2000],
            },
            "experiences": [],
            "educations": [],
            "skills": [],
        }
        response = self._client.profile.storing.add_json(self._source_key, profile_json)
        code = response.get("code")
        if code == 400 and "Already used" in (response.get("message") or ""):
            return  # profil déjà indexé, rien à faire
        if code not in (200, 201):
            raise RuntimeError(f"HrFlow profile storing {cv_path.name!r} — code {code} : {response.get('message')}")

    def run(
        self,
        *,
        limit: Optional[int] = None,
        announcement: Path = DEFAULT_ANNOUNCEMENT,
        category: str = DEFAULT_CV_CATEGORY,
        save: bool = True,
    ) -> list[tuple[str, float]]:
        """Pipeline complet : initialisation client, chargement, scoring et sauvegarde.

        Args:
            limit:        Nombre maximum de CVs à traiter (``None``/``0`` = tous).
            announcement: PDF de l'annonce (défaut : annonce par défaut du projet).
            category:     Catégorie de CVs (sous-dossier de ``CV_DIR``).
            save:         Si vrai, écrit le classement sous ``results/<METHOD>/<horodatage>/``.

        Returns:
            Le classement, paires ``(cv_id, score)`` triées décroissant.
        """
        self.import_model()

        offre = load_announcement(announcement)
        cv_files = sorted((CV_DIR / category).glob("*.pdf"))
        if limit is not None and limit > 0:
            cv_files = cv_files[:limit]

        # 1. Indexer l'offre
        job_key = self._store_job(offre["content"])
        print(f"  Offre indexée (job_key={job_key!r})")

        # 2. Indexer les CVs
        for cv_path in cv_files:
            self._index_profile(cv_path)
            print(f"  CV indexé : {cv_path.name}")

        # 3. Scoring : tous les profils de la source face à l'offre
        cv_ids = {p.name for p in cv_files}
        response = self._client.profile.scoring.list(
            source_keys=[self._source_key],
            board_key=self._board_key,
            job_key=job_key,
            limit=len(cv_files),
        )
        if response.get("code") != 200:
            raise RuntimeError(f"HrFlow scoring — code {response.get('code')} : {response.get('message')}")

        # 4. Mapper référence → (cv_id, score), normaliser 0–1 → 0–100
        scored = []
        for item in response["data"]["profiles"]:
            ref = item["profile"].get("reference", "")
            if ref not in cv_ids:
                continue
            raw_score = float(item.get("score", 0.0))
            scored.append((ref, round(raw_score * 100, 1)))

        scored.sort(key=lambda x: x[1], reverse=True)

        for cv_id, score in scored:
            print(f"{score:5.1f}%  {cv_id}")

        if save:
            params = {
                "announcement": Path(announcement).name,
                "category": category,
                "limit": limit if limit is not None else 0,
                "num_cvs": len(cv_files),
            }
            out = save_results(
                METHOD,
                build_ranking(scored),
                params,
                results_dir=timestamped_run_dir(METHOD),
                stamp_filename=False,
            )
            print(f"\nRésultats sauvegardés dans : {out}")
        return scored
