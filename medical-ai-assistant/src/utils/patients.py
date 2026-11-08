"""Pacientes SINTÉTICOS: carga validada e pseudonimização (minimização de dados)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.models.schemas import Patient
from src.security.anonymizer import anonymize


def load_patients(path: Path) -> list[Patient]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Patient.model_validate(r) for r in rows]


def pseudonymize(patient: Patient, salt: str) -> dict[str, Any]:
    """Remove identificadores diretos; mantém apenas ano de nascimento e notas anonimizadas.

    O salt deve ser mantido fora do repositório. Mesmo pseudonimizado, o dado continua pessoal (LGPD).
    """
    token = hashlib.sha256(f"{salt}:{patient.patient_id}".encode()).hexdigest()[:16]
    birth_year = patient.birth_date.split("/")[-1] if "/" in patient.birth_date else None
    return {"pseudo_id": token, "birth_year": birth_year, "notes": anonymize(patient.notes)}
