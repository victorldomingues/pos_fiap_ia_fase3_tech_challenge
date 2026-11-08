"""Inicialização transacional da base sintética persistente do notebook."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from src.database.sqlite import ClinicDatabase, Principal, Role


@dataclass(frozen=True)
class DemoProfiles:
    database: ClinicDatabase
    patient: Principal
    doctor: Principal
    reviewer: Principal
    additional_patients: tuple[Principal, ...] = ()


def demo_key_path(path: Path) -> Path:
    """A chave fica ao lado do banco, mas fora do SQLite e do código versionado."""
    return path.with_name(path.name + ".demo-key")


def _write_key(path: Path, secret: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(secret)
        file.flush()
        os.fsync(file.fileno())


def _available_demo_slot(connection, patient_id: str, doctor_id: str) -> str:
    candidate = datetime.fromisoformat("2030-01-17T10:00-03:00")
    while connection.execute(
        "SELECT 1 FROM appointments WHERE scheduled_at = ? AND (patient_id = ? OR doctor_id = ?)",
        (candidate.isoformat(timespec="minutes"), patient_id, doctor_id),
    ).fetchone():
        candidate += timedelta(days=1)
    return candidate.isoformat(timespec="minutes")


def setup_auto_persistent_demo(path: Path, *, previous_secret: str | None = None) -> DemoProfiles:
    """Gera uma chave local uma única vez ou recupera um banco criado com chave anterior."""
    if not path.is_absolute():
        raise ValueError("O caminho do SQLite deve ser absoluto.")
    key_path = demo_key_path(path)
    if key_path.exists():
        if not path.exists():
            raise RuntimeError("Chave existente sem banco SQLite. Restaure o banco antes de continuar.")
        return setup_persistent_demo(path, key_path.read_text(encoding="utf-8"))
    if path.exists():
        if previous_secret is None:
            raise RuntimeError(
                "Chave automática ausente para banco existente. Informe a chave anterior para recuperá-la; "
                "nenhum dado será recriado."
            )
        profiles = setup_persistent_demo(path, previous_secret)
        _write_key(key_path, previous_secret)
        return profiles
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_urlsafe(32)
    _write_key(key_path, secret)
    return setup_persistent_demo(path, secret)


def setup_persistent_demo(path: Path, secret: str) -> DemoProfiles:
    """Cria os exemplos uma vez ou autentica os existentes, sem alterar suas credenciais."""
    if len(secret) < 32 or secret != secret.strip():
        raise ValueError("A chave da demonstração deve ter pelo menos 32 caracteres e não ter espaços nas extremidades.")
    if not path.is_absolute():
        raise ValueError("O caminho do SQLite deve ser absoluto.")
    database = ClinicDatabase(path)
    identities = (
        ("SYN-P-DEMO", Role.PATIENT),
        ("SYN-D-DEMO", Role.DOCTOR),
        ("SYN-D-REVISOR", Role.DOCTOR),
        ("SYN-P-DEMO-02", Role.PATIENT),
    )
    principals = tuple(
        Principal(identifier, role, hmac.new(
            secret.encode("utf-8"), f"iadt-demo-v1:{identifier}".encode("utf-8"), hashlib.sha256
        ).hexdigest())
        for identifier, role in identities
    )
    patient, doctor, reviewer, additional_patient = principals
    with database._transaction() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS demo_metadata (version INTEGER NOT NULL)")
        row = connection.execute("SELECT version FROM demo_metadata").fetchone()
        if row is None:
            existing = connection.execute(
                "SELECT 1 FROM patients WHERE patient_id IN (?, ?) UNION ALL "
                "SELECT 1 FROM doctors WHERE doctor_id IN (?, ?) LIMIT 1",
                (patient.user_id, additional_patient.user_id, doctor.user_id, reviewer.user_id),
            ).fetchone()
            if existing:
                raise RuntimeError("Identificadores da demonstração já existem sem metadados; use outro banco ou investigue.")
            now = database._now()
            for principal, name, cpf, birth_date, phone, email in (
                (patient, "Paciente Sintético", "111.444.777-35", "1980-01-02",
                 "(11) 91234-5678", "paciente@example.test"),
                (additional_patient, "Paciente Sintético Dois", "222.555.888-46", "1990-05-10",
                 "(21) 99876-5432", "paciente2@example.test"),
            ):
                connection.execute(
                    "INSERT INTO patients (patient_id, full_name, cpf, birth_date, phone, email, access_token_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (principal.user_id, name, cpf, birth_date, phone, email,
                     database._token_hash(principal.access_token), now),
                )
            for principal, name in ((doctor, "Doutora Demonstração"), (reviewer, "Médico Revisor Sintético")):
                connection.execute(
                    "INSERT INTO doctors VALUES (?, ?, ?, ?, ?)",
                    (principal.user_id, name, "consulta_geral", database._token_hash(principal.access_token), now),
                )
            for identifier, professional, scheduled in (
                ("SYN-A-DEMO", doctor, "2030-01-15T10:00-03:00"),
                ("SYN-A-REVISOR", reviewer, "2030-01-16T10:00-03:00"),
            ):
                connection.execute(
                    "INSERT INTO appointments VALUES (?, ?, ?, ?, 'consulta_geral', 'scheduled', ?)",
                    (identifier, patient.user_id, professional.user_id, scheduled, now),
                )
            connection.execute(
                "INSERT INTO appointments VALUES (?, ?, ?, ?, 'retorno_ambulatorial', 'scheduled', ?)",
                ("SYN-A-DEMO-02", additional_patient.user_id, doctor.user_id, "2030-01-17T10:00-03:00", now),
            )
            connection.execute(
                "INSERT INTO clinical_notes VALUES (?, ?, ?, ?, ?)",
                ("SYN-N-DEMO", patient.user_id, doctor.user_id, "Anotação sintética de retorno agendado.", now),
            )
            connection.execute(
                "INSERT INTO prescriptions VALUES (?, ?, ?, ?, 'pending_review', NULL, 0, ?, NULL)",
                ("SYN-R-DEMO", patient.user_id, doctor.user_id,
                 "Proposta fictícia fornecida pelo profissional para revisão.", now),
            )
            connection.execute("INSERT INTO demo_metadata VALUES (2)")
            database._record(connection, "bootstrap", "system", "demo_initialized", None)
        elif row["version"] == 1:
            now = database._now()
            existing_patient = connection.execute(
                "SELECT 1 FROM patients WHERE patient_id = ?", (additional_patient.user_id,)
            ).fetchone()
            if existing_patient is None:
                connection.execute(
                    "INSERT INTO patients (patient_id, full_name, cpf, birth_date, phone, email, access_token_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (additional_patient.user_id, "Paciente Sintético Dois", "222.555.888-46", "1990-05-10",
                     "(21) 99876-5432", "paciente2@example.test",
                     database._token_hash(additional_patient.access_token), now),
                )
            else:
                database._authorized(connection, additional_patient)
            existing_appointment = connection.execute(
                "SELECT patient_id, doctor_id FROM appointments WHERE appointment_id = ?",
                ("SYN-A-DEMO-02",),
            ).fetchone()
            if existing_appointment is None:
                scheduled_at = _available_demo_slot(
                    connection,
                    additional_patient.user_id,
                    doctor.user_id,
                )
                connection.execute(
                    "INSERT INTO appointments VALUES (?, ?, ?, ?, 'retorno_ambulatorial', 'scheduled', ?)",
                    ("SYN-A-DEMO-02", additional_patient.user_id, doctor.user_id, scheduled_at, now),
                )
            elif (
                existing_appointment["patient_id"] != additional_patient.user_id
                or existing_appointment["doctor_id"] != doctor.user_id
            ):
                raise RuntimeError("O identificador do agendamento sintético adicional já está em uso.")
            connection.execute("UPDATE demo_metadata SET version = 2")
            database._record(connection, "bootstrap", "system", "demo_data_migrated", "additional_patient")
        elif row["version"] != 2:
            raise RuntimeError("Versão dos dados de demonstração incompatível.")
        for principal in principals:
            database._authorized(connection, principal)
    return DemoProfiles(database, patient, doctor, reviewer, (additional_patient,))
