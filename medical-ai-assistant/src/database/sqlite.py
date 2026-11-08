"""SQLite local para os fluxos administrativos da demonstração (somente dados sintéticos)."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from src.security.anonymizer import contains_pii


class Role(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"

APPOINTMENT_TYPES = frozenset({"consulta_geral", "retorno_ambulatorial"})


class AuthorizationError(PermissionError):
    """A identidade autenticada não tem permissão para a operação solicitada."""


class ConflictError(ValueError):
    """A operação viola uma restrição de integridade do banco."""


@dataclass(frozen=True)
class Principal:
    """Identidade autenticada; o papel é derivado do cadastro, nunca confiado ao chamador."""

    user_id: str
    role: Role
    access_token: str


def validate_patient_field(field: str, value: str) -> str:
    """Valida campos do formulário sintético sem incluir valores nas mensagens de erro."""
    value = value.strip()
    if field == "administrative_note":
        if len(value) > 500 or contains_pii(value, labels=None):
            raise ValueError("Observação inválida: use até 500 caracteres sem dados pessoais.")
        return value
    labels = {"full_name": "Nome", "cpf": "CPF", "birth_date": "Nascimento",
              "phone": "Telefone", "email": "E-mail"}
    if field not in labels:
        raise ValueError("Campo desconhecido.")
    if not value or len(value) > 150:
        raise ValueError(f"{labels[field]} é obrigatório e deve ter até 150 caracteres.")
    if field == "cpf" and not re.fullmatch(r"(?:\d{11}|\d{3}\.\d{3}\.\d{3}-\d{2})", value):
        raise ValueError("CPF inválido. Use 11 dígitos ou o formato 000.000.000-00.")
    if field == "phone" and not re.fullmatch(r"(?:\d{10,11}|\(\d{2}\)\s?\d{4,5}-\d{4})", value):
        raise ValueError("Telefone inválido. Informe DDD e número.")
    if field == "email" and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
        raise ValueError("E-mail inválido.")
    if field == "birth_date":
        try:
            parsed = datetime.strptime(value, "%d/%m/%Y").date()
        except ValueError as exc:
            raise ValueError("Nascimento inválido. Use DD/MM/AAAA.") from exc
        if parsed > date.today():
            raise ValueError("Nascimento não pode estar no futuro.")
        return parsed.isoformat()
    return value


class ClinicDatabase:
    """Repositório SQLite com autenticação por token e autorização verificada em cada operação.

    O token em texto claro é entregue somente no cadastro/provisionamento; apenas seu SHA-256
    é persistido. O banco contém dados sintéticos e não substitui autenticação corporativa.
    """

    def __init__(self, path: Path, bootstrap_token: str | None = None) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._bootstrap_token = bootstrap_token
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Cria o esquema versionado sem apagar ou recriar um banco existente."""
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    version INTEGER NOT NULL
                );
                INSERT INTO schema_metadata(version)
                    SELECT 3 WHERE NOT EXISTS (SELECT 1 FROM schema_metadata);

                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    cpf TEXT NOT NULL,
                    birth_date TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    email TEXT NOT NULL,
                    administrative_note TEXT NOT NULL DEFAULT '',
                    access_token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS doctors (
                    doctor_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    specialty TEXT NOT NULL,
                    access_token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS appointments (
                    appointment_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id),
                    scheduled_at TEXT NOT NULL,
                    appointment_type TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('scheduled', 'completed', 'cancelled')),
                    created_at TEXT NOT NULL,
                    UNIQUE(patient_id, scheduled_at),
                    UNIQUE(doctor_id, scheduled_at)
                );
                CREATE TABLE IF NOT EXISTS prescriptions (
                    prescription_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    author_doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id),
                    prescription_text TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending_review', 'approved', 'rejected')),
                    reviewer_doctor_id TEXT REFERENCES doctors(doctor_id),
                    review_note_provided INTEGER NOT NULL DEFAULT 0 CHECK (review_note_provided IN (0, 1)),
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS clinical_notes (
                    note_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    author_doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entity_id TEXT,
                    outcome TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_appointments_patient
                    ON appointments(patient_id, scheduled_at);
                CREATE INDEX IF NOT EXISTS idx_appointments_doctor
                    ON appointments(doctor_id, scheduled_at);
                CREATE INDEX IF NOT EXISTS idx_prescriptions_patient
                    ON prescriptions(patient_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_clinical_notes_patient
                    ON clinical_notes(patient_id, created_at);
                """
            )
            version = connection.execute("SELECT version FROM schema_metadata").fetchone()["version"]
            if version not in {1, 2, 3}:
                raise RuntimeError(f"Versão de esquema SQLite incompatível: {version}.")
            if version in {1, 2}:
                connection.execute("ALTER TABLE patients ADD COLUMN administrative_note TEXT NOT NULL DEFAULT ''")
                connection.execute("UPDATE schema_metadata SET version = 3")
            connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _required(value: str, label: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{label} é obrigatório.")
        return normalized

    @staticmethod
    def _mask_cpf(value: str) -> str:
        digits = "".join(character for character in value if character.isdigit())
        return f"***.***.***-{digits[-2:]}" if len(digits) >= 2 else "[CPF]"

    @staticmethod
    def _mask_phone(value: str) -> str:
        digits = "".join(character for character in value if character.isdigit())
        return f"(**) *****-{digits[-4:]}" if len(digits) >= 4 else "[TELEFONE]"

    @staticmethod
    def _mask_email(value: str) -> str:
        local, separator, domain = value.partition("@")
        return f"{local[:1]}***@{domain}" if separator and local and domain else "[EMAIL]"

    @staticmethod
    def _mask_name(value: str) -> str:
        return " ".join(f"{part[:1]}***" for part in value.split())

    def _record(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        role: str,
        action: str,
        entity_id: str | None,
        outcome: str = "success",
    ) -> None:
        connection.execute(
            """INSERT INTO audit_logs(timestamp, user_id, profile, action, entity_id, outcome)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (self._now(), user_id, role, action, entity_id, outcome),
        )

    def _authenticate(self, connection: sqlite3.Connection, principal: Principal) -> sqlite3.Row:
        if not principal.access_token.strip():
            raise AuthorizationError("Credencial ausente.")
        token_hash = self._token_hash(principal.access_token)
        identity_sources = (
            ("patients", "patient_id", Role.PATIENT),
            ("doctors", "doctor_id", Role.DOCTOR),
        )
        for table_name, column, expected_role in identity_sources:
            row = connection.execute(
                f"SELECT {column}, access_token_hash FROM {table_name} WHERE access_token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row:
                actual_id = row[column]
                if (
                    expected_role == principal.role
                    and actual_id == principal.user_id
                    and hmac.compare_digest(row["access_token_hash"], token_hash)
                ):
                    return row
        raise AuthorizationError("Credencial inválida ou perfil não corresponde à identidade cadastrada.")

    def _authorized(self, connection: sqlite3.Connection, principal: Principal) -> sqlite3.Row:
        return self._authenticate(connection, principal)

    @staticmethod
    def _patient_in_doctor_scope(
        connection: sqlite3.Connection, doctor_id: str, patient_id: str
    ) -> bool:
        row = connection.execute(
            """SELECT 1 FROM appointments
               WHERE doctor_id = ? AND patient_id = ? AND status IN ('scheduled', 'completed')
               LIMIT 1""",
            (doctor_id, patient_id),
        ).fetchone()
        return row is not None

    def register_patient(
        self,
        full_name: str,
        cpf: str,
        birth_date: str,
        phone: str,
        email: str,
        *,
        synthetic_demo_confirmed: bool,
    ) -> Principal:
        """Cadastra um paciente fictício e retorna uma credencial exibida somente uma vez."""
        if not synthetic_demo_confirmed:
            raise ValueError("Este protótipo aceita somente dados sintéticos; confirme antes do cadastro.")
        values = (
            self._required(full_name, "Nome"),
            self._required(cpf, "CPF"),
            self._required(birth_date, "Data de nascimento"),
            self._required(phone, "Telefone"),
            self._required(email, "E-mail"),
        )
        if "@" not in values[4]:
            raise ValueError("E-mail inválido.")
        patient_id = f"SYN-P-{uuid.uuid4().hex[:12].upper()}"
        token = secrets.token_urlsafe(32)
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO patients
                   (patient_id, full_name, cpf, birth_date, phone, email, access_token_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (patient_id, *values, self._token_hash(token), self._now()),
            )
            self._record(connection, patient_id, Role.PATIENT.value, "patient_registered", patient_id)
        return Principal(patient_id, Role.PATIENT, token)

    def register_patient_by_doctor(
        self, principal: Principal, full_name: str, cpf: str, birth_date: str,
        phone: str, email: str, administrative_note: str = "", *, synthetic_demo_confirmed: bool,
    ) -> Principal:
        """Cadastra paciente sintético sem criar vínculo de acesso para o profissional."""
        if not synthetic_demo_confirmed:
            raise ValueError("Confirme o uso exclusivo de dados sintéticos.")
        values = tuple(validate_patient_field(field, value) for field, value in (
            ("full_name", full_name), ("cpf", cpf), ("birth_date", birth_date),
            ("phone", phone), ("email", email), ("administrative_note", administrative_note),
        ))
        patient_id = f"SYN-P-{uuid.uuid4().hex[:12].upper()}"
        token = secrets.token_urlsafe(32)
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role != Role.DOCTOR:
                raise AuthorizationError("Somente profissional autenticado pode cadastrar outro paciente.")
            connection.execute(
                """INSERT INTO patients (patient_id, full_name, cpf, birth_date, phone, email,
                   administrative_note, access_token_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (patient_id, *values, self._token_hash(token), self._now()),
            )
            self._record(connection, principal.user_id, principal.role.value, "patient_registered_by_doctor", patient_id)
        return Principal(patient_id, Role.PATIENT, token)

    def get_session_identity(self, principal: Principal) -> dict[str, str]:
        """Identifica a própria sessão autenticada, sem disponibilizar dados de terceiros."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            table, column = ("patients", "patient_id") if principal.role == Role.PATIENT else ("doctors", "doctor_id")
            row = connection.execute(f"SELECT full_name FROM {table} WHERE {column} = ?", (principal.user_id,)).fetchone()
            return {"user_id": principal.user_id, "name": row["full_name"]}

    def provision_doctor(
        self,
        doctor_id: str,
        full_name: str,
        specialty: str,
        bootstrap_token: str,
        *,
        synthetic_demo_confirmed: bool,
    ) -> Principal:
        """Provisiona profissional fictício; exige segredo administrativo configurado fora do código."""
        if not self._bootstrap_token or not hmac.compare_digest(bootstrap_token, self._bootstrap_token):
            raise AuthorizationError("Token administrativo inválido ou não configurado.")
        if not synthetic_demo_confirmed:
            raise ValueError("Este protótipo aceita somente perfis profissionais sintéticos.")
        doctor_id = self._required(doctor_id, "Identificador do profissional")
        if not re.fullmatch(r"SYN-D-[A-Z0-9-]{1,32}", doctor_id):
            raise ValueError("O identificador do profissional deve usar o formato sintético SYN-D-*.") 
        name = self._required(full_name, "Nome")
        specialty = self._required(specialty, "Especialidade")
        token = secrets.token_urlsafe(32)
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO doctors(doctor_id, full_name, specialty, access_token_hash, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (doctor_id, name, specialty, self._token_hash(token), self._now()),
            )
            self._record(connection, "bootstrap", "system", "doctor_provisioned", doctor_id)
        return Principal(doctor_id, Role.DOCTOR, token)

    def authenticate(self, access_token: str) -> Principal:
        """Resolve o perfil a partir da credencial persistida, sem confiar em perfil informado."""
        token_hash = self._token_hash(access_token.strip())
        with self._connection() as connection:
            for table, id_column, role in (
                ("patients", "patient_id", Role.PATIENT),
                ("doctors", "doctor_id", Role.DOCTOR),
            ):
                row = connection.execute(
                    f"SELECT {id_column} AS user_id, access_token_hash FROM {table} WHERE access_token_hash = ?",
                    (token_hash,),
                ).fetchone()
                if row and hmac.compare_digest(row["access_token_hash"], token_hash):
                    return Principal(row["user_id"], role, access_token.strip())
        raise AuthorizationError("Credencial inválida.")

    def create_appointment(
        self,
        principal: Principal,
        doctor_id: str,
        scheduled_at: str,
        appointment_type: str,
    ) -> dict[str, str]:
        """Paciente agenda apenas para si; conflitos de horário são rejeitados pelo SQLite."""
        try:
            scheduled = datetime.fromisoformat(scheduled_at)
        except ValueError as exc:
            raise ValueError("Data/hora deve estar em formato ISO-8601 com fuso horário.") from exc
        if scheduled.tzinfo is None or scheduled.utcoffset() is None:
            raise ValueError("Data/hora deve incluir fuso horário.")
        appointment_type = self._required(appointment_type, "Tipo de consulta")
        if appointment_type not in APPOINTMENT_TYPES:
            raise ValueError("Tipo de consulta não permitido.")
        appointment_id = f"SYN-A-{uuid.uuid4().hex[:12].upper()}"
        try:
            with self._transaction() as connection:
                self._authorized(connection, principal)
                if principal.role != Role.PATIENT:
                    raise AuthorizationError("Somente o paciente pode solicitar o próprio agendamento.")
                if not connection.execute("SELECT 1 FROM doctors WHERE doctor_id = ?", (doctor_id,)).fetchone():
                    raise ValueError("Profissional inexistente.")
                connection.execute(
                    """INSERT INTO appointments
                       (appointment_id, patient_id, doctor_id, scheduled_at, appointment_type, status, created_at)
                       VALUES (?, ?, ?, ?, ?, 'scheduled', ?)""",
                    (
                        appointment_id,
                        principal.user_id,
                        doctor_id,
                        scheduled.isoformat(timespec="minutes"),
                        appointment_type,
                        self._now(),
                    ),
                )
                self._record(connection, principal.user_id, principal.role.value, "appointment_created", appointment_id)
        except sqlite3.IntegrityError as exc:
            raise ConflictError("Horário indisponível ou agendamento duplicado.") from exc
        return {
            "appointment_id": appointment_id,
            "scheduled_at": scheduled.isoformat(timespec="minutes"),
            "status": "scheduled",
        }

    def list_doctors(self, principal: Principal, *, show_names: bool = False) -> list[dict[str, str]]:
        """Lista profissionais cadastrados sem expor credenciais ou nomes completos."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            rows = connection.execute(
                "SELECT doctor_id, full_name, specialty FROM doctors ORDER BY doctor_id"
            ).fetchall()
            self._record(connection, principal.user_id, principal.role.value, "doctors_listed", None)
            return [
                {
                    "doctor_id": row["doctor_id"],
                    "name": row["full_name"] if show_names else self._mask_name(row["full_name"]),
                    "specialty": row["specialty"],
                }
                for row in rows
            ]

    def list_appointments(
        self, principal: Principal, *, show_names: bool = False, show_patient_details: bool = False
    ) -> list[dict[str, str]]:
        """Lista somente os agendamentos do paciente ou do médico autenticado."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            owner_column = "a.patient_id" if principal.role == Role.PATIENT else "a.doctor_id"
            names = ", p.full_name AS patient_name, d.full_name AS doctor_name" if show_names else ""
            details = ", p.cpf, p.birth_date, p.phone, p.email, p.administrative_note" if show_patient_details else ""
            rows = connection.execute(
                f"""SELECT a.appointment_id, a.patient_id, a.doctor_id, a.scheduled_at,
                           a.appointment_type, a.status {names} {details}
                    FROM appointments a
                    JOIN patients p ON p.patient_id = a.patient_id
                    JOIN doctors d ON d.doctor_id = a.doctor_id
                    WHERE {owner_column} = ? ORDER BY a.scheduled_at""",
                (principal.user_id,),
            ).fetchall()
            self._record(connection, principal.user_id, principal.role.value, "appointments_listed", None)
            result = [dict(row) for row in rows]
            if show_patient_details:
                for item in result:
                    item["cpf"] = self._mask_cpf(item["cpf"])
                    item["phone"] = self._mask_phone(item["phone"])
                    item["email"] = self._mask_email(item["email"])
            return result

    def get_patient(
        self, principal: Principal, patient_id: str, *, show_name: bool = False, show_details: bool = False
    ) -> dict[str, str]:
        """Retorna dados próprios ou de paciente dentro do escopo de agenda do médico, sempre mascarados."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role == Role.PATIENT and patient_id != principal.user_id:
                raise AuthorizationError("Paciente só pode consultar os próprios dados.")
            if principal.role == Role.DOCTOR and not self._patient_in_doctor_scope(
                connection, principal.user_id, patient_id
            ):
                raise AuthorizationError("Paciente fora do escopo de autorização deste profissional.")
            row = connection.execute(
                "SELECT patient_id, full_name, cpf, birth_date, phone, email, administrative_note FROM patients WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Paciente inexistente.")
            self._record(connection, principal.user_id, principal.role.value, "patient_record_viewed", patient_id)
            result = {
                "patient_id": row["patient_id"],
                "name": row["full_name"] if show_name else self._mask_name(row["full_name"]),
                "cpf": self._mask_cpf(row["cpf"]),
                "birth_date": row["birth_date"] if show_details else "[DATA DE NASCIMENTO]",
                "phone": self._mask_phone(row["phone"]),
                "email": self._mask_email(row["email"]),
            }
            if show_details:
                result["administrative_note"] = row["administrative_note"]
            return result

    def list_patients(
        self, principal: Principal, *, show_names: bool = False, show_details: bool = False
    ) -> list[dict[str, str]]:
        """Médico consulta somente pacientes com vínculo em uma agenda sob sua responsabilidade."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role != Role.DOCTOR:
                raise AuthorizationError("Somente profissionais podem listar pacientes vinculados.")
            rows = connection.execute(
                """SELECT DISTINCT p.patient_id, p.full_name, p.cpf, p.birth_date, p.phone, p.email, p.administrative_note
                   FROM patients p JOIN appointments a ON a.patient_id = p.patient_id
                   WHERE a.doctor_id = ? AND a.status IN ('scheduled', 'completed')
                   ORDER BY p.patient_id""",
                (principal.user_id,),
            ).fetchall()
            self._record(connection, principal.user_id, principal.role.value, "patients_listed", None)
            result = [
                {
                    "patient_id": row["patient_id"],
                    "name": row["full_name"] if show_names else self._mask_name(row["full_name"]),
                    "cpf": self._mask_cpf(row["cpf"]),
                    "phone": self._mask_phone(row["phone"]),
                    "email": self._mask_email(row["email"]),
                }
                for row in rows
            ]
            if show_details:
                for item, row in zip(result, rows):
                    item["birth_date"] = row["birth_date"]
                    item["administrative_note"] = row["administrative_note"]
            return result

    def create_prescription_draft(
        self,
        principal: Principal,
        patient_id: str,
        prescription_text: str,
        *,
        authored_by_human: bool,
    ) -> str:
        """Registra texto fornecido pelo médico; não chama nem solicita geração por IA."""
        if not authored_by_human:
            raise AuthorizationError("Prescrição gerada ou completada por IA não pode ser registrada.")
        text = self._required(prescription_text, "Texto fornecido pelo profissional")
        if contains_pii(text, labels=None):
            raise ValueError("Remova identificadores pessoais do texto antes de registrá-lo.")
        prescription_id = f"SYN-RX-{uuid.uuid4().hex[:12].upper()}"
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role != Role.DOCTOR:
                raise AuthorizationError("Somente um profissional autenticado pode registrar uma proposta própria.")
            if not self._patient_in_doctor_scope(connection, principal.user_id, patient_id):
                raise AuthorizationError("Paciente fora do escopo de autorização deste profissional.")
            connection.execute(
                """INSERT INTO prescriptions
                   (prescription_id, patient_id, author_doctor_id, prescription_text, status, created_at)
                   VALUES (?, ?, ?, ?, 'pending_review', ?)""",
                (prescription_id, patient_id, principal.user_id, text, self._now()),
            )
            self._record(
                connection, principal.user_id, principal.role.value, "human_prescription_draft_created", prescription_id
            )
        return prescription_id

    def add_clinical_note(
        self,
        principal: Principal,
        patient_id: str,
        content: str,
        *,
        authored_by_human: bool,
    ) -> str:
        """Armazena anotação fictícia escrita por profissional, sem conteúdo produzido por IA."""
        if not authored_by_human:
            raise AuthorizationError("Anotações clínicas geradas ou completadas por IA não podem ser persistidas.")
        note = self._required(content, "Anotação")
        if contains_pii(note, labels=None):
            raise ValueError("Remova identificadores pessoais da anotação antes de registrá-la.")
        note_id = f"SYN-N-{uuid.uuid4().hex[:12].upper()}"
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role != Role.DOCTOR:
                raise AuthorizationError("Somente profissional autenticado pode registrar anotação fictícia.")
            if not self._patient_in_doctor_scope(connection, principal.user_id, patient_id):
                raise AuthorizationError("Paciente fora do escopo de autorização deste profissional.")
            connection.execute(
                """INSERT INTO clinical_notes(note_id, patient_id, author_doctor_id, content, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (note_id, patient_id, principal.user_id, note, self._now()),
            )
            self._record(connection, principal.user_id, principal.role.value, "clinical_note_created", note_id)
        return note_id

    def get_clinical_notes(self, principal: Principal, patient_id: str) -> list[dict[str, str]]:
        """Autoriza o escopo antes de devolver qualquer anotação para recuperação dinâmica."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role == Role.PATIENT and patient_id != principal.user_id:
                raise AuthorizationError("Paciente só pode consultar as próprias anotações.")
            if principal.role == Role.DOCTOR and not self._patient_in_doctor_scope(
                connection, principal.user_id, patient_id
            ):
                raise AuthorizationError("Paciente fora do escopo de autorização deste profissional.")
            rows = connection.execute(
                """SELECT note_id, content, created_at FROM clinical_notes
                   WHERE patient_id = ? ORDER BY created_at""",
                (patient_id,),
            ).fetchall()
            self._record(connection, principal.user_id, principal.role.value, "clinical_notes_retrieved", patient_id)
            return [dict(row) for row in rows]

    def get_doctor_patient_record(
        self, principal: Principal, patient_id: str, *, show_name: bool = False, show_details: bool = False
    ) -> dict[str, Any]:
        """Retorna os registros sintéticos existentes para médico com vínculo autorizado."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role != Role.DOCTOR:
                raise AuthorizationError("Somente profissional autenticado pode consultar este conjunto de registros.")
            if not self._patient_in_doctor_scope(connection, principal.user_id, patient_id):
                raise AuthorizationError("Paciente fora do escopo de autorização deste profissional.")
            patient = connection.execute(
                "SELECT patient_id, full_name, cpf, birth_date, phone, email, administrative_note FROM patients WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()
            if patient is None:
                raise ValueError("Paciente inexistente.")
            appointments = connection.execute(
                """SELECT appointment_id, doctor_id, scheduled_at, appointment_type, status, created_at
                   FROM appointments WHERE patient_id = ? ORDER BY scheduled_at, appointment_id""",
                (patient_id,),
            ).fetchall()
            notes = connection.execute(
                """SELECT note_id, author_doctor_id, content, created_at FROM clinical_notes
                   WHERE patient_id = ? ORDER BY created_at, note_id""",
                (patient_id,),
            ).fetchall()
            prescriptions = connection.execute(
                """SELECT prescription_id, author_doctor_id, prescription_text, status,
                          reviewer_doctor_id, created_at, reviewed_at
                   FROM prescriptions WHERE patient_id = ? ORDER BY created_at, prescription_id""",
                (patient_id,),
            ).fetchall()
            self._record(connection, principal.user_id, principal.role.value, "patient_records_viewed", patient_id)
            return {
                "patient": {
                    "patient_id": patient["patient_id"],
                    "name": patient["full_name"] if show_name else self._mask_name(patient["full_name"]),
                    "cpf": self._mask_cpf(patient["cpf"]),
                    "birth_date": patient["birth_date"] if show_details else "[DATA DE NASCIMENTO]",
                    "phone": self._mask_phone(patient["phone"]),
                    "email": self._mask_email(patient["email"]),
                    **({"administrative_note": patient["administrative_note"]} if show_details else {}),
                },
                "appointments": [dict(row) for row in appointments],
                "notes": [dict(row) for row in notes],
                "prescriptions": [dict(row) for row in prescriptions],
            }

    def review_prescription(
        self,
        principal: Principal,
        prescription_id: str,
        approve: bool,
        note: str = "",
    ) -> str:
        """Exige revisão por segundo profissional autorizado; a IA nunca aprova."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role != Role.DOCTOR:
                raise AuthorizationError("Somente um profissional pode revisar esta proposta.")
            row = connection.execute(
                """SELECT patient_id, author_doctor_id, status FROM prescriptions
                   WHERE prescription_id = ?""",
                (prescription_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Proposta inexistente.")
            if row["author_doctor_id"] == principal.user_id:
                raise AuthorizationError("O autor não pode aprovar a própria proposta.")
            if not self._patient_in_doctor_scope(connection, principal.user_id, row["patient_id"]):
                raise AuthorizationError("Paciente fora do escopo de autorização deste profissional.")
            if row["status"] != "pending_review":
                raise ConflictError("A proposta já foi revisada.")
            status = "approved" if approve else "rejected"
            connection.execute(
                """UPDATE prescriptions
                   SET status = ?, reviewer_doctor_id = ?, review_note_provided = ?, reviewed_at = ?
                   WHERE prescription_id = ? AND status = 'pending_review'""",
                (status, principal.user_id, int(bool(note.strip())), self._now(), prescription_id),
            )
            self._record(
                connection,
                principal.user_id,
                principal.role.value,
                "prescription_reviewed",
                prescription_id,
                status,
            )
            return status

    def list_prescriptions(self, principal: Principal, pending_only: bool = False) -> list[dict[str, str]]:
        """Lista metadados das prescrições no escopo do médico, sem divulgar o conteúdo clínico."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role != Role.DOCTOR:
                raise AuthorizationError("Este protótipo não disponibiliza prescrições para o perfil paciente.")
            status_filter = "AND r.status = 'pending_review'" if pending_only else ""
            rows = connection.execute(
                f"""SELECT r.prescription_id, r.patient_id, r.status, r.created_at
                    FROM prescriptions r
                    JOIN appointments a ON a.patient_id = r.patient_id AND a.doctor_id = ?
                    WHERE a.status IN ('scheduled', 'completed') {status_filter}
                    GROUP BY r.prescription_id ORDER BY r.created_at""",
                (principal.user_id,),
            ).fetchall()
            self._record(connection, principal.user_id, principal.role.value, "prescriptions_listed", None)
            return [dict(row) for row in rows]

    def read_audit_logs(self, principal: Principal) -> list[dict[str, Any]]:
        """Somente médicos autenticados leem eventos funcionais minimizados."""
        with self._transaction() as connection:
            self._authorized(connection, principal)
            if principal.role != Role.DOCTOR:
                raise AuthorizationError("Somente profissionais podem consultar a auditoria funcional.")
            rows = connection.execute(
                """SELECT timestamp, user_id, profile, action, entity_id, outcome
                   FROM audit_logs
                   WHERE user_id = ? OR profile = 'system'
                   ORDER BY audit_id""",
                (principal.user_id,),
            ).fetchall()
            self._record(connection, principal.user_id, principal.role.value, "audit_logs_read", None)
            return [dict(row) for row in rows]
