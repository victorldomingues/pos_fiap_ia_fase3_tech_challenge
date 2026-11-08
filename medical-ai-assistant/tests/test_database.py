from __future__ import annotations

import sqlite3

import pytest

from src.database.sqlite import AuthorizationError, ClinicDatabase, ConflictError, Role


def test_registration_requires_synthetic_attestation_and_issues_authenticated_profile(clinic):
    database, _, _, patient = clinic
    with pytest.raises(ValueError, match="somente dados sintéticos"):
        database.register_patient("Nome", "cpf", "data", "telefone", "a@b.test", synthetic_demo_confirmed=False)

    authenticated = database.authenticate(patient.access_token)
    assert authenticated.user_id == patient.user_id
    assert authenticated.role == Role.PATIENT
    with pytest.raises(AuthorizationError):
        database.list_appointments(authenticated.__class__(patient.user_id, Role.DOCTOR, patient.access_token))


def test_doctor_registration_requires_authorization_and_appointment_scope(clinic):
    database, doctor, reviewer, patient = clinic
    arguments = ("Pessoa Fictícia", "123.456.789-09", "30/01/1960", "(41) 99123-4567",
                 "ficticia@example.test", "Acompanhante permitido.")
    with pytest.raises(AuthorizationError):
        database.register_patient_by_doctor(patient, *arguments, synthetic_demo_confirmed=True)
    with pytest.raises(AuthorizationError):
        database.register_patient_by_doctor(
            doctor.__class__(doctor.user_id, Role.DOCTOR, "invalid"), *arguments, synthetic_demo_confirmed=True
        )
    with pytest.raises(ValueError):
        database.register_patient_by_doctor(doctor, *arguments, synthetic_demo_confirmed=False)
    new_patient = database.register_patient_by_doctor(doctor, *arguments, synthetic_demo_confirmed=True)
    assert database.get_patient(new_patient, new_patient.user_id, show_name=True, show_details=True) == {
        "patient_id": new_patient.user_id, "name": "Pessoa Fictícia", "cpf": "***.***.***-09",
        "birth_date": "1960-01-30", "phone": "(**) *****-4567", "email": "f***@example.test",
        "administrative_note": "Acompanhante permitido.",
    }
    with pytest.raises(AuthorizationError):
        database.get_patient(doctor, new_patient.user_id, show_details=True)
    with pytest.raises(AuthorizationError):
        database.get_patient(patient, new_patient.user_id, show_details=True)
    database.create_appointment(new_patient, doctor.user_id, "2031-02-01T10:00:00-03:00", "consulta_geral")
    assert database.get_patient(doctor, new_patient.user_id, show_details=True)["birth_date"] == "1960-01-30"
    with pytest.raises(AuthorizationError):
        database.get_patient(reviewer, new_patient.user_id, show_details=True)
    appointment = database.list_appointments(doctor, show_names=True, show_patient_details=True)[0]
    assert appointment["cpf"] == "***.***.***-09" and appointment["phone"] == "(**) *****-4567"
    assert appointment["email"] == "f***@example.test"
    events = database.read_audit_logs(doctor)
    assert any(event["action"] == "patient_registered_by_doctor" for event in events)
    assert "123.456.789-09" not in str(events) and "Acompanhante permitido" not in str(events)


@pytest.mark.parametrize("index,value", [
    (1, "123"), (2, "31/02/1960"), (3, "123"), (4, "invalido"),
    (5, "Contato: pessoa@example.test"),
])
def test_doctor_registration_rejects_invalid_fields(clinic, index, value):
    database, doctor, _, _ = clinic
    args = ["Pessoa Fictícia", "123.456.789-09", "30/01/1960", "(41) 99123-4567",
            "ficticia@example.test", "Observação administrativa."]
    args[index] = value
    with pytest.raises(ValueError):
        database.register_patient_by_doctor(doctor, *args, synthetic_demo_confirmed=True)


def test_schema_v2_migrates_existing_patients(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE schema_metadata (version INTEGER NOT NULL);
            INSERT INTO schema_metadata VALUES (2);
            CREATE TABLE patients (
                patient_id TEXT PRIMARY KEY, full_name TEXT NOT NULL, cpf TEXT NOT NULL,
                birth_date TEXT NOT NULL, phone TEXT NOT NULL, email TEXT NOT NULL,
                access_token_hash TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
            );
            INSERT INTO patients VALUES ('SYN-P-OLD', 'Paciente', '000', '1980-01-01',
                '000', 'a@example.test', 'hash', '2020-01-01');
        """)
    ClinicDatabase(path)
    ClinicDatabase(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT version FROM schema_metadata").fetchone()[0] == 3
        assert connection.execute("SELECT administrative_note FROM patients").fetchone()[0] == ""


def test_patient_and_doctor_access_is_limited_to_authorized_records(clinic):
    database, doctor, _, patient = clinic
    with pytest.raises(ValueError, match="fuso"):
        database.create_appointment(patient, doctor.user_id, "2030-01-15T10:00", "consulta_geral")

    booking = database.create_appointment(
        patient, doctor.user_id, "2030-01-15T10:00:00-03:00", "consulta_geral"
    )
    assert database.list_appointments(patient)[0]["appointment_id"] == booking["appointment_id"]
    assert database.list_appointments(doctor)[0]["patient_id"] == patient.user_id
    with pytest.raises(AuthorizationError, match="próprios dados"):
        database.get_patient(patient, "SYN-P-OUTRO")

    record = database.get_patient(doctor, patient.user_id)
    assert record["cpf"] == "***.***.***-35"
    assert record["phone"] == "(**) *****-5678"
    assert record["birth_date"] == "[DATA DE NASCIMENTO]"
    assert "Paciente" not in record["name"]


def test_patient_agenda_is_private_while_doctor_sees_all_assigned_patients(clinic):
    database, doctor, _, first_patient = clinic
    second_patient = database.register_patient(
        "Segundo Paciente Sintético",
        "222.555.888-46",
        "1990-05-10",
        "(21) 99876-5432",
        "paciente2@example.test",
        synthetic_demo_confirmed=True,
    )
    database.create_appointment(first_patient, doctor.user_id, "2031-02-01T10:00:00-03:00", "consulta_geral")
    database.create_appointment(second_patient, doctor.user_id, "2031-02-02T10:00:00-03:00", "retorno_ambulatorial")

    first_agenda = database.list_appointments(first_patient, show_names=True)
    second_agenda = database.list_appointments(second_patient, show_names=True)
    doctor_agenda = database.list_appointments(doctor, show_names=True)

    assert first_agenda and all(item["patient_id"] == first_patient.user_id for item in first_agenda)
    assert second_agenda and all(item["patient_id"] == second_patient.user_id for item in second_agenda)
    assert {item["patient_id"] for item in doctor_agenda} == {first_patient.user_id, second_patient.user_id}


def test_doctor_scope_and_credential_are_checked_on_every_operation(clinic):
    database, doctor, reviewer, patient = clinic
    with pytest.raises(AuthorizationError):
        database.get_patient(doctor, patient.user_id)
    with pytest.raises(AuthorizationError):
        database.get_patient(doctor.__class__(doctor.user_id, Role.DOCTOR, "invalid"), patient.user_id)

    database.create_appointment(patient, reviewer.user_id, "2030-01-15T10:00:00-03:00", "consulta_geral")
    assert database.get_patient(reviewer, patient.user_id)["patient_id"] == patient.user_id


def test_explicit_name_display_keeps_authorization_and_document_masking(clinic):
    database, doctor, reviewer, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2031-01-15T10:00:00-03:00", "consulta_geral")
    assert database.list_doctors(patient, show_names=True)[0]["name"] == "Doutora Demonstração"
    assert database.list_patients(doctor, show_names=True)[0]["name"] == "Paciente Sintético"
    record = database.get_doctor_patient_record(doctor, patient.user_id, show_name=True)["patient"]
    assert record["name"] == "Paciente Sintético" and record["cpf"] == "***.***.***-35"
    assert database.get_patient(patient, patient.user_id, show_name=True)["name"] == "Paciente Sintético"
    appointment = database.list_appointments(patient, show_names=True)[0]
    assert appointment["doctor_name"] == "Doutora Demonstração"
    assert appointment["patient_name"] == "Paciente Sintético"
    assert "doctor_name" not in database.list_appointments(patient)[0]
    with pytest.raises(AuthorizationError):
        database.get_doctor_patient_record(reviewer, patient.user_id, show_name=True)
    with pytest.raises(AuthorizationError):
        database.list_patients(patient, show_names=True)


def test_appointment_conflict_is_reported_without_leaking_patient_data(clinic):
    database, doctor, _, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2030-01-15T10:00:00-03:00", "consulta_geral")
    with pytest.raises(ConflictError, match="Horário indisponível"):
        database.create_appointment(
            patient,
            doctor.user_id,
            "2030-01-15T10:00:00-03:00",
            "retorno_ambulatorial",
        )


def test_prescription_is_human_authored_reviewed_and_never_exposed_to_patient(clinic):
    database, doctor, reviewer, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2030-01-15T10:00:00-03:00", "consulta_geral")
    database.create_appointment(patient, reviewer.user_id, "2030-01-16T10:00:00-03:00", "consulta_geral")
    with pytest.raises(AuthorizationError, match="gerada ou completada por IA"):
        database.create_prescription_draft(
            doctor,
            patient.user_id,
            "Texto fictício",
            authored_by_human=False,
        )

    prescription_id = database.create_prescription_draft(
        doctor,
        patient.user_id,
        "Texto de decisão clínica fictícia fornecido pelo profissional.",
        authored_by_human=True,
    )
    with pytest.raises(AuthorizationError, match="própria proposta"):
        database.review_prescription(doctor, prescription_id, approve=True)
    assert database.review_prescription(reviewer, prescription_id, approve=True, note="Revisado") == "approved"
    assert database.list_prescriptions(reviewer)[0]["status"] == "approved"
    with pytest.raises(AuthorizationError, match="perfil paciente"):
        database.list_prescriptions(patient)

    logs = database.read_audit_logs(doctor)
    assert any(log["action"] == "human_prescription_draft_created" for log in logs)
    reviewer_logs = database.read_audit_logs(reviewer)
    assert any(log["action"] == "prescription_reviewed" for log in reviewer_logs)
    assert all("Texto de decisão clínica" not in str(log) for log in logs)
    assert all("111.444.777-35" not in str(log) for log in logs)


def test_clinical_notes_require_human_author_and_authorized_patient(clinic):
    database, doctor, _, patient = clinic
    with pytest.raises(AuthorizationError, match="geradas ou completadas por IA"):
        database.add_clinical_note(
            doctor,
            patient.user_id,
            "Anotação fictícia",
            authored_by_human=False,
        )
    with pytest.raises(AuthorizationError):
        database.add_clinical_note(
            doctor,
            patient.user_id,
            "Anotação fictícia",
            authored_by_human=True,
        )
    database.create_appointment(patient, doctor.user_id, "2030-01-15T10:00:00-03:00", "consulta_geral")
    note_id = database.add_clinical_note(
        doctor,
        patient.user_id,
        "Registro clínico fictício sobre retorno ambulatorial.",
        authored_by_human=True,
    )
    assert database.get_clinical_notes(patient, patient.user_id)[0]["note_id"] == note_id
    with pytest.raises(AuthorizationError):
        database.get_clinical_notes(patient, "SYN-P-OUTRO")


def test_doctor_record_contains_only_authorized_patient_records_and_human_proposals(clinic):
    database, doctor, reviewer, patient = clinic
    with pytest.raises(AuthorizationError):
        database.get_doctor_patient_record(doctor, patient.user_id)
    with pytest.raises(AuthorizationError):
        database.get_doctor_patient_record(patient, patient.user_id)
    database.create_appointment(patient, doctor.user_id, "2030-01-15T10:00:00-03:00", "consulta_geral")
    database.create_appointment(patient, reviewer.user_id, "2030-01-16T10:00:00-03:00", "consulta_geral")
    note = database.add_clinical_note(reviewer, patient.user_id, "Nota sintética humana.", authored_by_human=True)
    proposal = database.create_prescription_draft(
        reviewer, patient.user_id, "Proposta sintética humana.", authored_by_human=True
    )
    records = database.get_doctor_patient_record(doctor, patient.user_id)
    assert records["patient"]["cpf"] == "***.***.***-35"
    assert len(records["appointments"]) == 2
    assert records["notes"][0]["note_id"] == note
    assert records["prescriptions"][0]["prescription_id"] == proposal
    assert records["prescriptions"][0]["status"] == "pending_review"
    assert records["prescriptions"][0]["prescription_text"] == "Proposta sintética humana."
    assert any(log["action"] == "patient_records_viewed" for log in database.read_audit_logs(doctor))
    with pytest.raises(AuthorizationError):
        database.get_doctor_patient_record(doctor.__class__(doctor.user_id, Role.DOCTOR, "invalid"), patient.user_id)
    other_patient = database.register_patient(
        "Outra Pessoa Sintética", "111.444.777-35", "1980-01-02", "(11) 91234-5678",
        "outro@example.test", synthetic_demo_confirmed=True
    )
    database.create_appointment(other_patient, reviewer.user_id, "2030-01-17T10:00:00-03:00", "consulta_geral")
    with pytest.raises(AuthorizationError):
        database.get_doctor_patient_record(doctor, other_patient.user_id)
    with pytest.raises(AuthorizationError):
        database.get_doctor_patient_record(doctor, "SYN-P-OUTRO")


def test_sqlite_foreign_keys_and_no_patient_contact_in_audit_database(clinic):
    database, doctor, _, patient = clinic
    with pytest.raises(sqlite3.IntegrityError):
        with database._transaction() as connection:
            connection.execute(
                """INSERT INTO appointments
                   (appointment_id, patient_id, doctor_id, scheduled_at, appointment_type, status, created_at)
                   VALUES ('bad', ?, ?, '2030-01-15T10:00:00-03:00', 'consulta_geral', 'scheduled', 'now')""",
                ("SYN-P-UNKNOWN", doctor.user_id),
            )
    logs = database.read_audit_logs(doctor)
    assert all(patient.access_token not in str(log) for log in logs)
    assert all("paciente@example.test" not in str(log) for log in logs)
