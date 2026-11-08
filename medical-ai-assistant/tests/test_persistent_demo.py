import sqlite3

import pytest

from src.database.sqlite import AuthorizationError, ClinicDatabase
from src.demo.persistent_setup import demo_key_path, setup_auto_persistent_demo, setup_persistent_demo


SECRET = "synthetic-test-secret-0123456789-abcdef"


def test_reopen_preserves_profiles_created_patients_and_appointments(tmp_path):
    path = tmp_path / "demo.sqlite3"
    first = setup_persistent_demo(path, SECRET)
    patient = first.database.register_patient_by_doctor(
        first.doctor, "Paciente Exemplo", "123.456.789-09", "30/01/1960",
        "(41) 99123-4567", "teste@example.test", "Observação sintética.",
        synthetic_demo_confirmed=True,
    )
    first.database.create_appointment(patient, first.doctor.user_id, "2031-02-01T10:00:00-03:00", "consulta_geral")
    second = setup_persistent_demo(path, SECRET)
    assert second.database.authenticate(patient.access_token) == patient
    assert second.database.get_patient(second.doctor, patient.user_id, show_details=True)["administrative_note"] == "Observação sintética."
    assert len(second.database.list_appointments(second.doctor)) == 3
    assert len(second.database.list_prescriptions(second.reviewer)) == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM doctors").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM clinical_notes").fetchone()[0] == 1
        assert patient.access_token not in str(connection.execute("SELECT * FROM patients").fetchall())
    assert len(second.additional_patients) == 1
    additional_patient = second.additional_patients[0]
    assert len(second.database.list_appointments(additional_patient)) == 1
    assert second.database.list_appointments(second.doctor)[1]["patient_id"] == additional_patient.user_id


def test_existing_demo_database_is_migrated_with_second_patient_and_private_agendas(tmp_path):
    path = tmp_path / "demo.sqlite3"
    initial = setup_persistent_demo(path, SECRET)
    extra = initial.additional_patients[0]
    with initial.database._transaction() as connection:
        connection.execute("DELETE FROM appointments WHERE appointment_id = 'SYN-A-DEMO-02'")
    initial.database.create_appointment(
        initial.patient,
        initial.doctor.user_id,
        "2030-01-17T10:00-03:00",
        "consulta_geral",
    )
    with initial.database._transaction() as connection:
        connection.execute("DELETE FROM patients WHERE patient_id = ?", (extra.user_id,))
        connection.execute("UPDATE demo_metadata SET version = 1")

    migrated = setup_persistent_demo(path, SECRET)
    patient_one, patient_two = migrated.patient, migrated.additional_patients[0]
    first_appointments = migrated.database.list_appointments(patient_one)
    second_appointments = migrated.database.list_appointments(patient_two)
    doctor_appointments = migrated.database.list_appointments(migrated.doctor)

    assert all(appointment["patient_id"] == patient_one.user_id for appointment in first_appointments)
    assert [appointment["patient_id"] for appointment in second_appointments] == [patient_two.user_id]
    assert second_appointments[0]["scheduled_at"] != "2030-01-17T10:00-03:00"
    assert {appointment["patient_id"] for appointment in doctor_appointments} == {
        patient_one.user_id,
        patient_two.user_id,
    }


def test_wrong_secret_does_not_replace_credentials_or_data(tmp_path):
    path = tmp_path / "demo.sqlite3"
    original = setup_persistent_demo(path, SECRET)
    with pytest.raises(AuthorizationError):
        setup_persistent_demo(path, "different-synthetic-secret-0123456789")
    reopened = setup_persistent_demo(path, SECRET)
    assert reopened.patient == original.patient
    assert len(reopened.database.list_appointments(reopened.patient)) == 2


def test_existing_conflicting_identity_is_not_overwritten(tmp_path):
    path = tmp_path / "existing.sqlite3"
    database = ClinicDatabase(path, bootstrap_token="bootstrap")
    doctor = database.provision_doctor(
        "SYN-D-DEMO", "Outro profissional", "Clínica", "bootstrap", synthetic_demo_confirmed=True,
    )
    with pytest.raises(RuntimeError, match="já existem"):
        setup_persistent_demo(path, SECRET)
    assert database.authenticate(doctor.access_token) == doctor


@pytest.mark.parametrize("secret", ["curta", " " + SECRET])
def test_weak_or_badly_padded_secret_does_not_create_database(tmp_path, secret):
    path = tmp_path / "demo.sqlite3"
    with pytest.raises(ValueError):
        setup_persistent_demo(path, secret)
    assert not path.exists()


def test_auto_key_is_generated_once_and_reused_without_prompt(tmp_path):
    path = tmp_path / "demo.sqlite3"
    first = setup_auto_persistent_demo(path)
    key_path = demo_key_path(path)
    initial_key = key_path.read_text(encoding="utf-8")
    assert len(initial_key) >= 32
    assert key_path.stat().st_mode & 0o077 == 0
    first.database.create_appointment(first.patient, first.doctor.user_id, "2031-02-01T10:00:00-03:00", "consulta_geral")
    second = setup_auto_persistent_demo(path)
    assert second.patient == first.patient
    assert key_path.read_text(encoding="utf-8") == initial_key
    assert len(second.database.list_appointments(second.patient)) == 3


def test_existing_database_requires_original_key_once(tmp_path):
    path = tmp_path / "demo.sqlite3"
    original = setup_persistent_demo(path, SECRET)
    with pytest.raises(RuntimeError, match="Chave automática ausente"):
        setup_auto_persistent_demo(path)
    with pytest.raises(AuthorizationError):
        setup_auto_persistent_demo(path, previous_secret="different-synthetic-secret-0123456789")
    assert not demo_key_path(path).exists()
    restored = setup_auto_persistent_demo(path, previous_secret=SECRET)
    assert restored.patient == original.patient
    assert setup_auto_persistent_demo(path).patient == original.patient


def test_existing_key_is_not_replaced_for_wrong_database(tmp_path):
    path = tmp_path / "demo.sqlite3"
    setup_persistent_demo(path, SECRET)
    key_path = demo_key_path(path)
    key_path.write_text("different-synthetic-secret-0123456789", encoding="utf-8")
    with pytest.raises(AuthorizationError):
        setup_auto_persistent_demo(path, previous_secret=SECRET)
    assert key_path.read_text(encoding="utf-8") == "different-synthetic-secret-0123456789"


def test_missing_database_with_existing_key_requires_restore(tmp_path):
    path = tmp_path / "demo.sqlite3"
    demo_key_path(path).write_text(SECRET, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Restaure o banco"):
        setup_auto_persistent_demo(path)
    assert not path.exists()
