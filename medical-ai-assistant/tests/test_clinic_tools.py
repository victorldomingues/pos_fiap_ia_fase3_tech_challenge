import pytest

from src.assistant.clinic_tools import build_clinic_tools
from src.database.sqlite import AuthorizationError


def test_langchain_tools_are_bound_to_authenticated_identity(clinic, retriever):
    database, doctor, _, patient = clinic
    tools = {item.name: item for item in build_clinic_tools(database, doctor, retriever)}
    assert set(tools) == {
        "appointment_tool",
        "patient_tool",
        "protocol_tool",
        "rag_tool",
        "prescription_tool",
    }
    with pytest.raises(AuthorizationError):
        tools["patient_tool"].invoke({"patient_id": patient.user_id})


def test_clinical_note_rag_tool_authorizes_patient_scope_before_retrieval(clinic, retriever, embeddings):
    from src.rag.patient_retriever import PatientNoteRetriever

    database, doctor, _, patient = clinic
    tools = {
        item.name: item
        for item in build_clinic_tools(
            database,
            doctor,
            retriever,
            PatientNoteRetriever(database, embeddings),
        )
    }
    with pytest.raises(AuthorizationError):
        tools["rag_tool"].invoke({"query": "retorno", "patient_id": patient.user_id})

    database.create_appointment(patient, doctor.user_id, "2030-01-15T10:00:00-03:00", "consulta_geral")
    database.add_clinical_note(
        doctor,
        patient.user_id,
        "Registro clínico fictício sobre retorno ambulatorial.",
        authored_by_human=True,
    )
    result = tools["rag_tool"].invoke({"query": "retorno", "patient_id": patient.user_id})
    assert "SYN-N-" in result


def test_clinic_demo_entrypoint_reuses_authenticated_demo_database(clinic):
    from scripts.clinic_demo import main

    database, doctor, _, _ = clinic
    inputs = iter(["medico", doctor.access_token, "0"])
    assert main([], database=database, input_fn=lambda _: next(inputs)) == 0
