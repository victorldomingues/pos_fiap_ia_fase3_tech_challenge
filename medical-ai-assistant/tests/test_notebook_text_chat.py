"""Exercita a célula textual do notebook com entrada simulada."""

import builtins
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "IADT 3 - Victor Luiz Domingues.ipynb"


def execute_chat(monkeypatch, assistant, inputs, **context):
    cell = next(
        cell for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell.get("id") == "interactive-chat"
    )
    entries = iter(inputs)
    monkeypatch.setattr(builtins, "input", lambda prompt: next(entries))
    if "doctor_demo" in context:
        context.setdefault("reviewer_demo", context["doctor_demo"])
    context.setdefault("additional_patient_demos", ())
    namespace = {"assistant": assistant, **context}
    exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)


def test_text_chat_answer_refusal_and_exit(monkeypatch, assistant, clinic, capsys):
    database, doctor, _, patient = clinic
    execute_chat(
        monkeypatch, assistant, ["", "1", "Qual é o horário de visitas?", "Qual a dose de paracetamol?", "/sair"],
        sqlite_demo=database, doctor_demo=doctor, patient_demo=patient,
    )
    output = capsys.readouterr().out
    assert "Escolha um perfil" in output
    assert "Sessão sintética autenticada: paciente" in output and "/profissionais" in output
    assert "PROT-005" in output and "Validação humana obrigatória." in output
    assert "dose_medicamento" in output
    assert "Chat encerrado." in output
    assert assistant.audit.read()[-1]["user_role"] == "patient"


def test_text_chat_handles_records_with_authenticated_scope(monkeypatch, clinic, capsys):
    database, doctor, _, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2030-01-15T10:00:00-03:00", "consulta_geral")
    database.add_clinical_note(doctor, patient.user_id, "Nota sintética.", authored_by_human=True)
    database.create_prescription_draft(doctor, patient.user_id, "Proposta sintética.", authored_by_human=True)
    assistant = MagicMock()
    execute_chat(
        monkeypatch, assistant,
        ["1", "meu prontuário", "/registros", "/logout", "2",
         "/registros", "inválido", "/registros", "1", "/sair"],
        sqlite_demo=database, doctor_demo=doctor, patient_demo=patient,
    )
    output = capsys.readouterr().out
    assert "use /registros" in output
    assert "Sessão sintética autenticada: médico" in output
    assert "Seleção inválida" in output
    assert output.count("Nota sintética.") == 2
    assert output.count("Proposta sintética.") == 1
    assert "111.444.777-35" not in output
    assistant.ask.assert_not_called()


def test_text_chat_assistant_error(monkeypatch, clinic, capsys):
    database, doctor, _, patient = clinic
    assistant = MagicMock()
    assistant.ask.side_effect = RuntimeError("detalhe sensível")
    execute_chat(monkeypatch, assistant, ["2", "pergunta", "/sair"],
                 sqlite_demo=database, doctor_demo=doctor, patient_demo=patient)
    output = capsys.readouterr().out
    assert "RuntimeError" in output and "detalhe sensível" not in output
    assistant.ask.assert_called_once_with("pergunta", user_id=doctor.user_id, user_role="doctor")


def test_only_authenticated_doctor_can_review_assistant_validations(monkeypatch, clinic, assistant, capsys):
    from src.assistant.explain import render_response

    database, doctor, _, patient = clinic
    response = assistant.ask(
        "Qual é o horário de visitas?",
        user_id=patient.user_id,
        user_role="patient",
    )
    assert response.review_status.value == "pendente"
    assert "Validação humana obrigatória." in render_response(response)

    execute_chat(
        monkeypatch,
        assistant,
        ["1", "/validacoes", f"/aprovar-validacao {response.request_id}", "/sair"],
        sqlite_demo=database,
        doctor_demo=doctor,
        patient_demo=patient,
    )
    patient_output = capsys.readouterr().out
    assert "somente para o perfil médico" in patient_output
    assert assistant.approvals.status(response.request_id).value == "pendente"

    execute_chat(
        monkeypatch,
        assistant,
        ["2", "/validacoes", f"/aprovar-validacao {response.request_id}", "/sair"],
        sqlite_demo=database,
        doctor_demo=doctor,
        patient_demo=patient,
    )
    doctor_output = capsys.readouterr().out
    assert response.request_id in doctor_output
    assert response.question in doctor_output
    assert response.answer in doctor_output
    assert "Trecho recuperado:" in doctor_output
    assert "não torna a resposta recomendação clínica" in doctor_output
    assert assistant.approvals.status(response.request_id).value == "aprovada"
    review_event = next(event for event in assistant.audit.read() if event["event"] == "review")
    assert review_event["reviewer"] == doctor.user_id
    assert review_event["request_id"] == response.request_id
    assert assistant.approvals.verify() and assistant.audit.verify()


def test_doctor_cannot_approve_pending_validation_without_review_content(monkeypatch, clinic, assistant, capsys):
    database, doctor, _, patient = clinic
    assistant.approvals.submit("legacy-request", patient.user_id)
    execute_chat(
        monkeypatch,
        assistant,
        ["2", "/validacoes", "/aprovar-validacao legacy-request", "/sair"],
        sqlite_demo=database,
        doctor_demo=doctor,
        patient_demo=patient,
    )
    output = capsys.readouterr().out
    assert "Conteúdo indisponível para revisão" in output
    assert "não foi aprovada" in output
    assert assistant.approvals.status("legacy-request").value == "pendente"


def test_text_chat_requires_sqlite_identities(monkeypatch):
    with pytest.raises(RuntimeError, match="Parte 5"):
        execute_chat(monkeypatch, MagicMock(), [])


def test_text_chat_profile_menu_and_access(monkeypatch, clinic, capsys):
    database, doctor, _, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2030-01-15T10:00:00-03:00", "consulta_geral")
    execute_chat(
        monkeypatch, MagicMock(), ["9", "1", "/ajuda", "/agenda", "/profissionais", "/trocar",
                                   "2", "/ajuda", "/profissionais", "/agenda", "/sair"],
        sqlite_demo=database, doctor_demo=doctor, patient_demo=patient,
    )
    output = capsys.readouterr().out
    assert "Perfil inválido" in output
    assert "Paciente: /profissionais" in output and "Médico: /agenda" in output
    assert "Este fluxo está disponível para o perfil paciente" in output
    assert output.count(patient.user_id) >= 2


def test_patient_chat_can_select_only_own_agenda_and_doctor_sees_multiple_patients(
    monkeypatch, clinic, assistant, capsys
):
    database, doctor, _, patient = clinic
    second_patient = database.register_patient(
        "Segundo Paciente Sintético",
        "222.555.888-46",
        "1990-05-10",
        "(21) 99876-5432",
        "paciente2@example.test",
        synthetic_demo_confirmed=True,
    )
    first_booking = database.create_appointment(
        patient, doctor.user_id, "2031-02-01T10:00:00-03:00", "consulta_geral"
    )
    second_booking = database.create_appointment(
        second_patient, doctor.user_id, "2031-02-02T10:00:00-03:00", "retorno_ambulatorial"
    )

    execute_chat(
        monkeypatch,
        assistant,
        ["1", "1", "/agenda", "/sair"],
        sqlite_demo=database,
        doctor_demo=doctor,
        patient_demo=patient,
        additional_patient_demos=(second_patient,),
    )
    first_output = capsys.readouterr().out
    first_agenda = first_output.split("── Agenda ──", 1)[1]
    assert first_booking["appointment_id"] in first_agenda
    assert second_booking["appointment_id"] not in first_agenda
    assert second_patient.user_id not in first_agenda

    execute_chat(
        monkeypatch,
        assistant,
        ["1", "2", "/agenda", "/sair"],
        sqlite_demo=database,
        doctor_demo=doctor,
        patient_demo=patient,
        additional_patient_demos=(second_patient,),
    )
    second_output = capsys.readouterr().out
    second_agenda = second_output.split("── Agenda ──", 1)[1]
    assert second_booking["appointment_id"] in second_agenda
    assert first_booking["appointment_id"] not in second_agenda
    assert patient.user_id not in second_agenda

    execute_chat(
        monkeypatch,
        assistant,
        ["2", "/agenda", "/sair"],
        sqlite_demo=database,
        doctor_demo=doctor,
        patient_demo=patient,
        additional_patient_demos=(second_patient,),
    )
    doctor_agenda = capsys.readouterr().out.split("── Agenda ──", 1)[1]
    assert first_booking["appointment_id"] in doctor_agenda
    assert second_booking["appointment_id"] in doctor_agenda
    assert patient.user_id in doctor_agenda and second_patient.user_id in doctor_agenda


def test_text_chat_invalid_or_mismatched_demo_credential(monkeypatch, clinic, capsys):
    database, doctor, _, patient = clinic
    execute_chat(monkeypatch, MagicMock(), ["1", "2", "/sair"],
                 sqlite_demo=database, patient_demo=doctor, doctor_demo=doctor)
    output = capsys.readouterr().out
    assert "A credencial não corresponde ao perfil escolhido" in output
    assert "Sessão sintética autenticada: médico" in output


def test_text_chat_rechecks_credential_before_general_question(monkeypatch, clinic, capsys):
    from src.database.sqlite import AuthorizationError

    database, doctor, _, patient = clinic
    authenticate = database.authenticate
    calls = 0

    def revoked(token):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AuthorizationError("revogada")
        return authenticate(token)

    monkeypatch.setattr(database, "authenticate", revoked)
    assistant = MagicMock()
    execute_chat(monkeypatch, assistant, ["1", "Qual é o horário de visitas?", "/sair"],
                 sqlite_demo=database, doctor_demo=doctor, patient_demo=patient)
    assert "Credencial inválida. Escolha novamente o perfil." in capsys.readouterr().out
    assistant.ask.assert_not_called()


def test_text_chat_requires_assistant(monkeypatch):
    cell = next(
        cell for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell.get("id") == "interactive-chat"
    )
    with pytest.raises(RuntimeError, match="3.3"):
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), {})


def test_module_chat_accepts_explicit_io(clinic):
    from src.demo.text_chat import run_notebook_text_chat

    database, doctor, _, patient = clinic
    inputs = iter(["1", "/sair"])
    outputs = []
    run_notebook_text_chat(MagicMock(), database, patient, doctor,
                           input_fn=lambda prompt: next(inputs), output_fn=outputs.append)
    assert "Sessão sintética autenticada: paciente" in outputs
    assert outputs[-1] == "Chat encerrado."


def test_guided_booking_confirmation_and_conflict(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    inputs = [
        "1", "5", "1", "2031-02-01T10:00:00-03:00", "1", "NAO",
        "/agendar", "1", "2031-02-01T10:00:00-03:00", "1", "SIM",
        "/agendar", "1", "2031-02-01T10:00:00-03:00", "1", "SIM", "/sair",
    ]
    execute_chat(monkeypatch, MagicMock(), inputs, sqlite_demo=database, patient_demo=patient,
                 doctor_demo=doctor, reviewer_demo=reviewer)
    output = capsys.readouterr().out
    assert "Agendamento cancelado." in output
    assert "Agendamento criado:" in output
    assert "Conflito: horário indisponível" in output
    assert len(database.list_appointments(patient)) == 1


def test_guided_human_record_and_review_by_other_doctor(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2031-01-15T10:00:00-03:00", "consulta_geral")
    database.create_appointment(patient, reviewer.user_id, "2031-01-16T10:00:00-03:00", "consulta_geral")
    execute_chat(
        monkeypatch, MagicMock(),
        ["2", "4", "1", "Anotação sintética humana", "SIM",
         "5", "1", "Proposta sintética humana", "SIM",
         "6", "1", "/trocar", "3", "6", "1", "APROVAR", "SIM", "/sair"],
        sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer,
    )
    output = capsys.readouterr().out
    assert "Anotação humana registrada:" in output
    assert "O autor não pode revisar a própria proposta" in output
    assert "Texto humano existente para revisão" in output
    assert "Revisão humana registrada: approved" in output
    assert database.list_prescriptions(reviewer)[0]["status"] == "approved"


def test_guided_cancel_and_role_restrictions(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2031-01-15T10:00:00-03:00", "consulta_geral")
    execute_chat(
        monkeypatch, MagicMock(),
        ["1", "/proposta", "/revisar", "/auditoria", "/agendar", "/cancelar",
         "/trocar", "2", "/anotar", "1", "Conteúdo fictício", "NAO", "/sair"],
        sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer,
    )
    output = capsys.readouterr().out
    assert output.count("Este fluxo está disponível para o perfil médico.") == 3
    assert "Operação cancelada." in output and "Registro cancelado." in output
    assert database.get_clinical_notes(patient, patient.user_id) == []


def test_guided_note_retrieval_uses_authorized_patient(monkeypatch, assistant, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2031-01-15T10:00:00-03:00", "consulta_geral")
    database.add_clinical_note(doctor, patient.user_id, "Retorno sintético agendado.", authored_by_human=True)
    execute_chat(monkeypatch, assistant, ["1", "3", "retorno agendado", "/sair"],
                 sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer)
    output = capsys.readouterr().out
    assert "Retorno sintético agendado." in output
    assert "similaridade" in output


def test_notebook_setup_and_reviewer_flow(monkeypatch, assistant, capsys, tmp_path):
    import getpass

    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    setup = next(cell for cell in cells if cell.get("id") == "sqlite-profiles-demo")
    chat = next(cell for cell in cells if cell.get("id") == "interactive-chat")
    entries = iter(["3", "/revisar", "1", "REJEITAR", "SIM", "/sair"])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(entries))
    def unexpected_prompt(prompt):
        raise AssertionError("A inicialização automática não deve pedir a chave.")

    monkeypatch.setattr(getpass, "getpass", unexpected_prompt)
    namespace = {"assistant": assistant, "PROJECT_DIR": tmp_path, "IN_COLAB": False}
    exec(compile("".join(setup["source"]), str(NOTEBOOK), "exec"), namespace)
    exec(compile("".join(chat["source"]), str(NOTEBOOK), "exec"), namespace)
    assert "Revisão humana registrada: rejected" in capsys.readouterr().out
    exec(compile("".join(setup["source"]), str(NOTEBOOK), "exec"), namespace)
    assert namespace["sqlite_demo"].list_prescriptions(namespace["reviewer_demo"])[0]["status"] == "rejected"


def test_notebook_migrates_existing_key_once(monkeypatch, tmp_path):
    import getpass

    from src.demo.persistent_setup import setup_persistent_demo

    path = tmp_path / "data" / "sqlite" / "notebook_demo.sqlite3"
    original = setup_persistent_demo(path, "old-synthetic-test-secret-0123456789")
    cell = next(cell for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
                if cell.get("id") == "sqlite-profiles-demo")
    calls = []

    def prompt(message):
        calls.append(message)
        return "old-synthetic-test-secret-0123456789"

    monkeypatch.setattr(getpass, "getpass", prompt)
    namespace = {"PROJECT_DIR": tmp_path, "IN_COLAB": False}
    code = compile("".join(cell["source"]), str(NOTEBOOK), "exec")
    exec(code, namespace)
    exec(code, namespace)
    assert len(calls) == 1
    assert namespace["patient_demo"] == original.patient
    assert len(namespace["sqlite_demo"].list_appointments(namespace["patient_demo"])) == 2


def test_guided_booking_rejects_missing_timezone_before_confirmation(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    execute_chat(monkeypatch, MagicMock(), ["1", "/agendar", "1", "2031-01-15T10:00", "/cancelar", "/sair"],
                 sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer)
    assert "Informe o fuso horário" in capsys.readouterr().out
    assert database.list_appointments(patient) == []


def test_booking_retries_invalid_input_and_context_does_not_call_rag(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    assistant = MagicMock()
    execute_chat(
        monkeypatch, assistant,
        ["1", "5", "1", "", "02026-11-08T10:00:00-03:00)", "2031-01-15T10:00:00-03:00",
         "9", "1", "SIM", "O que eu faço agora?", "Não entendi",
         "02026-11-08T10:00:00-03:00)", "/sair"],
        sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer,
    )
    output = capsys.readouterr().out
    assert output.count("Data inválida") == 2
    assert "Tipo inválido" in output and "Agendamento criado" in output
    assert "Doutora Demonstração" in output
    assert output.count("Seu agendamento foi registrado.") == 2
    assert "parece uma data ou número" in output
    assert len(database.list_appointments(patient)) == 1
    assistant.ask.assert_not_called()


def test_doctor_names_visible_only_with_authorized_scope(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2031-01-15T10:00:00-03:00", "consulta_geral")
    execute_chat(monkeypatch, MagicMock(), ["2", "/registros", "1", "/sair"],
                 sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer)
    output = capsys.readouterr().out
    assert "Paciente Sintético" in output
    assert "111.444.777-35" not in output


def test_read_only_lists_are_numbered_and_explain_next_steps(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    database.create_appointment(patient, doctor.user_id, "2031-01-15T10:00:00-03:00", "consulta_geral")
    database.create_appointment(patient, reviewer.user_id, "2031-01-16T10:00:00-03:00", "consulta_geral")
    assistant = MagicMock()
    execute_chat(monkeypatch, assistant, ["1", "/agenda", "/profissionais", "/registros", "99", "/sair"],
                 sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer)
    output = capsys.readouterr().out
    agenda = output.split("── Agenda ──", 1)[1].split("── Profissionais disponíveis ──", 1)[0]
    professionals = output.split("── Profissionais disponíveis ──", 1)[1].split(
        "Registros sintéticos disponíveis", 1
    )[0]
    assert "1.\n  Agendamento:" in agenda and "2.\n  Agendamento:" in agenda
    assert "1.\n  Profissional:" in professionals and "2.\n  Profissional:" in professionals
    assert "não os selecionam" in agenda and "não os selecionam" in professionals
    assert "digite /agendar" in professionals
    assert "Para buscar nas anotações, digite /buscar" in output
    assert "Opção do menu inválida" in output
    assistant.ask.assert_not_called()


def test_empty_agenda_and_number_after_directory_use_menu(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    assistant = MagicMock()
    execute_chat(monkeypatch, assistant, ["1", "/profissionais", "1", "/sair"],
                 sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer)
    output = capsys.readouterr().out
    assert output.index("── Profissionais disponíveis ──") < output.index("── Agenda ──")
    assert "Nenhum registro disponível." in output
    assert "Lista apenas para consulta: os números identificam itens, não os selecionam." in output
    assert "Para agendar uma consulta, digite /agendar" in output
    assistant.ask.assert_not_called()


def test_selectable_professionals_have_clear_numbered_prompt(monkeypatch, clinic):
    from src.demo.text_chat import run_notebook_text_chat

    database, doctor, reviewer, patient = clinic
    values = iter(["1", "/agendar", "2", "2031-02-01T10:00:00-03:00", "1", "SIM", "/sair"])
    outputs = []
    prompts = []

    def read(prompt):
        prompts.append(prompt)
        return next(values)

    run_notebook_text_chat(MagicMock(), database, patient, doctor,
                           input_fn=read, output_fn=outputs.append, reviewer_demo=reviewer)
    assert "Selecione um profissional pelo número da lista; digite /cancelar para voltar." in outputs
    assert any(line.startswith("1. Doutora Demonstração") for line in outputs)
    assert any(line.startswith("2. Médico Revisor") for line in outputs)
    assert any("Número do profissional" in prompt for prompt in prompts)
    assert database.list_appointments(patient)[0]["doctor_id"] == reviewer.user_id


def test_doctor_registers_patient_switches_session_and_books(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    assistant = MagicMock()
    registered = []
    register = database.register_patient_by_doctor

    def track_registration(*args, **kwargs):
        created = register(*args, **kwargs)
        registered.append(created)
        with pytest.raises(PermissionError):
            database.get_patient(doctor, created.user_id)
        return created

    monkeypatch.setattr(database, "register_patient_by_doctor", track_registration)
    execute_chat(
        monkeypatch, assistant,
        ["2", "8", "SIM", "Pessoa Fictícia", "inválido", "123.456.789-09",
         "30/02/1960", "30/01/1960", "(41) 99123-4567", "ficticia@example.test",
         "Acompanhante permitido.", "SIM", "/trocar", "1", "9", "2",
         "/agendar", "1", "2031-02-01T10:00:00-03:00", "1", "SIM", "/agenda",
         "/trocar", "2", "/registros", "1", "/sair"],
        sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer,
    )
    output = capsys.readouterr().out
    assert f"Identidade ativa: Doutora Demonstração — {doctor.user_id}" in output
    assert f"Identidade ativa: Pessoa Fictícia — {registered[0].user_id}" in output
    assert "CPF inválido" in output and "Nascimento inválido" in output
    assert "Seleção inválida" in output and "Agendamento criado" in output
    assert "Nascimento: 1960-01-30" in output and "Observação administrativa: Acompanhante permitido." in output
    assert "***.***.***-09" in output and "(**) *****-4567" in output and "f***@example.test" in output
    assert "123.456.789-09" not in output and "(41) 99123-4567" not in output
    assert "ficticia@example.test" not in output and registered[0].access_token not in output
    assert len(database.list_appointments(registered[0])) == 1
    assistant.ask.assert_not_called()


def test_patient_cannot_register_and_doctor_can_cancel_form(monkeypatch, clinic, capsys):
    database, doctor, reviewer, patient = clinic
    register = MagicMock(wraps=database.register_patient_by_doctor)
    monkeypatch.setattr(database, "register_patient_by_doctor", register)
    execute_chat(
        monkeypatch, MagicMock(),
        ["1", "/cadastrar-paciente", "/trocar", "2", "8", "SIM", "/cancelar", "/sair"],
        sqlite_demo=database, patient_demo=patient, doctor_demo=doctor, reviewer_demo=reviewer,
    )
    output = capsys.readouterr().out
    assert f"Identidade ativa: Paciente Sintético — {patient.user_id}" in output
    assert "Este fluxo está disponível para o perfil médico." in output
    assert "Operação cancelada." in output
    register.assert_not_called()
