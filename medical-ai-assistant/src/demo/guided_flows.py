"""Fluxos guiados do chat de demonstração; persistência e autorização ficam no SQLite."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from src.assistant.assistant import MedicalAssistant
from src.database.sqlite import (
    APPOINTMENT_TYPES, AuthorizationError, ClinicDatabase, ConflictError, Principal, Role,
    validate_patient_field,
)
from src.rag.patient_retriever import PatientNoteRetriever
from src.rag.retriever import CombinedRetriever

Input = Callable[[str], str]
Output = Callable[[str], None]


class FlowCancelled(Exception):
    """O usuário cancelou uma operação guiada."""


def _ask(input_fn: Input, prompt: str) -> str:
    value = input_fn(prompt + " (/cancelar para voltar): ").strip()
    if value.lower() == "/cancelar":
        raise FlowCancelled
    return value


def _select(items: list[dict[str, str]], key: str, label: str, input_fn: Input, out: Output) -> str | None:
    if not items:
        out(f"Nenhum {label} disponível no seu escopo.")
        return None
    out(f"Selecione um {label} pelo número da lista; digite /cancelar para voltar.")
    for number, item in enumerate(items, 1):
        details = f" — {item['specialty']}" if "specialty" in item else ""
        name = f"{item['name']} — " if item.get("name") else ""
        out(f"{number}. {name}{item[key]}{details}")
        if "cpf" in item:
            out(f"   Nascimento: {item['birth_date']} | CPF: {item['cpf']} | "
                f"Telefone: {item['phone']} | E-mail: {item['email']}")
            out(f"   Observação administrativa: {item['administrative_note'] or '—'}")
    while True:
        choice = _ask(input_fn, f"Número do {label}")
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice) - 1][key]
        out("Seleção inválida. Escolha um número da lista ou digite /cancelar.")


def register_patient_flow(database: ClinicDatabase, principal: Principal, input_fn: Input, out: Output) -> Principal | None:
    """Cadastra paciente sintético e devolve a credencial somente à sessão local."""
    try:
        current = database.authenticate(principal.access_token)
        if current != principal or current.role != Role.DOCTOR:
            raise AuthorizationError("Credencial de profissional inválida.")
        out("Cadastro exclusivamente sintético. Não use dados reais nem conteúdo clínico.")
        if _ask(input_fn, "Confirma o uso exclusivo de dados sintéticos? Digite SIM") != "SIM":
            out("Cadastro cancelado.")
            return None
        fields = (
            ("full_name", "Nome completo fictício"),
            ("cpf", "CPF fictício (000.000.000-00)"),
            ("birth_date", "Nascimento fictício (DD/MM/AAAA)"),
            ("phone", "Telefone fictício com DDD"),
            ("email", "E-mail fictício"),
            ("administrative_note", "Observação administrativa fictícia, sem dados pessoais (opcional)"),
        )
        values = {}
        for field, prompt in fields:
            while True:
                value = _ask(input_fn, prompt)
                try:
                    validate_patient_field(field, value)
                except ValueError as exc:
                    out(str(exc))
                    continue
                values[field] = value
                break
        out(f"Cadastro de {values['full_name']}. CPF e contatos não são exibidos no resumo.")
        if _ask(input_fn, "Confirma o cadastro sintético? Digite SIM") != "SIM":
            out("Cadastro cancelado.")
            return None
        patient = database.register_patient_by_doctor(
            principal, values["full_name"], values["cpf"], values["birth_date"], values["phone"],
            values["email"], values["administrative_note"], synthetic_demo_confirmed=True,
        )
        out(f"Paciente cadastrado: {patient.user_id}. O cadastro não cria vínculo com o médico.")
        out("Use /trocar e escolha paciente para iniciar a sessão dele e agendar uma consulta.")
        out("A seleção do novo paciente está disponível apenas enquanto este chat estiver aberto.")
        return patient
    except FlowCancelled:
        out("Operação cancelada.")
    except AuthorizationError:
        out("Acesso negado: credencial de profissional inválida.")
    except ValueError as exc:
        logging.getLogger("medical_assistant.notebook").warning("Cadastro inválido: %s", type(exc).__name__)
        out(f"Não foi possível concluir o cadastro ({type(exc).__name__}).")
    except Exception as exc:
        logging.getLogger("medical_assistant.notebook").error("Falha no cadastro: %s", type(exc).__name__)
        out(f"Erro ao cadastrar paciente ({type(exc).__name__}). Consulte logs/app.log.")
    return None


def run_guided_flow(
    command: str,
    assistant: MedicalAssistant,
    database: ClinicDatabase,
    principal: Principal,
    input_fn: Input,
    out: Output,
) -> str | None:
    """Executa somente ações explícitas, confirmadas e verificadas pelo serviço."""
    try:
        if command == "/agendar":
            if principal.role != Role.PATIENT:
                out("Este fluxo está disponível para o perfil paciente.")
                return
            doctor_id = _select(database.list_doctors(principal, show_names=True), "doctor_id", "profissional", input_fn, out)
            if doctor_id is None:
                return
            while True:
                scheduled = _ask(input_fn, "Data e hora ISO com fuso (ex.: 2031-01-15T10:00:00-03:00)")
                try:
                    date = datetime.fromisoformat(scheduled)
                except ValueError:
                    out("Data inválida. Informe data e hora no formato indicado ou /cancelar.")
                    continue
                if date.tzinfo is None or date.utcoffset() is None:
                    out("Informe o fuso horário, por exemplo -03:00, ou /cancelar.")
                    continue
                break
            types = sorted(APPOINTMENT_TYPES)
            out("Selecione o tipo de consulta pelo número:")
            for number, appointment_type in enumerate(types, 1):
                out(f"{number}. {appointment_type}")
            while True:
                choice = _ask(input_fn, "Número do tipo de consulta")
                if choice.isdigit() and 1 <= int(choice) <= len(types):
                    break
                out("Tipo inválido. Escolha um número da lista ou digite /cancelar.")
            appointment_type = types[int(choice) - 1]
            out(f"Resumo: {doctor_id} | {scheduled} | {appointment_type}")
            if _ask(input_fn, "Digite SIM para confirmar o agendamento sintético") != "SIM":
                out("Agendamento cancelado.")
                return
            result = database.create_appointment(principal, doctor_id, scheduled, appointment_type)
            out(f"Agendamento criado: {result['appointment_id']} | {result['scheduled_at']}.")
            out("Próximos passos: consulte /agenda para revisar a consulta ou /menu para outros serviços.")
            return "appointment_created"

        if command in {"/anotar", "/proposta", "/revisar", "/auditoria"} and principal.role != Role.DOCTOR:
            out("Este fluxo está disponível para o perfil médico.")
            return
        if command == "/auditoria":
            events = database.read_audit_logs(principal)
            out(f"Auditoria funcional: {len(events)} evento(s) no escopo da sessão.")
            for event in events:
                out(f"{event['timestamp']} | {event['action']} | {event['outcome']}")
            return
        if command == "/revisar":
            proposals = database.list_prescriptions(principal, pending_only=True)
            proposal_id = _select(proposals, "prescription_id", "proposta pendente", input_fn, out)
            if proposal_id is None:
                return
            proposal = next(item for item in proposals if item["prescription_id"] == proposal_id)
            record = database.get_doctor_patient_record(principal, proposal["patient_id"])
            selected = next(item for item in record["prescriptions"] if item["prescription_id"] == proposal_id)
            if selected["author_doctor_id"] == principal.user_id:
                out("O autor não pode revisar a própria proposta. Troque para outro médico vinculado.")
                return
            out("Texto humano existente para revisão (não gerado nem aprovado pela IA):")
            out(selected["prescription_text"])
            decision = _ask(input_fn, "Decisão humana: APROVAR ou REJEITAR")
            if decision not in {"APROVAR", "REJEITAR"}:
                out("Decisão inválida; nenhuma revisão foi registrada.")
                return
            if _ask(input_fn, f"Digite SIM para confirmar {decision}") != "SIM":
                out("Revisão cancelada.")
                return
            status = database.review_prescription(principal, proposal_id, approve=decision == "APROVAR")
            out(f"Revisão humana registrada: {status}.")
            return

        patient_id = principal.user_id
        if principal.role == Role.DOCTOR:
            selected_id = _select(database.list_patients(principal, show_names=True, show_details=True), "patient_id", "paciente vinculado", input_fn, out)
            if selected_id is None:
                return
            patient_id = selected_id
        if command in {"/anotar", "/proposta"}:
            out("Digite somente conteúdo sintético escrito pelo profissional, sem geração ou complemento por IA.")
            text = _ask(input_fn, "Texto de autoria humana")
            if not text:
                out("O texto não pode ficar vazio. Execute o fluxo novamente.")
                return
            if _ask(input_fn, "Confirma autoria humana e conteúdo sintético? Digite SIM") != "SIM":
                out("Registro cancelado.")
                return
            if command == "/anotar":
                record_id = database.add_clinical_note(principal, patient_id, text, authored_by_human=True)
                out(f"Anotação humana registrada: {record_id}.")
            else:
                record_id = database.create_prescription_draft(principal, patient_id, text, authored_by_human=True)
                out(f"Proposta humana registrada: {record_id}; revisão por outro médico é obrigatória.")
            return
        if command == "/buscar":
            query = _ask(input_fn, "Busca nas anotações sintéticas, sem dados pessoais")
            retriever = assistant.retriever
            if isinstance(retriever, CombinedRetriever):
                retriever = retriever.protocol
            embeddings = retriever.index.store.embeddings
            if embeddings is None:
                raise RuntimeError("Embeddings não disponíveis para recuperação de anotações.")
            matches = PatientNoteRetriever(database, embeddings).retrieve(principal, patient_id, query)
            if not matches:
                out("Nenhuma anotação encontrada.")
            for number, match in enumerate(matches, 1):
                out(f"{number}. {match.note_id} | {match.created_at} | similaridade {match.score:.2f}\n{match.excerpt}")
            out("Recuperação extrativa; similaridade não comprova relevância. Validação humana quando aplicável.")
            out("Resultados apenas para consulta. Digite /buscar para pesquisar novamente ou /menu para outras opções.")
    except FlowCancelled:
        out("Operação cancelada.")
    except (EOFError, KeyboardInterrupt):
        raise
    except AuthorizationError:
        out("Acesso negado: credencial inválida ou paciente fora do escopo autorizado.")
    except ConflictError:
        out("Conflito: horário indisponível ou operação já realizada. Verifique os dados e tente novamente.")
    except ValueError as exc:
        logging.getLogger("medical_assistant.notebook").warning("Fluxo guiado inválido: %s", type(exc).__name__)
        out(f"Não foi possível concluir o fluxo ({type(exc).__name__}). Confira os dados e tente novamente.")
    except Exception as exc:
        logging.getLogger("medical_assistant.notebook").error("Falha no fluxo guiado: %s", type(exc).__name__)
        out(f"Erro ao executar o fluxo ({type(exc).__name__}). Consulte logs/app.log.")
