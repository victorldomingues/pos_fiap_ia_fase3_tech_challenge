"""Chat textual para o notebook, usando as identidades sintéticas da demonstração."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from src.assistant.assistant import MedicalAssistant
from src.assistant.explain import render_response
from src.assistant.hitl import ApprovalError
from src.assistant.record_intent import is_record_request
from src.database.sqlite import AuthorizationError, ClinicDatabase, Principal, Role
from src.demo.guided_flows import register_patient_flow, run_guided_flow
from src.security.guardrails import normalize


_LABELS = {
    'patient': 'Paciente', 'patient_id': 'Paciente', 'doctor_id': 'Profissional',
    'patient_name': 'Nome do paciente', 'doctor_name': 'Nome do profissional',
    'appointment_id': 'Agendamento', 'scheduled_at': 'Data e hora', 'appointment_type': 'Tipo',
    'status': 'Estado', 'name': 'Nome', 'specialty': 'Especialidade', 'cpf': 'CPF',
    'phone': 'Telefone', 'email': 'E-mail', 'birth_date': 'Nascimento',
    'administrative_note': 'Observação administrativa',
    'notes': 'Anotações', 'note_id': 'Anotação', 'content': 'Conteúdo',
    'prescriptions': 'Propostas humanas', 'prescription_id': 'Proposta',
    'prescription_text': 'Texto humano', 'author_doctor_id': 'Autor',
    'reviewer_doctor_id': 'Revisor', 'appointments': 'Agendamentos',
    'created_at': 'Criado em', 'reviewed_at': 'Revisado em',
}
_DISPLAY_VALUES = {
    'scheduled': 'Agendada', 'completed': 'Concluída', 'cancelled': 'Cancelada',
    'consulta_geral': 'Consulta geral', 'retorno_ambulatorial': 'Retorno ambulatorial',
}


def _format_records(value: Any, indent: int = 0) -> str:
    prefix = ' ' * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = _LABELS.get(key, key)
            if isinstance(item, (dict, list)):
                lines.append(f'{prefix}{label}:\n{_format_records(item, indent + 2)}')
            else:
                display = _DISPLAY_VALUES.get(item, item) if isinstance(item, str) else item
                lines.append(f'{prefix}{label}: {display if display not in (None, "") else "—"}')
        return '\n'.join(lines)
    if isinstance(value, list):
        return '\n'.join(
            f'{prefix}{number}.\n{_format_records(item, indent + 2)}'
            for number, item in enumerate(value, 1)
        ) if value else f'{prefix}Nenhum registro.'
    return f'{prefix}{value}'


def run_notebook_text_chat(
    assistant: MedicalAssistant,
    sqlite_demo: ClinicDatabase,
    patient_demo: Principal,
    doctor_demo: Principal,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    reviewer_demo: Principal | None = None,
    additional_patient_demos: tuple[Principal, ...] = (),
) -> None:
    principal = None
    last_flow = None
    demo_patients = [patient_demo, *additional_patient_demos]
    output_fn('══ Assistente Médico Virtual | Demonstração acadêmica ══')
    output_fn('Chat informacional — use somente dados sintéticos; as sessões são demonstrações locais.')
    output_fn('Faça perguntas gerais ou escolha um fluxo. A IA não diagnostica, prescreve ou decide condutas.')

    def show_help(role: Role) -> None:
        output_fn('\n── Menu de fluxos ──')
        output_fn('Fluxos comuns: perguntas sobre protocolos; /agenda; /registros; /trocar; /ajuda; /sair.')
        output_fn('1 /agenda  |  2 /registros  |  3 /buscar (RAG de anotações)  |  0 /sair')
        if role == Role.PATIENT:
            output_fn('Paciente: /profissionais lista o diretório; /agenda e /registros mostram somente os próprios dados.')
            output_fn('4 /profissionais  |  5 /agendar (solicitação sintética guiada)')
        else:
            output_fn('Médico: /agenda mostra pacientes agendados com você; /registros consulta apenas pacientes vinculados.')
            output_fn('4 /anotar  |  5 /proposta  |  6 /revisar  |  7 /auditoria')
            output_fn('8 /cadastrar-paciente (cadastro sintético)')
            output_fn('Validação do assistente: /validacoes lista pendências; /aprovar-validacao <id> registra a revisão.')
            output_fn('Anotações e propostas são humanas; a revisão exige outro médico vinculado.')
        output_fn('Use /menu ou /ajuda para voltar a este guia; /cancelar interrompe um fluxo guiado.')
        output_fn('No chat, números escolhem opções do menu; em uma seleção, escolhem um item da lista.')
        output_fn('Exemplo de pergunta: Qual é o horário de visitas? Não inclua dados pessoais.')
        output_fn('Treinamento, avaliação e indexação são executados nas etapas próprias do notebook.')
    while True:
        try:
            prompt = ('Perfil [1 paciente, 2 médico, 3 revisor, /sair]: ' if reviewer_demo
                      else 'Perfil [1 paciente, 2 médico, /sair]: ') if principal is None else 'Opção do menu, comando ou pergunta: '
            line = input_fn(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            output_fn('\nChat encerrado.')
            return
        if not line:
            output_fn('Escolha um perfil ou digite uma pergunta, conforme a sessão.')
            continue
        command = line.lower()
        if command == '/sair':
            output_fn('Chat encerrado.')
            return
        if command in ('/logout', '/trocar'):
            principal = None
            last_flow = None
            output_fn('Sessão encerrada. Escolha um perfil sintético.')
            continue
        if principal is None:
            if command not in ('1', '2', '/paciente', '/medico') and not (
                reviewer_demo is not None and command in ('3', '/revisor')
            ):
                output_fn('Perfil inválido. Escolha 1 para paciente ou 2 para médico.')
                continue
            is_patient = command in ('1', '/paciente')
            identity = patient_demo if is_patient else reviewer_demo if command in ('3', '/revisor') else doctor_demo
            assert identity is not None
            try:
                if is_patient and len(demo_patients) > 1:
                    output_fn('── Escolha o paciente sintético desta demonstração ──')
                    for number, candidate in enumerate(demo_patients, 1):
                        details = sqlite_demo.get_session_identity(candidate)
                        output_fn(f"{number}. {details['name']} — {details['user_id']}")
                    while True:
                        choice = input_fn('Número do paciente (0 cancela): ').strip()
                        if choice in ('0', '/cancelar'):
                            output_fn('Seleção cancelada.')
                            break
                        if choice.isdigit() and 1 <= int(choice) <= len(demo_patients):
                            identity = demo_patients[int(choice) - 1]
                            break
                        output_fn('Seleção inválida. Escolha um número da lista ou 0 para cancelar.')
                    if choice in ('0', '/cancelar'):
                        continue
                principal = sqlite_demo.authenticate(identity.access_token)
                expected_role = Role.PATIENT if is_patient else Role.DOCTOR
                if principal.role != expected_role:
                    principal = None
                    output_fn('A credencial não corresponde ao perfil escolhido.')
                    continue
                output_fn('Sessão sintética autenticada: ' + ('paciente' if principal.role == Role.PATIENT else 'médico'))
                active = sqlite_demo.get_session_identity(principal)
                output_fn(f"Identidade ativa: {active['name']} — {active['user_id']}")
                show_help(principal.role)
            except AuthorizationError:
                principal = None
                output_fn('Credencial sintética inválida.')
            except (EOFError, KeyboardInterrupt):
                output_fn('\nChat encerrado.')
                return
            continue
        if command in ('/ajuda', '/menu'):
            show_help(principal.role)
            continue
        if command in ('/validacoes', '/aprovar-validacao') or command.startswith('/aprovar-validacao '):
            if principal.role != Role.DOCTOR:
                output_fn('As validações humanas do assistente estão disponíveis somente para o perfil médico.')
                continue
            try:
                current = sqlite_demo.authenticate(principal.access_token)
                if current.user_id != principal.user_id or current.role != Role.DOCTOR:
                    raise AuthorizationError('A identidade da sessão mudou.')
                if command == '/validacoes':
                    records = assistant.approvals.pending_records()
                    output_fn('── Validações humanas obrigatórias pendentes ──')
                    if not records:
                        output_fn('Nenhuma validação pendente.')
                    for record in records:
                        output_fn(f'ID: {record.request_id} | Estado: PENDENTE | Usuário: {record.user_id or "não informado"}')
                        if record.question is None or record.answer is None:
                            output_fn('Conteúdo indisponível para revisão; não aprove esta solicitação.')
                            continue
                        output_fn(f'Pergunta: {record.question}')
                        output_fn(f'Resposta do assistente: {record.answer}')
                        for source in record.sources or []:
                            output_fn(
                                f"Fonte: {source.get('doc_id', 'n/d')} v{source.get('version', 'n/d')} "
                                f"— {source.get('title', 'sem título')}"
                            )
                            if source.get('excerpt'):
                                output_fn(f"Trecho recuperado: {' '.join(source['excerpt'].split())[:500]}")
                    output_fn('A aprovação registra uma revisão humana demonstrativa; não torna a resposta recomendação clínica.')
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        output_fn('Uso: /aprovar-validacao <id-da-solicitacao>')
                        continue
                    request_id = parts[1].strip()
                    record = next(
                        (item for item in assistant.approvals.pending_records() if item.request_id == request_id),
                        None,
                    )
                    if record is None:
                        output_fn('Solicitação inexistente ou sem estado pendente.')
                        continue
                    if record.question is None or record.answer is None:
                        output_fn('Conteúdo necessário à revisão indisponível; a solicitação não foi aprovada.')
                        continue
                    status = assistant.approvals.decide(
                        request_id,
                        approve=True,
                        reviewer=current.user_id,
                        note='Revisão humana registrada no chat da demonstração.',
                    )
                    assistant.audit.record(
                        'review',
                        request_id=request_id,
                        reviewer=current.user_id,
                        decision=status.value,
                    )
                    output_fn(
                        f'Solicitação {request_id}: {status.value}. '
                        'A decisão foi auditada e não valida clinicamente o conteúdo.'
                    )
            except AuthorizationError:
                principal = None
                output_fn('Credencial inválida. Escolha novamente o perfil.')
            except ApprovalError as exc:
                output_fn(f'Não foi possível registrar a revisão: {exc}')
            except Exception as exc:
                logging.getLogger('medical_assistant.notebook').error(
                    'Falha ao consultar ou registrar validação HITL: %s',
                    type(exc).__name__,
                )
                output_fn(f'Erro ao processar validação humana ({type(exc).__name__}).')
            continue
        menu = {'0': '/sair', '1': '/agenda', '2': '/registros', '3': '/buscar'}
        menu.update({'4': '/profissionais', '5': '/agendar'} if principal.role == Role.PATIENT else
                    {'4': '/anotar', '5': '/proposta', '6': '/revisar', '7': '/auditoria', '8': '/cadastrar-paciente'})
        if command.isdigit() and command not in menu:
            output_fn('Opção do menu inválida. Digite /menu para ver as opções disponíveis.')
            continue
        command = menu.get(command, command)
        if command == '/sair':
            output_fn('Chat encerrado.')
            return
        if command == '/cadastrar-paciente':
            if principal.role != Role.DOCTOR:
                output_fn('Este fluxo está disponível para o perfil médico.')
                continue
            try:
                new_patient = register_patient_flow(sqlite_demo, principal, input_fn, output_fn)
                if new_patient is not None:
                    demo_patients.append(new_patient)
                    last_flow = None
            except (EOFError, KeyboardInterrupt):
                output_fn('\nChat encerrado.')
                return
            continue
        if command in ('/agendar', '/anotar', '/proposta', '/revisar', '/buscar', '/auditoria'):
            try:
                last_flow = run_guided_flow(command, assistant, sqlite_demo, principal, input_fn, output_fn)
            except (EOFError, KeyboardInterrupt):
                output_fn('\nChat encerrado.')
                return
            continue
        if command in ('/agenda', '/profissionais'):
            if command == '/profissionais' and principal.role != Role.PATIENT:
                output_fn('Este fluxo está disponível para o perfil paciente.')
                continue
            try:
                records = sqlite_demo.list_appointments(principal, show_names=True, show_patient_details=True) if command == '/agenda' else sqlite_demo.list_doctors(principal, show_names=True)
                output_fn('── Agenda ──' if command == '/agenda' else '── Profissionais disponíveis ──')
                output_fn(_format_records(records) if records else 'Nenhum registro disponível.')
                output_fn('Lista apenas para consulta: os números identificam itens, não os selecionam.')
                if command == '/profissionais':
                    output_fn('Para escolher um profissional e agendar, digite /agendar. Para outras opções, digite /menu.')
                elif principal.role == Role.PATIENT:
                    output_fn('Para agendar uma consulta, digite /agendar. Para outras opções, digite /menu.')
                else:
                    output_fn('Para consultar um paciente vinculado, digite /registros. Para outras opções, digite /menu.')
            except AuthorizationError:
                principal = None
                output_fn('Credencial inválida. Escolha novamente o perfil.')
            continue
        if command == '/registros':
            try:
                if principal.role == Role.DOCTOR:
                    patients = sqlite_demo.list_patients(principal, show_names=True, show_details=True)
                    if not patients:
                        output_fn('Nenhum paciente autorizado para este profissional.')
                        output_fn('Digite /agenda para consultar os agendamentos ou /menu para outras opções.')
                        continue
                    output_fn('── Selecione um paciente autorizado ──')
                    for number, patient in enumerate(patients, 1):
                        output_fn(f"{number}.\n{_format_records(patient, 2)}")
                    choice = input_fn('Número do paciente (0 cancela): ').strip()
                    if choice == '0':
                        output_fn('Consulta cancelada.')
                        continue
                    if not choice.isdigit() or not 1 <= int(choice) <= len(patients):
                        output_fn('Seleção inválida.')
                        continue
                    records = sqlite_demo.get_doctor_patient_record(principal, patients[int(choice) - 1]['patient_id'], show_name=True, show_details=True)
                else:
                    records = {
                        'patient': sqlite_demo.get_patient(principal, principal.user_id, show_name=True, show_details=True),
                        'notes': sqlite_demo.get_clinical_notes(principal, principal.user_id),
                    }
                output_fn('Registros sintéticos disponíveis; não representam um prontuário completo.')
                output_fn(_format_records(records))
                output_fn('Lista apenas para consulta: os números identificam itens, não os selecionam.')
                output_fn('Para buscar nas anotações, digite /buscar; para outras opções, digite /menu.')
            except AuthorizationError:
                output_fn('Acesso negado: credencial inválida ou paciente fora do escopo autorizado.')
            except (EOFError, KeyboardInterrupt):
                output_fn('\nChat encerrado.')
                return
            except Exception as exc:
                logging.getLogger('medical_assistant.notebook').error('Falha ao consultar registros: %s', type(exc).__name__)
                output_fn(f'Erro ao consultar registros ({type(exc).__name__}).')
            continue
        if line.startswith('/'):
            output_fn('Comando desconhecido. Digite /ajuda.')
            continue
        if is_record_request(line):
            output_fn('Para consultar registros sintéticos autorizados, use /registros.')
            continue
        try:
            current = sqlite_demo.authenticate(principal.access_token)
            if current.user_id != principal.user_id or current.role != principal.role:
                raise AuthorizationError('A identidade da sessão mudou.')
            normalized = normalize(line).strip(' ?.!')
            if normalized in {'o que eu faco agora', 'o que faco agora', 'e agora', 'nao entendi'}:
                if last_flow == "appointment_created":
                    output_fn('Seu agendamento foi registrado. Use /agenda para conferir os detalhes ou /menu para ver os próximos serviços.')
                else:
                    output_fn('Posso ajudar com o menu e os serviços. Use /menu ou explique qual etapa não ficou clara.')
                continue
            if re.fullmatch(r'[\d\s:TtZz+()./-]+', line):
                output_fn('Isto parece uma data ou número, não uma pergunta. Para agendar, escolha /agendar e siga as etapas; use /menu para ajuda.')
                continue
            response = assistant.ask(line, user_id=current.user_id, user_role=current.role.value)
            output_fn(render_response(response))
            last_flow = None
        except AuthorizationError:
            principal = None
            output_fn('Credencial inválida. Escolha novamente o perfil.')
        except Exception as exc:
            output_fn(f'Erro ao processar a pergunta ({type(exc).__name__}). Consulte logs/app.log.')
