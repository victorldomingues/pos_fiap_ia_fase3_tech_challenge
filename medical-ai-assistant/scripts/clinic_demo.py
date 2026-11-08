"""CLI demonstrativa de cadastro sintético, agenda, dados autorizados e revisão humana."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.factory import build_assistant  # noqa: E402
from src.assistant.explain import render_response  # noqa: E402
from src.database.sqlite import AuthorizationError, ClinicDatabase, ConflictError, Principal, Role  # noqa: E402
from src.models.schemas import DISCLAIMER  # noqa: E402
from src.rag.embeddings import get_embeddings  # noqa: E402
from src.rag.patient_retriever import PatientNoteRetriever  # noqa: E402
from src.utils.config import Settings  # noqa: E402


class _ScriptedInput:
    def __init__(self, values: list[str]) -> None:
        self._values = iter(values)
        self.events: list[dict[str, str]] = []

    def __call__(self, prompt: str) -> str:
        try:
            value = next(self._values)
        except StopIteration as exc:
            raise RuntimeError(f"Roteiro automatizado sem resposta para: {prompt}") from exc
        sensitive_prompt = re.search(r"cpf|telefone|e-mail|token|nome fictício|nascimento", prompt, re.I)
        self.events.append({
            "prompt": prompt,
            "value": "[REDACTED]" if sensitive_prompt else value,
        })
        return value

    def assert_exhausted(self) -> None:
        try:
            next(self._values)
        except StopIteration:
            return
        raise RuntimeError("Roteiro automatizado terminou antes de consumir todas as entradas.")


class _DeterministicDemoAssistant:
    def ask(self, question: str, *, user_id: str, user_role: str) -> str:
        return (
            "Resposta demonstrativa determinística baseada no fluxo informacional. "
            "Este conteúdo não é recomendação do assistente."
        )


class _DemoEmbeddings:
    """Embedding constante para executar o fluxo Chroma sem baixar um modelo."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _redact_output(text: str) -> str:
    text = re.sub(r"(Token de acesso[^:\n]*:\s*).+", r"\1[REDACTED]", text)
    text = re.sub(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", "[CPF REDIGIDO]", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[E-MAIL REDIGIDO]", text)
    return re.sub(r"\(\d{2}\)\s?\d{4,5}-\d{4}", "[TELEFONE REDIGIDO]", text)


def run_automated_tests() -> int:
    """Percorre os fluxos da CLI em SQLite temporário e imprime transcript redigido."""
    from src.demo.persistent_setup import setup_persistent_demo

    completed_scenarios: list[str] = []

    with tempfile.TemporaryDirectory(prefix="clinic-demo-") as temp_dir:
        profiles = setup_persistent_demo(
            Path(temp_dir) / "clinic.sqlite3",
            "automated-clinic-demo-secret-2026",
        )
        database = profiles.database
        if Path(database.path).resolve().parent != Path(temp_dir).resolve():
            raise RuntimeError("A demonstração automatizada precisa usar somente um SQLite temporário.")

        original_build_assistant = globals()["build_assistant"]
        original_render_response = globals()["render_response"]
        original_get_embeddings = globals()["get_embeddings"]
        globals()["build_assistant"] = lambda settings: _DeterministicDemoAssistant()
        globals()["render_response"] = lambda response: response
        globals()["get_embeddings"] = lambda model_id: _DemoEmbeddings()

        def execute(
            name: str,
            values: list[str],
            *,
            expected_status: int = 0,
            expected_output: tuple[str, ...] = (),
            forbidden_output: tuple[str, ...] = (),
        ) -> None:
            scripted = _ScriptedInput(values)
            captured = io.StringIO()
            error: str | None = None
            status = -1
            try:
                with contextlib.redirect_stdout(captured):
                    status = run(database, scripted)
                scripted.assert_exhausted()
            except Exception as exc:
                error = f"{type(exc).__name__}: {_redact_output(str(exc))}"
            output = _redact_output(captured.getvalue())
            if status != expected_status:
                error = error or f"Status {status}, esperado {expected_status}."
            for expected in expected_output:
                if expected not in output:
                    error = error or f"Saída esperada ausente: {expected}"
            for forbidden in forbidden_output:
                if forbidden in output:
                    error = error or f"Saída indevida encontrada: {forbidden}"
            print(f"\n{'=' * 78}")
            print(f"Cenário {len(completed_scenarios) + 1}: {name} — {'FALHOU' if error else 'APROVADO'}")
            print("Entradas do cenário:")
            for event in scripted.events:
                print(f"  {event['prompt']}{event['value']}")
            print("Saída do cenário:")
            print(output.rstrip() or "(sem saída)")
            if error:
                print(f"Erro: {error}")
            if error:
                raise AssertionError(f"Cenário {name} falhou: {error}")
            completed_scenarios.append(name)

        try:
            execute(
                "paciente_fluxos_1_a_4",
                [
                    "paciente", "", profiles.patient.access_token,
                    "1", "2", profiles.doctor.user_id, "2035-02-01T10:00:00-03:00",
                    "consulta_geral", "3", "4", "Qual é o horário de visitas?", "0",
                ],
                expected_output=("SYN-A-DEMO", "Resposta demonstrativa determinística"),
            )
            execute(
                "paciente_isolamento_da_agenda",
                ["paciente", "", profiles.additional_patients[0].access_token, "3", "0"],
                expected_output=("SYN-A-DEMO-02",),
                forbidden_output=("SYN-A-DEMO\"",),
            )
            execute(
                "medico_fluxos_1_a_9",
                [
                    "medico", profiles.doctor.access_token,
                    "1", "2", "3", profiles.patient.user_id,
                    "4", "Qual é o horário de visitas?",
                    "5", "SIM", profiles.patient.user_id, "Nota sintética de retorno para demonstração.",
                    "6", "SIM", profiles.patient.user_id,
                    "Proposta fictícia escrita pelo profissional para revisão humana.",
                    "8", "9", profiles.patient.user_id, "retorno", "0",
                ],
                expected_output=("SYN-P-DEMO-02", "SYN-N-DEMO", "Resposta demonstrativa determinística"),
            )
            execute(
                "medico_revisor_aprova_validacao",
                [
                    "médico", profiles.reviewer.access_token,
                    "7", "SYN-R-DEMO", "SIM", "Revisão humana sintética.", "0",
                ],
                expected_output=("approved",),
            )
            execute(
                "cadastro_de_paciente_sintetico",
                [
                    "paciente", "cadastrar", "SIM", "Paciente Sintético Autoteste",
                    "333.666.999-50", "10/05/1990", "(31) 99999-0000",
                    "autoteste@example.test", "0",
                ],
                expected_output=("Cadastro concluído.", "[REDACTED]"),
            )
            execute(
                "perfil_incompativel_com_credencial",
                ["paciente", "", profiles.doctor.access_token],
                expected_status=2,
                expected_output=("não corresponde ao perfil selecionado",),
            )
        finally:
            globals()["build_assistant"] = original_build_assistant
            globals()["render_response"] = original_render_response
            globals()["get_embeddings"] = original_get_embeddings

    print(f"Testes automatizados da CLI: {len(completed_scenarios)} cenários aprovados.")
    print("Cobertura dos menus: paciente 1–4; médico 1–9; cadastro e falha de autenticação.")
    print("Dados: somente sintéticos; SQLite temporário; LLM/embeddings externos não utilizados.")
    print("Histórico completo exibido somente neste output; nenhum arquivo de log é gravado.")
    return 0


def _register_patient(database: ClinicDatabase, input_fn: Callable[[str], str]) -> Principal:
    print("Demonstração somente com dados fictícios; não informe dados reais de pacientes.")
    confirmation = input_fn("Digite SIM para confirmar o uso de dados sintéticos: ").strip()
    return database.register_patient(
        full_name=input_fn("Nome fictício: "),
        cpf=input_fn("CPF fictício: "),
        birth_date=input_fn("Nascimento fictício (AAAA-MM-DD): "),
        phone=input_fn("Telefone fictício: "),
        email=input_fn("E-mail fictício: "),
        synthetic_demo_confirmed=confirmation == "SIM",
    )


def _authenticate(database: ClinicDatabase, input_fn: Callable[[str], str]) -> Principal | None:
    selected_profile = input_fn("Perfil (paciente/médico): ").strip().lower()
    if selected_profile not in {"paciente", "medico", "médico"}:
        print("Perfil não reconhecido.")
        return None

    if selected_profile == "paciente":
        action = input_fn("Digite 'cadastrar' para novo paciente fictício ou Enter para entrar: ").strip().lower()
        if action == "cadastrar":
            principal = _register_patient(database, input_fn)
            print(f"Cadastro concluído. Identificador sintético: {principal.user_id}")
            print(f"Token de acesso (guarde-o; será mostrado somente agora): {principal.access_token}")
            return principal

    token = input_fn("Token de acesso: ").strip()
    principal = database.authenticate(token)
    expected_role = Role.PATIENT if selected_profile == "paciente" else Role.DOCTOR
    if principal.role != expected_role:
        raise AuthorizationError("O token autenticado não corresponde ao perfil selecionado.")
    print(f"Perfil confirmado pelo cadastro: {principal.role.value}.")
    return principal


def _ask_assistant(principal: Principal, input_fn: Callable[[str], str]) -> None:
    question = input_fn("Pergunta geral (não inclua dados pessoais): ").strip()
    assistant = build_assistant(Settings())
    response = assistant.ask(question, user_id=principal.user_id, user_role=principal.role.value)
    print(render_response(response))


def _patient_menu(database: ClinicDatabase, principal: Principal, input_fn: Callable[[str], str]) -> None:
    print(DISCLAIMER)
    while True:
        print("\nPaciente: 1=profissionais 2=agendar 3=consultas 4=pergunta geral 0=sair")
        action = input_fn("Opção: ").strip()
        if action == "0":
            return
        if action == "1":
            print(json.dumps(database.list_doctors(principal), ensure_ascii=False, indent=2))
        elif action == "2":
            result = database.create_appointment(
                principal,
                doctor_id=input_fn("Identificador do profissional: ").strip(),
                scheduled_at=input_fn("Data/hora ISO-8601 com fuso (ex.: 2030-01-15T10:00:00-03:00): ").strip(),
                appointment_type=input_fn("Tipo de consulta: ").strip(),
            )
            print(json.dumps(result, ensure_ascii=False))
        elif action == "3":
            print(json.dumps(database.list_appointments(principal), ensure_ascii=False, indent=2))
        elif action == "4":
            _ask_assistant(principal, input_fn)
        else:
            print("Opção desconhecida.")


def _doctor_menu(database: ClinicDatabase, principal: Principal, input_fn: Callable[[str], str]) -> None:
    while True:
        print("\nMédico: 1=agenda 2=pacientes vinculados 3=consultar paciente")
        print("4=pergunta sobre protocolos 5=anotação fictícia 6=proposta própria")
        print("7=revisar proposta 8=auditoria 9=RAG de anotações 0=sair")
        action = input_fn("Opção: ").strip()
        if action == "0":
            return
        if action == "1":
            result = database.list_appointments(principal)
        elif action == "2":
            result = database.list_patients(principal)
        elif action == "3":
            result = database.get_patient(principal, input_fn("Identificador sintético do paciente: ").strip())
        elif action == "4":
            _ask_assistant(principal, input_fn)
            continue
        elif action == "5":
            if input_fn("A anotação é fictícia e foi escrita pelo profissional? (SIM): ").strip() != "SIM":
                print("Operação cancelada: somente anotação fictícia de autoria humana pode ser registrada.")
                continue
            result = {
                "note_id": database.add_clinical_note(
                    principal,
                    input_fn("Identificador sintético do paciente: ").strip(),
                    input_fn("Anotação clínica fictícia, sem dados pessoais: ").strip(),
                    authored_by_human=True,
                )
            }
        elif action == "6":
            if input_fn("O texto foi escrito pelo profissional, sem geração/complemento por IA? (SIM): ").strip() != "SIM":
                print("Operação cancelada: somente texto de autoria humana pode ser registrado.")
                continue
            result = {
                "prescription_id": database.create_prescription_draft(
                    principal,
                    input_fn("Identificador sintético do paciente: ").strip(),
                    input_fn("Texto da proposta fictícia do profissional: ").strip(),
                    authored_by_human=True,
                )
            }
        elif action == "7":
            pending = database.list_prescriptions(principal, pending_only=True)
            print(json.dumps(pending, ensure_ascii=False, indent=2))
            prescription_id = input_fn("Identificador a revisar: ").strip()
            decision = input_fn("Aprovar? (SIM aprova; qualquer outro valor rejeita): ").strip()
            result = {
                "prescription_id": prescription_id,
                "status": database.review_prescription(
                    principal,
                    prescription_id,
                    approve=decision == "SIM",
                    note=input_fn("Nota administrativa sem dados pessoais: ").strip(),
                ),
            }
        elif action == "8":
            result = database.read_audit_logs(principal)
        elif action == "9":
            patient_id = input_fn("Identificador sintético do paciente: ").strip()
            query = input_fn("Consulta de recuperação (não peça diagnóstico ou tratamento): ").strip()
            matches = PatientNoteRetriever(
                database,
                get_embeddings(Settings().embedding_model_id),
            ).retrieve(principal, patient_id, query)
            result = [
                {
                    "note_id": match.note_id,
                    "created_at": match.created_at,
                    "excerpt": match.excerpt,
                    "score": match.score,
                }
                for match in matches
            ]
        else:
            print("Opção desconhecida.")
            continue
        print(json.dumps(result, ensure_ascii=False, indent=2))


def provision_doctor(settings: Settings, input_fn: Callable[[str], str] = input) -> int:
    """Provisionamento de demonstração exige segredo de bootstrap fornecido fora do repositório."""
    database = ClinicDatabase(settings.database_path, settings.database_bootstrap_token)
    if input_fn("O profissional é fictício e este fluxo não será usado com dados reais? (SIM): ").strip() != "SIM":
        print("Provisionamento cancelado: somente perfis sintéticos são aceitos.")
        return 2
    principal = database.provision_doctor(
        doctor_id=input_fn("ID sintético do profissional: ").strip(),
        full_name=input_fn("Nome fictício: ").strip(),
        specialty=input_fn("Especialidade fictícia: ").strip(),
        bootstrap_token=input_fn("Token administrativo de bootstrap: ").strip(),
        synthetic_demo_confirmed=True,
    )
    print(f"Profissional cadastrado: {principal.user_id}")
    print(f"Token de acesso (guarde-o; será mostrado somente agora): {principal.access_token}")
    return 0


def run(database: ClinicDatabase, input_fn: Callable[[str], str] = input) -> int:
    try:
        principal = _authenticate(database, input_fn)
        if principal is None:
            return 2
        if principal.role == Role.PATIENT:
            _patient_menu(database, principal, input_fn)
        else:
            _doctor_menu(database, principal, input_fn)
    except (AuthorizationError, ConflictError, ValueError) as exc:
        print(f"Operação não concluída: {exc}")
        return 2
    return 0


def main(
    argv: list[str] | None = None,
    *,
    database: ClinicDatabase | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    parser = argparse.ArgumentParser(description="Fluxos administrativos fictícios em SQLite")
    parser.add_argument("--provision-doctor", action="store_true", help="provisiona médico sintético com token de bootstrap")
    parser.add_argument(
        "--automated-tests",
        action="store_true",
        help="executa automaticamente os fluxos da CLI em SQLite temporário",
    )
    args = parser.parse_args(argv)
    if args.automated_tests:
        if args.provision_doctor or database is not None:
            parser.error("--automated-tests não pode ser combinado com provisionamento ou banco injetado")
        return run_automated_tests()
    settings = Settings()
    database = database or ClinicDatabase(settings.database_path, settings.database_bootstrap_token)
    if args.provision_doctor:
        try:
            return provision_doctor(settings, input_fn)
        except (AuthorizationError, ValueError) as exc:
            print(f"Provisionamento não concluído: {exc}", file=sys.stderr)
            return 2
    return run(database, input_fn)


if __name__ == "__main__":
    raise SystemExit(main())
