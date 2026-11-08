"""Chat via terminal (CLI): Usuário → LangChain → RAG → LLM → Resposta (+ HITL)."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from src.assistant.assistant import MedicalAssistant
from src.assistant.explain import render_response
from src.assistant.hitl import ApprovalError
from src.models.schemas import HUMAN_VALIDATION_NOTICE

HELP = """Comandos:
  /pendentes                 lista respostas aguardando validação humana
  /aprovar <id> [nota]       aprova uma resposta (requer --reviewer)
  /rejeitar <id> <motivo>    rejeita uma resposta (requer --reviewer)
  /auditoria                 verifica a integridade dos logs de auditoria
  /ajuda                     mostra esta ajuda
  /sair                      encerra"""

BANNER = (
    "Assistente Médico Virtual — apoio informacional (sem diagnóstico, prescrição ou doses).\n"
    f"{HUMAN_VALIDATION_NOTICE}  Digite /ajuda para comandos."
)


def _handle_command(line: str, assistant: MedicalAssistant, reviewer: str | None, out: Callable[[str], None]) -> bool:
    """Retorna False para encerrar."""
    parts = line.split(maxsplit=2)
    cmd = parts[0].lower()
    if cmd in {"/sair", "/exit", "/quit"}:
        return False
    if cmd == "/ajuda":
        out(HELP)
    elif cmd == "/pendentes":
        pending = assistant.approvals.pending()
        out("Pendentes: " + (", ".join(pending) if pending else "nenhuma"))
    elif cmd == "/auditoria":
        ok = assistant.audit.verify() and assistant.approvals.verify()
        out("Integridade dos logs: " + ("OK" if ok else "FALHA — possível adulteração"))
    elif cmd in {"/aprovar", "/rejeitar"}:
        if len(parts) < 2:
            out(f"Uso: {cmd} <id> [nota]")
        elif not reviewer:
            out("Informe o revisor ao iniciar: python main.py --reviewer <id-do-revisor>")
        else:
            note = parts[2] if len(parts) > 2 else ""
            try:
                status = assistant.approvals.decide(parts[1], cmd == "/aprovar", reviewer, note)
                assistant.audit.record("review", request_id=parts[1], reviewer=reviewer, decision=status.value)
                out(f"Solicitação {parts[1]}: {status.value}")
            except ApprovalError as exc:
                out(f"Erro: {exc}")
    else:
        out("Comando desconhecido. Use /ajuda.")
    return True


def run_chat(
    assistant: MedicalAssistant,
    user_id: str,
    reviewer: str | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    output_fn(BANNER)
    while True:
        try:
            line = input_fn("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nEncerrando.")
            return
        if line.startswith("/"):
            if not _handle_command(line, assistant, reviewer, output_fn):
                return
            continue
        try:
            response = assistant.ask(line, user_id=user_id)
        except Exception as exc:  # erro visível ao usuário; detalhes ficam no log local
            output_fn(f"Erro ao processar a pergunta ({type(exc).__name__}). Consulte logs/app.log.")
            continue
        output_fn(render_response(response))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assistente Médico Virtual (CLI)")
    p.add_argument("--user", default="anon-cli", help="identificador não sensível do usuário (para auditoria)")
    p.add_argument("--reviewer", default=None, help="identificador do profissional que valida as respostas")
    return p.parse_args(argv)
