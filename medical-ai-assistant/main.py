"""Ponto de entrada do Assistente Médico Virtual (CLI)."""

from __future__ import annotations

import sys

from src.assistant.cli import parse_args, run_chat
from src.assistant.factory import build_assistant
from src.rag.indexer import IndexNotReadyError
from src.finetuning.artifacts import AdapterError
from src.utils.config import Settings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        assistant = build_assistant(Settings())
    except (IndexNotReadyError, AdapterError, FileNotFoundError) as exc:
        print(f"Falha na inicialização: {exc}", file=sys.stderr)
        return 2
    run_chat(assistant, user_id=args.user, reviewer=args.reviewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
