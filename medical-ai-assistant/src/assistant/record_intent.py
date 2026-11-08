"""Encaminhamento conservador para consulta explícita de registros sintéticos."""

import re

from src.security.guardrails import check_input, normalize

_RECORD_TERMS = re.compile(
    r"\b(?:prontuarios?|anotac(?:ao|oes)|registros?\s+(?:clinicos?|medicos?|administrativos?)"
    r"|historico\s+(?:clinico|medico))\b"
)


def is_record_request(question: str) -> bool:
    """Reconhece termos comuns; não autentica nem consulta dados."""
    return check_input(question).allowed and bool(_RECORD_TERMS.search(normalize(question)))
