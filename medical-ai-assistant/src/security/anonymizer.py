"""Detecção e anonimização de PII/PHI brasileiros por regras (regex).

Limitação: regras não garantem anonimização completa (ex.: nomes sem pista contextual).
Para uso real, combine com NER validado e revisão humana.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordem importa: padrões mais específicos antes dos mais genéricos (CPF antes de telefone).
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CPF", re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")),
    ("CNS", re.compile(r"(?<!\d)[12789]\d{14}(?!\d)")),
    ("TELEFONE", re.compile(r"(?<!\d)(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}(?!\d)")),
    ("RG", re.compile(r"\bRG\s*:?\s*[\d.]{5,12}-?[\dXx]?\b", re.IGNORECASE)),
    ("PRONTUARIO", re.compile(r"\b(?:prontu[aá]rio|MRN)\s*(?:n[ºo°.]*)?\s*:?\s*#?\d{3,}\b", re.IGNORECASE)),
    ("CEP", re.compile(r"(?<!\d)\d{5}-\d{3}(?!\d)")),
    ("DATA_NASCIMENTO", re.compile(r"(?<!\d)\d{2}/\d{2}/\d{4}(?!\d)")),
)

# Nome após pista contextual (ex.: "meu nome é Maria Souza", "paciente João da Silva").
_NAME_CUE = re.compile(
    r"\b(?i:meu nome [eé]|me chamo|sou (?:o|a)|paciente|sr\.?|sra\.?|dr\.?|dra\.?)\s+"
    r"((?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\w'’-]+)(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\w'’-]+)*)",
)

# Subconjunto usado pelos guardrails (evita falso positivo com datas de versão de protocolos).
GUARDRAIL_LABELS = frozenset({"EMAIL", "CPF", "CNS", "TELEFONE", "RG", "PRONTUARIO", "NOME"})


@dataclass(frozen=True)
class PIIMatch:
    label: str
    start: int
    end: int


def find_pii(text: str, labels: frozenset[str] | None = None) -> list[PIIMatch]:
    """Retorna ocorrências de PII; `labels` restringe os tipos verificados."""
    matches: list[PIIMatch] = []
    taken: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < te and e > ts for ts, te in taken)

    for label, pattern in _PATTERNS:
        if labels is not None and label not in labels:
            continue
        for m in pattern.finditer(text):
            if not overlaps(m.start(), m.end()):
                matches.append(PIIMatch(label, m.start(), m.end()))
                taken.append((m.start(), m.end()))

    if labels is None or "NOME" in labels:
        for m in _NAME_CUE.finditer(text):
            s, e = m.span(1)
            if not overlaps(s, e):
                matches.append(PIIMatch("NOME", s, e))
                taken.append((s, e))

    return sorted(matches, key=lambda x: x.start)


def contains_pii(text: str, labels: frozenset[str] | None = GUARDRAIL_LABELS) -> bool:
    return bool(find_pii(text, labels))


def anonymize(text: str, labels: frozenset[str] | None = None) -> str:
    """Substitui PII por marcadores como [CPF] mantendo o restante do texto."""
    out: list[str] = []
    cursor = 0
    for m in find_pii(text, labels):
        out.append(text[cursor : m.start])
        out.append(f"[{m.label}]")
        cursor = m.end
    out.append(text[cursor:])
    return "".join(out)
