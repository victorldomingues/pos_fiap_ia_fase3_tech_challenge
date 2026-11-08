"""Guardrails de entrada e saída (regras determinísticas, conservadoras por desenho).

Bloqueiam: diagnósticos, prescrições, doses de medicamentos e informações pessoais.
Preferem falso positivo (bloquear demais) a falso negativo, dado o contexto clínico.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from src.security.anonymizer import contains_pii


class BlockCategory(str, Enum):
    DIAGNOSIS = "diagnostico"
    PRESCRIPTION = "prescricao"
    DOSAGE = "dose_medicamento"
    PII = "informacao_pessoal"
    EMPTY = "entrada_vazia"
    UNSAFE_CONTENT = "conteudo_inseguro"


REFUSAL_MESSAGES: dict[BlockCategory, str] = {
    BlockCategory.DIAGNOSIS: "Não posso realizar nem sugerir diagnósticos. Procure um profissional de saúde habilitado.",
    BlockCategory.PRESCRIPTION: "Não posso prescrever nem sugerir medicamentos ou tratamentos. Procure um profissional de saúde habilitado.",
    BlockCategory.DOSAGE: "Não posso informar a quantidade nem a frequência de uso de medicamentos. Consulte um médico ou farmacêutico.",
    BlockCategory.PII: "Detectei informações pessoais na mensagem. Por privacidade (LGPD), não as processo; reformule com os dados mascarados.",
    BlockCategory.EMPTY: "Por favor, digite uma pergunta.",
    BlockCategory.UNSAFE_CONTENT: "Não consegui gerar uma resposta segura para esta pergunta. Encaminhe-a a um profissional de saúde.",
}


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    category: BlockCategory | None = None
    matched: str | None = None

    @property
    def message(self) -> str:
        return REFUSAL_MESSAGES[self.category] if self.category else ""


def normalize(text: str) -> str:
    """Minúsculas e sem acentos, para casar variações ortográficas."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def is_likely_english(text: str) -> bool:
    """Heurística conservadora para impedir que frases claramente inglesas sejam exibidas."""
    normalized = normalize(text)
    english_count = sum(len(pattern.findall(normalized)) for pattern in _ENGLISH_MARKERS)
    portuguese_count = sum(len(pattern.findall(normalized)) for pattern in _PORTUGUESE_MARKERS)
    return (english_count >= 2 and english_count > portuguese_count) or (
        english_count > 0 and portuguese_count == 0
    )


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


_UNITS = r"(?:mg|mcg|ug|g|ml|ui|gotas?|comprimidos?|capsulas?|cp|mg/kg|mg/ml)"

# ---- entrada (texto normalizado) ----
_IN_DIAGNOSIS = _compile(
    r"\bdiagnose\b",
    r"\bdiagnostiqu\w*",
    r"\b(?:faca|forneca)\s+(?:um\s+)?diagnostico\b",
    r"\b(?:me\s+)?de\s+um\s+diagnostico\b",
    r"\bdiagnostico\s+(?:por favor|para mim|do meu caso)\b",
    r"\bqual\s+(?:e\s+)?(?:o\s+)?meu\s+(?:diagnostico|problema)",
    r"\bque\s+doenca\b",
    r"\bo\s+que\s+(?:eu\s+)?tenho\b",
    r"\bwhat\s+do\s+i\s+have\b",
    r"\bsera\s+que\s+(?:eu\s+)?(?:tenho|estou)\b",
    r"\b(?:tenho|estou\s+com)\b[^?.!]*\b(?:o\s+que\s+(?:pode\s+ser|e)|sera)\b",
)
_IN_PRESCRIPTION = _compile(
    r"\bprescrev\w*",
    r"\bprescribe\b",
    r"\breceit(?:e|ar|a)\b",
    r"\b(?:qual|que)\s+(?:remedio|medicamento|antibiotico|analgesico)\b",
    r"\b(?:posso|devo|preciso)\s+tomar\b",
    r"\bindi(?:que|ca)\s+(?:um\s+|algum\s+)?(?:remedio|medicamento)\b",
    r"\bo\s+que\s+(?:eu\s+)?(?:devo|posso)\s+tomar\b",
)
_IN_DOSAGE = _compile(
    r"\bdoses?\b",
    r"\bdosagem\b",
    r"\bposologia\b",
    r"\bquant\w+\s+(?:mg|miligramas|comprimidos?|gotas?|ml)\b",
    rf"\b\d+(?:[.,]\d+)?\s*{_UNITS}\b",
)
_RESEARCH_CONTEXT = _compile(r"\b(?:estudo|artigo|pesquisa|ensaio|dataset|base de dados)\b")
_RESEARCH_FINDING = _compile(
    r"\b(?:avaliou|analisou|relata|reporta|descreve|registrou|observou|utilizou|usou|testou|"
    r"encontrou|foi|foram)\b"
)
_PERSONAL_ADVICE = _compile(
    r"\b(?:devo|posso|preciso|meu|minha|para mim|para voce|recomende|prescrev|indique)\b"
)
# Instruções embutidas em documentos recuperados (prompt injection indireta).
_INJECTION = _compile(
    r"ignore\s+(?:all|any|the)?\s*(?:previous|prior|above)\s+instructions",
    r"ignore\s+(?:as\s+)?instrucoes\s+(?:anteriores|acima)",
    r"desconsidere\s+(?:as\s+)?(?:regras|instrucoes)",
    r"\bsystem\s+prompt\b",
    r"voce\s+agora\s+(?:e|deve)",
    r"revele\s+(?:o\s+)?prompt",
)

_ENGLISH_MARKERS = _compile(
    r"\b(?:the|and|is|are|was|were|with|for|from|this|that|these|those|which|who|their|have|has|"
    r"study|studies|found|findings|associated|reported|reports|showed|shows|describes|indicates|indicated|"
    r"participants|patients|symptoms|after|before|between|through|may|might|could|should|"
    r"according|increased|decreased|risk|disease|treatment|compared|laboratory|observation|"
    r"observations|research|results|condition|common|causes|using|used|improve|effects|evidence|"
    r"health|blood|glucose|adults|children)\b"
)
_PORTUGUESE_MARKERS = _compile(
    r"\b(?:o|a|os|as|um|uma|uns|umas|de|do|da|dos|das|no|na|nos|nas|em|por|para|com|sem|"
    r"que|se|ao|aos|pelo|pela|pelos|pelas|e|ou|mas|como|foi|foram|sao|esta|estao|tem|ha|"
    r"segundo|fonte|fontes|estudo|estudos|relata|descreve|paciente|pacientes|sintomas|"
    r"associados|apresenta|indica|observou|participantes)\b"
)

# ---- saída (texto normalizado) ----
_DRUGS = (
    "paracetamol|dipirona|ibuprofeno|amoxicilina|azitromicina|omeprazol|losartana|metformina|insulina|"
    "morfina|diclofenaco|dexametasona|prednisona|tramadol|codeina|rivotril|clonazepam|diazepam|"
    "heparina|varfarina|captopril|atenolol|sinvastatina|cefalexina|ciprofloxacino|aspirina|fluoxetina"
)
_KNOWN_DRUG = re.compile(rf"\b(?:{_DRUGS})\b")


def known_medications(text: str) -> set[str]:
    """Identifica termos conhecidos apenas para busca limitada, não para validar segurança."""
    return set(_KNOWN_DRUG.findall(normalize(text)))


_OUT_DOSAGE = _compile(
    rf"\b(?:tome|tomar|use|utilize|administre|ingerir)\b[^.!?]{{0,80}}\b\d+(?:[.,]\d+)?\s*{_UNITS}\b",
    rf"\b(?:take|use|administer)\b[^.!?]{{0,80}}\b\d+(?:[.,]\d+)?\s*{_UNITS}\b",
    r"\b(?:dose|dosagem|posologia)\b[^.!?]{0,60}\b(?:recomendada|indicada|ideal|correta|diaria|maxima|para voce|para mim)\b",
    r"\b(?:recommended|prescribed|indicated)\s+dose\b",
)
_OUT_DIAGNOSIS = _compile(
    r"\bvoce\s+(?:tem|esta\s+com|sofre\s+de|possui)\b",
    r"\bseu\s+diagnostico\b",
    r"\bdiagnostico\s+(?:provavel|confirmado|presuntivo)\b",
    r"\bdiagnostico\s+(?:e|seria)\s+(?:de\s+)?(?:um|uma|gripe|infeccao|cancer|diabetes)\b",
    r"\bprovavelmente\s+(?:voce\s+)?(?:e|tem|se\s+trata)\b",
    r"\bisso\s+(?:e|indica|significa\s+que\s+voce\s+tem)\s+(?:um|uma)?\s*(?:caso\s+de)\b",
    r"\byou\s+(?:have|suffer\s+from|are\s+diagnosed\s+with)\b",
)
_OUT_PRESCRIPTION = _compile(
    r"\b(?:recomendo|indico|prescrevo|prescreva|receito)\b",
    r"\b(?:recomenda-se|recomendar|recomendamos|indicamos|prescreve-se)\b",
    r"\b(?:voce\s+)?(?:deve|pode|precisa)\s+(?:tomar|usar|utilizar|administrar)\b",
    rf"\b(?:tome|tomar|use|utilize|administre)\s+(?:o\s+|a\s+|um\s+|uma\s+)?(?:medicamento|remedio|comprimido|antibiotico|analgesico|{_DRUGS})\b",
    r"\b(?:i\s+recommend|i\s+prescribe)\b",
    r"\b(?:you\s+should|you\s+can)\s+(?:take|use|administer)\b",
    rf"\b(?:take|use|administer)\s+(?:a\s+|an\s+|the\s+)?(?:medicine|medication|drug|pill|tablet|antibiotic|{_DRUGS})\b",
    r"\b(?:prescribe|recommend(?:s|ed)?)\s+(?:a\s+|an\s+|the\s+)?(?:medicine|medication|drug|pill|tablet|antibiotic|treatment|paracetamol|dipyrone)\b",
)


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(0)
    return None


def check_input(text: str) -> GuardrailResult:
    """Filtro de entrada: PII primeiro (evita que dados pessoais avancem), depois intenções vetadas."""
    if not text or not text.strip():
        return GuardrailResult(False, BlockCategory.EMPTY)
    if contains_pii(text):
        return GuardrailResult(False, BlockCategory.PII, "pii")
    norm = normalize(text)
    dataset_question = (
        _first_match(_RESEARCH_CONTEXT, norm) is not None
        and _first_match(_RESEARCH_FINDING, norm) is not None
        and _first_match(_PERSONAL_ADVICE, norm) is None
    )
    for category, patterns in (
        (BlockCategory.DOSAGE, _IN_DOSAGE),
        (BlockCategory.PRESCRIPTION, _IN_PRESCRIPTION),
        (BlockCategory.DIAGNOSIS, _IN_DIAGNOSIS),
    ):
        if dataset_question and category in (BlockCategory.DOSAGE, BlockCategory.PRESCRIPTION):
            continue
        hit = _first_match(patterns, norm)
        if hit:
            return GuardrailResult(False, category, hit)
    return GuardrailResult(True)


def check_output(text: str) -> GuardrailResult:
    """Filtro de saída: valida a resposta gerada antes de qualquer exibição."""
    if not text or not text.strip():
        return GuardrailResult(False, BlockCategory.UNSAFE_CONTENT, "resposta_vazia")
    if contains_pii(text):
        return GuardrailResult(False, BlockCategory.PII, "pii")
    norm = normalize(text)
    for category, patterns in (
        (BlockCategory.DOSAGE, _OUT_DOSAGE),
        (BlockCategory.PRESCRIPTION, _OUT_PRESCRIPTION),
        (BlockCategory.DIAGNOSIS, _OUT_DIAGNOSIS),
    ):
        hit = _first_match(patterns, norm)
        if hit:
            return GuardrailResult(False, category, hit)
    return GuardrailResult(True)


def is_suspicious_context(text: str) -> bool:
    """Conteúdo recuperado é dado não confiável: descarta trechos com instruções embutidas."""
    return _first_match(_INJECTION, normalize(text)) is not None
