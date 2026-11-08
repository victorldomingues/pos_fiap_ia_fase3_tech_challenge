"""Contratos de dados validados (pydantic) usados entre os componentes."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

HUMAN_VALIDATION_NOTICE = "Validação humana obrigatória."
DISCLAIMER = (
    "Este conteúdo é informativo e não constitui recomendação do agente, diagnóstico, prescrição "
    "ou decisão clínica. Procure um médico habilitado e especializado no assunto para orientação."
)


class ReviewStatus(str, Enum):
    PENDING = "pendente"
    APPROVED = "aprovada"
    REJECTED = "rejeitada"
    NOT_REQUIRED = "nao_aplicavel"


class SourceDocument(BaseModel):
    """Trecho recuperado, com a proveniência necessária para citação."""

    doc_id: str
    title: str
    version: str = "n/d"
    date: str = "n/d"
    source_type: str = "protocol"
    excerpt: str
    score: float = Field(ge=0.0, le=1.0)


class Explanation(BaseModel):
    sources: list[SourceDocument] = Field(default_factory=list)
    retrieval_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    protocol_used: str | None = None


class AssistantResponse(BaseModel):
    request_id: str
    user_id: str
    question: str
    answer: str
    blocked: bool = False
    block_reason: str | None = None
    clinical: bool = True
    requires_human_validation: bool = True
    review_status: ReviewStatus = ReviewStatus.PENDING
    explanation: Explanation = Field(default_factory=Explanation)
    created_at: str


class Patient(BaseModel):
    """Paciente SINTÉTICO (nunca use dados reais)."""

    patient_id: str
    name: str
    cpf: str
    birth_date: str
    phone: str
    email: str
    notes: str = ""
