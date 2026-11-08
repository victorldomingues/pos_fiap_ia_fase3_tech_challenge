"""Human-in-the-Loop: fila persistida de revisão com hash encadeado.

Toda resposta clínica nasce PENDENTE. Um revisor identificado aprova ou rejeita; as decisões são
auditáveis. A pergunta anonimizada, resposta e trechos de fonte são retidos para possibilitar
revisão; a aprovação não transforma conteúdo gerado em recomendação clínica.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.logging.audit import AuditLogger
from src.models.schemas import ReviewStatus


class ApprovalError(ValueError):
    pass


@dataclass
class ApprovalRecord:
    request_id: str
    status: ReviewStatus
    user_id: str | None = None
    question: str | None = None
    answer: str | None = None
    sources: list[dict[str, Any]] | None = None
    reviewer: str | None = None
    note: str | None = None


class ApprovalQueue:
    def __init__(self, path: Path) -> None:
        self._log = AuditLogger(path)

    def submit(
        self,
        request_id: str,
        user_id: str,
        *,
        question: str | None = None,
        answer: str | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        self._log.record(
            "approval_requested",
            request_id=request_id,
            user_id=user_id,
            question=question,
            answer=answer,
            sources=sources or [],
        )

    def _state(self) -> dict[str, ApprovalRecord]:
        state: dict[str, ApprovalRecord] = {}
        for e in self._log.read():
            rid = e.get("request_id")
            if e["event"] == "approval_requested":
                state[rid] = ApprovalRecord(
                    request_id=rid,
                    status=ReviewStatus.PENDING,
                    user_id=e.get("user_id"),
                    question=e.get("question"),
                    answer=e.get("answer"),
                    sources=e.get("sources", []),
                )
            elif e["event"] == "approval_decided" and rid in state:
                record = state[rid]
                state[rid] = ApprovalRecord(
                    request_id=record.request_id,
                    status=ReviewStatus(e["decision"]),
                    user_id=record.user_id,
                    question=record.question,
                    answer=record.answer,
                    sources=record.sources,
                    reviewer=e.get("reviewer"),
                    note=e.get("note"),
                )
        return state

    def status(self, request_id: str) -> ReviewStatus:
        rec = self._state().get(request_id)
        if rec is None:
            raise ApprovalError(f"Solicitação desconhecida: {request_id}")
        return rec.status

    def pending(self) -> list[str]:
        return [rid for rid, r in self._state().items() if r.status == ReviewStatus.PENDING]

    def pending_records(self) -> list[ApprovalRecord]:
        return [record for record in self._state().values() if record.status == ReviewStatus.PENDING]

    def decide(self, request_id: str, approve: bool, reviewer: str, note: str = "") -> ReviewStatus:
        if not reviewer or not reviewer.strip():
            raise ApprovalError("Identificação do revisor é obrigatória.")
        current = self.status(request_id)
        if current != ReviewStatus.PENDING:
            raise ApprovalError(f"Solicitação {request_id} já decidida: {current.value}.")
        decision = ReviewStatus.APPROVED if approve else ReviewStatus.REJECTED
        self._log.record(
            "approval_decided",
            request_id=request_id,
            decision=decision.value,
            reviewer=reviewer.strip(),
            note_provided=bool(note.strip()),
        )
        return decision

    def verify(self) -> bool:
        return self._log.verify()
