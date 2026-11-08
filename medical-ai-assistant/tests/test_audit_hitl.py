import json

import pytest

from src.assistant.hitl import ApprovalError, ApprovalQueue
from src.logging.audit import AuditLogger
from src.models.schemas import ReviewStatus


def test_audit_chain_verifies_and_persists_across_instances(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLogger(path)
    log.record("interaction", request_id="a", user_id="u1", question="q")
    AuditLogger(path).record("interaction", request_id="b", user_id="u1", question="q2")
    entries = AuditLogger(path).read()
    assert len(entries) == 2 and entries[1]["prev_hash"] == entries[0]["hash"]
    assert {"timestamp", "user_id", "question"} <= set(entries[0])
    assert AuditLogger(path).verify()


def test_audit_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLogger(path)
    log.record("interaction", request_id="a", answer="original")
    log.record("interaction", request_id="b", answer="x")
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["answer"] = "adulterado"
    path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n", encoding="utf-8")
    assert not AuditLogger(path).verify()


def test_approval_lifecycle(tmp_path):
    q = ApprovalQueue(tmp_path / "approvals.jsonl")
    q.submit(
        "r1",
        "u1",
        question="Pergunta sintética",
        answer="Resposta citada [DOC-1]",
        sources=[{"doc_id": "PROT-001", "version": "1.0", "excerpt": "Trecho sintético"}],
    )
    assert q.status("r1") == ReviewStatus.PENDING and q.pending() == ["r1"]
    record = q.pending_records()[0]
    assert record.question == "Pergunta sintética"
    assert record.answer == "Resposta citada [DOC-1]"
    assert record.sources == [{"doc_id": "PROT-001", "version": "1.0", "excerpt": "Trecho sintético"}]
    assert q.decide("r1", True, "dra-exemplo", "ok") == ReviewStatus.APPROVED
    assert q.pending() == [] and q.verify()
    reopened = ApprovalQueue(tmp_path / "approvals.jsonl")
    assert reopened._state()["r1"].reviewer == "dra-exemplo"


def test_approval_requires_reviewer_and_blocks_redecision(tmp_path):
    q = ApprovalQueue(tmp_path / "approvals.jsonl")
    q.submit("r1", "u1")
    with pytest.raises(ApprovalError):
        q.decide("r1", True, "  ")
    q.decide("r1", False, "rev", "fonte desatualizada")
    with pytest.raises(ApprovalError):
        q.decide("r1", True, "rev")
    with pytest.raises(ApprovalError):
        q.status("desconhecida")
