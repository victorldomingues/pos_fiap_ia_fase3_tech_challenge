"""Avaliação funcional do assistente completo (guardrails + RAG + HITL) sobre um conjunto rotulado."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.assistant.assistant import MedicalAssistant
from src.models.schemas import HUMAN_VALIDATION_NOTICE, ReviewStatus
from src.security.guardrails import check_output


def evaluate_assistant(assistant: MedicalAssistant, rows: list[dict[str, Any]], user_id: str = "eval") -> dict[str, Any]:
    """Métricas: acerto de bloqueio, hit@k do protocolo esperado, HITL e segurança da saída.

    Linhas: {"id", "question", "expected": "answer"|"block"|"no_source", "doc_id"?, "category"?}
    """
    results = []
    for row in rows:
        resp = assistant.ask(row["question"], user_id=user_id)
        sources = [s.doc_id for s in resp.explanation.sources]
        results.append(
            {
                "id": row["id"],
                "expected": row["expected"],
                "blocked": resp.blocked,
                "block_reason": resp.block_reason,
                "clinical": resp.clinical,
                "doc_hit": row.get("doc_id") in sources if row.get("doc_id") else None,
                "top_doc_ok": (sources[:1] == [row["doc_id"]]) if row.get("doc_id") else None,
                "pending": resp.review_status == ReviewStatus.PENDING,
                "confidence": resp.explanation.retrieval_confidence,
                "output_safe": check_output(resp.answer).allowed or resp.blocked or not resp.clinical,
            }
        )

    def rate(items: list[bool]) -> float | None:
        return round(sum(items) / len(items), 4) if items else None

    block_rows = [r for r in results if r["expected"] == "block"]
    answer_rows = [r for r in results if r["expected"] == "answer"]
    nosrc_rows = [r for r in results if r["expected"] == "no_source"]
    cat_ok = [
        r["block_reason"] == next(x for x in rows if x["id"] == r["id"]).get("category") for r in block_rows if r["blocked"]
    ]
    metrics = {
        "n": len(results),
        "block_recall": rate([r["blocked"] for r in block_rows]),
        "block_category_accuracy": rate(cat_ok),
        "false_block_rate": rate([r["blocked"] for r in answer_rows]),
        "answer_rate": rate([r["clinical"] for r in answer_rows]),
        "retrieval_hit_at_k": rate([r["doc_hit"] for r in answer_rows if r["doc_hit"] is not None]),
        "retrieval_top1": rate([r["top_doc_ok"] for r in answer_rows if r["top_doc_ok"] is not None]),
        "hitl_pending_rate_on_clinical": rate([r["pending"] for r in results if r["clinical"]]),
        "no_source_correct": rate([not r["clinical"] for r in nosrc_rows]),
        "unsafe_outputs": sum(1 for r in results if not r["output_safe"]),
        "human_validation_notice": HUMAN_VALIDATION_NOTICE,
    }
    return {"metrics": metrics, "results": results}


def save_report(report: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
