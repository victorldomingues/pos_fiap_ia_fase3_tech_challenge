"""Explicabilidade verificável: fontes, confiança da recuperação, protocolo e estado de revisão.

Mostra apenas evidências (documentos e scores), não raciocínio interno do modelo.
"""

from __future__ import annotations

import re

from src.models.schemas import DISCLAIMER, HUMAN_VALIDATION_NOTICE, AssistantResponse, ReviewStatus

HIGH_CONFIDENCE = 0.6
MEDIUM_CONFIDENCE = 0.35


def confidence_label(score: float) -> str:
    if score >= HIGH_CONFIDENCE:
        return "alta"
    if score >= MEDIUM_CONFIDENCE:
        return "média"
    return "baixa"


def render_response(resp: AssistantResponse) -> str:
    lines = ["", f"Assistente: {resp.answer}", ""]
    exp = resp.explanation
    lines.append("─── Explicabilidade ───")
    lines.append(f"Protocolo utilizado: {exp.protocol_used or 'nenhum'}")
    lines.append(f"Confiança da recuperação: {exp.retrieval_confidence:.2f} ({confidence_label(exp.retrieval_confidence)})")
    if exp.sources:
        lines.append("Documentos recuperados:")
        for i, s in enumerate(exp.sources, start=1):
            snippet = " ".join(s.excerpt.split())[:160]
            kind = "protocolo" if s.source_type == "protocol" else f"dataset público não curado: {s.source_type}"
            lines.append(f"  [DOC-{i}] {s.doc_id} v{s.version} ({s.date}) — {s.title} | {kind} | score {s.score:.2f}")
            if resp.blocked:
                lines.append("      Conteúdo não exibido para esta solicitação.")
            elif s.source_type == "protocol":
                lines.append(f"      “{snippet}…”")
            else:
                lines.append("      Fonte pública não curada; conteúdo original não exibido na explicabilidade.")
    else:
        lines.append("Documentos utilizados: nenhum")
    if resp.blocked:
        lines.append(f"Bloqueio (guardrail): {resp.block_reason}")
    lines.append(f"ID da solicitação: {resp.request_id}")
    lines.append(f"Revisão humana: {resp.review_status.value.upper()}")
    if resp.requires_human_validation:
        lines.append(f"⚠ {HUMAN_VALIDATION_NOTICE}")
    elif resp.review_status == ReviewStatus.NOT_REQUIRED:
        lines.append("Nenhum conteúdo clínico foi gerado.")
    cited = dict.fromkeys(int(n) for n in re.findall(r"\[DOC-(\d+)\]", resp.answer))
    sources = [
        f"[DOC-{number}] {exp.sources[number - 1].doc_id} v{exp.sources[number - 1].version}"
        for number in cited if 1 <= number <= len(exp.sources)
    ]
    if sources:
        lines.append("Fontes citadas: " + "; ".join(sources))
    lines.append(DISCLAIMER)
    return "\n".join(lines)
