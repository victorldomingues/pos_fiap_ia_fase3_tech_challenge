"""Localização limitada de referências, sem reproduzir conteúdo sobre medicação."""

from __future__ import annotations

import re

from src.models.schemas import SourceDocument
from src.rag.retriever import CombinedRetriever, ProtocolRetriever
from src.security.guardrails import is_suspicious_context, normalize

_SAFE_METADATA = re.compile(r"^[A-Za-z0-9._/-]{1,120}$")


def find_medication_sources(retriever: ProtocolRetriever, term: str) -> list[SourceDocument]:
    """Localiza menções literais em trechos recuperados; não valida a pertinência clínica."""
    retrievers = [retriever.protocol, retriever.public] if isinstance(retriever, CombinedRetriever) else [retriever]
    matches = []
    seen = set()
    pattern = re.compile(rf"\b{re.escape(term)}\b")
    for candidate in retrievers:
        for doc, score in candidate.index.search(term, k=candidate.top_k):
            meta = doc.metadata
            doc_id = meta.get("doc_id")
            source_type = meta.get("source_type", "protocol")
            if (
                score < candidate.min_confidence
                or not isinstance(doc_id, str)
                or not _SAFE_METADATA.fullmatch(doc_id)
                or source_type not in {"protocol", "medquad", "pubmedqa"}
                or (source_type, doc_id) in seen
                or is_suspicious_context(doc.page_content)
                or not pattern.search(normalize(doc.page_content))
            ):
                continue
            seen.add((source_type, doc_id))
            version = meta.get("version")
            date = meta.get("date")
            matches.append(SourceDocument(
                doc_id=doc_id,
                title="Referência recuperada (conteúdo não reproduzido)",
                version=version if isinstance(version, str) and _SAFE_METADATA.fullmatch(version) else "n/d",
                date=date if isinstance(date, str) and _SAFE_METADATA.fullmatch(date) else "n/d",
                source_type=source_type,
                excerpt="",
                score=round(score, 4),
            ))
    return matches[:retriever.top_k]
