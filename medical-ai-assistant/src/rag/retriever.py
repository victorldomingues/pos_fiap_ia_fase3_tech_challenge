"""Recuperação com score, filtro de conteúdo suspeito e confiança agregada."""

from __future__ import annotations

from src.models.schemas import Explanation, SourceDocument
from src.rag.indexer import ProtocolIndex
from src.security.guardrails import check_output, is_suspicious_context

EXCERPT_MAX_CHARS = 500


class ProtocolRetriever:
    def __init__(self, index: ProtocolIndex, top_k: int = 3, min_confidence: float = 0.35) -> None:
        self.index = index
        self.top_k = top_k
        self.min_confidence = min_confidence

    def retrieve(self, query: str) -> list[SourceDocument]:
        """Trechos ordenados por similaridade; descarta trechos com instruções embutidas."""
        hits = self.index.search(query, k=self.top_k)
        docs: list[SourceDocument] = []
        for doc, score in hits:
            if is_suspicious_context(doc.page_content):
                continue
            meta = doc.metadata
            source_type = meta.get("source_type", "protocol")
            if source_type != "protocol" and not check_output(doc.page_content).allowed:
                continue
            docs.append(
                SourceDocument(
                    doc_id=meta.get("doc_id", "desconhecido"),
                    title=meta.get("title", "sem título"),
                    version=meta.get("version", "n/d"),
                    date=meta.get("date", "n/d"),
                    source_type=source_type,
                    excerpt=doc.page_content[:EXCERPT_MAX_CHARS],
                    score=round(score, 4),
                )
            )
        return sorted(docs, key=lambda d: d.score, reverse=True)

    def explain(self, docs: list[SourceDocument]) -> Explanation:
        """Confiança = maior similaridade entre os trechos; protocolo = documento do melhor trecho."""
        if not docs:
            return Explanation()
        best = docs[0]
        return Explanation(
            sources=docs,
            retrieval_confidence=best.score,
            protocol_used=f"{best.doc_id} v{best.version} — {best.title}" if best.source_type == "protocol" else None,
        )

    def is_reliable(self, explanation: Explanation) -> bool:
        return bool(explanation.sources) and explanation.retrieval_confidence >= self.min_confidence


class CombinedRetriever(ProtocolRetriever):
    """Prioriza protocolos acima do limiar; usa fontes públicas como alternativa."""

    def __init__(self, protocol: ProtocolRetriever, public: ProtocolRetriever) -> None:
        self.protocol = protocol
        self.public = public
        self.top_k = protocol.top_k
        self.min_confidence = protocol.min_confidence

    def retrieve(self, query: str) -> list[SourceDocument]:
        protocols = self.protocol.retrieve(query)
        eligible = [doc for doc in protocols if doc.score >= self.min_confidence]
        if not eligible:
            eligible = [doc for doc in self.public.retrieve(query) if doc.score >= self.min_confidence]
        seen: set[tuple[str, str]] = set()
        unique = []
        for doc in sorted(eligible, key=lambda doc: doc.score, reverse=True):
            key = (doc.source_type, doc.doc_id)
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique[:self.top_k]
