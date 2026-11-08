"""Montagem das dependências do assistente a partir de Settings."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.assistant.assistant import MedicalAssistant
from src.assistant.hitl import ApprovalQueue
from src.logging.audit import AuditLogger, setup_app_logging
from src.models.llm import load_chat_model
from src.rag.embeddings import get_embeddings
from src.rag.indexer import IndexNotReadyError, ProtocolIndex
from src.rag.retriever import CombinedRetriever, ProtocolRetriever
from src.utils.config import Settings
from src.utils.observability import build_callbacks


def build_index(settings: Settings, embeddings: Embeddings | None = None) -> ProtocolIndex:
    embeddings = embeddings or get_embeddings(settings.embedding_model_id)
    return ProtocolIndex(
        settings.index_dir, embeddings, settings.embedding_model_id, settings.chunk_size, settings.chunk_overlap
    )


def build_public_index(settings: Settings, embeddings: Embeddings | None = None) -> ProtocolIndex:
    embeddings = embeddings or get_embeddings(settings.embedding_model_id)
    return ProtocolIndex(
        settings.public_index_dir, embeddings, settings.embedding_model_id, settings.chunk_size, settings.chunk_overlap
    )


def build_assistant(
    settings: Settings,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
) -> MedicalAssistant:
    """Valida o índice persistido (falha com instruções se ausente/incompatível) e monta a cadeia."""
    setup_app_logging(settings.app_log_path)
    index = build_index(settings, embeddings)
    index.validate()
    retriever = ProtocolRetriever(index, settings.top_k, settings.min_confidence)
    if settings.public_rag_enabled:
        public_index = build_public_index(settings, embeddings or index.store.embeddings)
        try:
            public_index.validate()
        except IndexNotReadyError as exc:
            raise IndexNotReadyError(
                f"Índice público indisponível ou incompatível: {exc} "
                "Execute python scripts/build_index.py --with-public-sources"
                " (acrescente --rebuild se o modelo de embeddings mudou)."
            ) from exc
        retriever = CombinedRetriever(retriever, ProtocolRetriever(public_index, settings.top_k, settings.min_confidence))
    llm = llm or load_chat_model(settings)
    return MedicalAssistant(
        retriever=retriever,
        llm=llm,
        audit=AuditLogger(settings.audit_log_path),
        approvals=ApprovalQueue(settings.approvals_path),
        callbacks=build_callbacks(settings),
        model_label=f"{settings.llm_backend}:{settings.base_model_id}",
    )
