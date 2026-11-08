"""Recuperação temporária de anotações sintéticas, após autorização no SQLite."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.database.sqlite import ClinicDatabase, Principal
from src.security.anonymizer import contains_pii
from src.security.guardrails import check_input, is_suspicious_context


@dataclass(frozen=True)
class PatientNoteMatch:
    note_id: str
    created_at: str
    excerpt: str
    score: float


class PatientNoteRetriever:
    """Cria um índice Chroma em memória apenas com as notas autorizadas para a requisição."""

    def __init__(self, database: ClinicDatabase, embeddings: Embeddings) -> None:
        self.database = database
        self.embeddings = embeddings

    def retrieve(
        self,
        principal: Principal,
        patient_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[PatientNoteMatch]:
        if top_k < 1:
            raise ValueError("top_k precisa ser maior que zero.")
        guardrail = check_input(query)
        if not guardrail.allowed:
            raise ValueError(guardrail.message)
        # A autorização deve acontecer no SQLite antes que notas cheguem ao embedding.
        notes = self.database.get_clinical_notes(principal, patient_id)
        if any(contains_pii(note["content"], labels=None) for note in notes):
            raise ValueError("Anotação com identificador pessoal detectada; recuperação interrompida.")
        if any(is_suspicious_context(note["content"]) for note in notes):
            raise ValueError("Anotação com instrução suspeita detectada; recuperação interrompida.")
        if not notes:
            return []
        documents = [
            Document(
                page_content=note["content"],
                metadata={"note_id": note["note_id"], "created_at": note["created_at"]},
            )
            for note in notes
        ]
        store = Chroma(collection_name=f"patient-notes-{uuid.uuid4().hex}", embedding_function=self.embeddings)
        try:
            store.add_documents(documents)
            matches = store.similarity_search_with_score(query, k=min(top_k, len(documents)))
        finally:
            store.delete_collection()
        return [
            PatientNoteMatch(
                note_id=document.metadata["note_id"],
                created_at=document.metadata["created_at"],
                excerpt=document.page_content[:500],
                score=max(0.0, min(1.0, 1.0 - float(distance))),
            )
            for document, distance in matches
        ]
