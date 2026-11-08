"""Adaptadores LangChain somente leitura, vinculados a uma identidade autenticada.

Operações mutáveis (cadastro, agendamento e revisão) são executadas pelo fluxo explícito
da aplicação, nunca por decisão autônoma do LLM.
"""

from __future__ import annotations

import json
from langchain_core.tools import BaseTool, tool

from src.database.sqlite import ClinicDatabase, Principal
from src.rag.patient_retriever import PatientNoteRetriever
from src.rag.retriever import ProtocolRetriever
from src.security.guardrails import check_input

NO_SOURCE_MESSAGE = (
    "Não encontrei fonte confiável nos protocolos disponíveis. "
    "Consulte um profissional de saúde habilitado."
)


def build_clinic_tools(
    database: ClinicDatabase,
    principal: Principal,
    retriever: ProtocolRetriever,
    patient_retriever: PatientNoteRetriever | None = None,
) -> list[BaseTool]:
    """Cria ferramentas com escopo de perfil; não as conecta automaticamente ao LLM."""

    def validate_query(query: str) -> None:
        result = check_input(query)
        if not result.allowed:
            raise ValueError(result.message)

    @tool
    def appointment_tool(action: str = "list") -> str:
        """Lista a agenda autenticada ou o diretório de profissionais para solicitar agendamento."""
        if action == "doctors":
            result = database.list_doctors(principal)
        elif action == "list":
            result = database.list_appointments(principal)
        else:
            raise ValueError("A ação da ferramenta deve ser 'list' ou 'doctors'.")
        return json.dumps(result, ensure_ascii=False)

    @tool
    def patient_tool(patient_id: str) -> str:
        """Consulta dados do próprio paciente ou de paciente autorizado para o médico."""
        return json.dumps(database.get_patient(principal, patient_id), ensure_ascii=False)

    @tool
    def protocol_tool(query: str) -> str:
        """Busca protocolos institucionais e retorna fontes recuperadas com versão."""
        validate_query(query)
        documents = retriever.retrieve(query)
        return json.dumps(
            [
                {
                    "doc_id": item.doc_id,
                    "title": item.title,
                    "version": item.version,
                    "date": item.date,
                    "excerpt": item.excerpt,
                }
                for item in documents
                if item.source_type == "protocol"
            ],
            ensure_ascii=False,
        )

    @tool
    def rag_tool(query: str, patient_id: str | None = None) -> str:
        """Recupera protocolos ou anotações sintéticas após autorizar o paciente no SQLite."""
        validate_query(query)
        if patient_id:
            if patient_retriever is None:
                raise RuntimeError("Recuperação de anotações pessoais não foi configurada.")
            notes = patient_retriever.retrieve(principal, patient_id, query)
            return json.dumps(
                [
                    {
                        "note_id": note.note_id,
                        "created_at": note.created_at,
                        "excerpt": note.excerpt,
                        "score": note.score,
                    }
                    for note in notes
                ],
                ensure_ascii=False,
            )
        documents = retriever.retrieve(query)
        explanation = retriever.explain(documents)
        if not retriever.is_reliable(explanation):
            return NO_SOURCE_MESSAGE
        return json.dumps(
            {
                "confidence": explanation.retrieval_confidence,
                "sources": [
                    {
                        "doc_id": item.doc_id,
                        "version": item.version,
                        "date": item.date,
                        "source_type": item.source_type,
                        "excerpt": item.excerpt if item.source_type == "protocol" else "[fonte pública não curada]",
                    }
                    for item in explanation.sources
                ],
            },
            ensure_ascii=False,
        )

    @tool
    def prescription_tool(pending_only: bool = True) -> str:
        """Lista apenas metadados de propostas no escopo médico; nunca gera nem aprova prescrição."""
        return json.dumps(
            database.list_prescriptions(principal, pending_only=pending_only),
            ensure_ascii=False,
        )

    return [appointment_tool, patient_tool, protocol_tool, prescription_tool, rag_tool]


def tool_names(tools: list[BaseTool]) -> set[str]:
    """Exposição mínima para testes e documentação da caixa de ferramentas."""
    return {item.name for item in tools}
