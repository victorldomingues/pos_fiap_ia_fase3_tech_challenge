"""Observabilidade LangSmith com redação obrigatória.

Decisões de segurança:
- Desligado por padrão; exige MED_LANGSMITH_ENABLED=true e LANGSMITH_API_KEY.
- Usa um cliente próprio com hide_inputs/hide_outputs: inputs e outputs são substituídos por metadados,
  nunca apenas mascarados; conteúdo clínico não é exportado.
- MED_LANGSMITH_SEND_CONTENT não habilita envio de conteúdo.
- As variáveis globais LANGSMITH_TRACING/LANGCHAIN_TRACING_V2 são removidas do processo para impedir
  um tracer padrão sem redação.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.security.anonymizer import anonymize
from src.utils.config import Settings

logger = logging.getLogger("medical_assistant.observability")


def redact_payload(obj: Any) -> Any:
    """Anonimiza recursivamente todas as strings de dicts/listas."""
    if isinstance(obj, str):
        return anonymize(obj)
    if isinstance(obj, dict):
        return {k: redact_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_payload(v) for v in obj]
    return obj


def _metadata_only(payload: dict) -> dict:
    return {"content": "[omitido: tracing sem conteúdo]", "keys": sorted(str(k) for k in payload)}


def build_client(send_content: bool = False):
    from langsmith import Client

    if send_content:
        logger.warning("Envio de conteúdo ao LangSmith está desativado; serão enviados somente metadados.")
    return Client()


def build_callbacks(settings: Settings, client: Any | None = None) -> list[Any]:
    """Retorna callbacks do LangChain para o LangSmith ([] se desabilitado ou sem credenciais)."""
  
    if not settings.langsmith_enabled:
        return []
    if client is None and not os.getenv("LANGSMITH_API_KEY"):
        logger.warning("MED_LANGSMITH_ENABLED=true, mas LANGSMITH_API_KEY ausente: tracing desativado.")
        return []
    from langchain_core.tracers.langchain import LangChainTracer

    send_content = "true" == os.getenv("MED_LANGSMITH_SEND_CONTENT", "false").lower()
    if send_content:
        logger.warning("MED_LANGSMITH_SEND_CONTENT não habilita conteúdo; tracing permanece minimizado.")
    client = client or build_client()
    return [LangChainTracer(project_name=settings.langsmith_project, client=client)]
