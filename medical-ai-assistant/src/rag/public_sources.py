"""Fontes públicas não curadas, separadas dos protocolos institucionais."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document


def load_public_sources(path: Path) -> list[Document]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus público ausente: {path}. Execute a preparação dos datasets antes de indexar.")
    docs: list[Document] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            source_id = row.get("source_id")
            question = row.get("question")
            answer = row.get("answer")
            context = row.get("context", "")
            if (source not in {"pubmedqa", "medquad"} or not isinstance(source_id, str)
                    or not source_id.startswith(source.upper() + "-")
                    or not isinstance(question, str) or not question.strip()
                    or not isinstance(answer, str) or not answer.strip()
                    or not isinstance(context, str)):
                raise ValueError(f"Fonte pública inválida em {path.name}, linha {line_number}.")
            if source_id in ids:
                raise ValueError(f"ID de fonte pública duplicado: {source_id}.")
            ids.add(source_id)
            content = f"Pergunta original: {question}\nResposta original: {answer}\nContexto: {context}"
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            docs.append(Document(
                page_content=content,
                metadata={
                    "doc_id": source_id,
                    "title": f"Dataset público {source.upper()} (não curado)",
                    "version": f"sha256-{digest[:16]}",
                    "date": "não informada",
                    "content_hash": digest,
                    "source_type": source,
                },
            ))
    if not docs:
        raise ValueError(f"Nenhuma fonte pública válida em {path}.")
    return docs
