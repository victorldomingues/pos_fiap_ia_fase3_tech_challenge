"""Carga de documentos-fonte (protocolos versionados em Markdown com front matter)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
REQUIRED_FIELDS = ("doc_id", "title", "version", "date")


class ProtocolFormatError(ValueError):
    """Documento sem metadados obrigatórios: sem proveniência não há citação confiável."""


def parse_protocol(text: str, source: str = "<memória>") -> Document:
    m = _FRONT_MATTER.match(text)
    if not m:
        raise ProtocolFormatError(f"{source}: front matter ausente (esperado bloco '---').")
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise ProtocolFormatError(f"{source}: campos obrigatórios ausentes: {', '.join(missing)}")
    body = m.group(2).strip()
    if not body:
        raise ProtocolFormatError(f"{source}: corpo vazio.")
    meta["source_file"] = source
    meta["content_hash"] = hashlib.sha256((meta["version"] + body).encode("utf-8")).hexdigest()
    return Document(page_content=body, metadata=meta)


def load_protocols(directory: Path) -> list[Document]:
    directory = Path(directory)
    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"Nenhum protocolo (*.md) encontrado em {directory}.")
    docs = [parse_protocol(f.read_text(encoding="utf-8"), f.name) for f in files]
    ids = [d.metadata["doc_id"] for d in docs]
    if len(ids) != len(set(ids)):
        raise ProtocolFormatError("doc_id duplicado entre protocolos.")
    return docs
