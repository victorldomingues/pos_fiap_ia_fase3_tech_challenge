"""Indexação em ChromaDB com manifesto (modelo de embeddings + hash por documento).

- Valida compatibilidade ao iniciar; falha com instrução clara se o índice for incompatível.
- Reindexa apenas documentos novos/alterados (atualização incremental).
- O índice é derivável: os documentos-fonte em `data/protocols` são a fonte de verdade e backup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("medical_assistant.rag")

COLLECTION = "hospital_protocols"
MANIFEST_NAME = "index_manifest.json"
MANIFEST_VERSION = 1


class IndexNotReadyError(RuntimeError):
    """Índice ausente ou incompatível."""


@dataclass
class SyncReport:
    added: list[str]
    updated: list[str]
    removed: list[str]
    unchanged: list[str]


class ProtocolIndex:
    def __init__(
        self,
        index_dir: Path,
        embeddings: Embeddings,
        embedding_model_id: str,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model_id = embedding_model_id
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(self.index_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ---- manifesto ----
    @property
    def _manifest_path(self) -> Path:
        return self.index_dir / MANIFEST_NAME

    def _read_manifest(self) -> dict | None:
        if not self._manifest_path.exists():
            return None
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, doc_hashes: dict[str, str]) -> None:
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "embedding_model": self.embedding_model_id,
            "documents": doc_hashes,
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def validate(self) -> dict:
        """Exige modelo compatível e ao menos um trecho válido de cada documento no manifesto."""
        manifest = self._read_manifest()
        if manifest is None:
            raise IndexNotReadyError(
                f"Índice não encontrado em {self.index_dir}. Execute explicitamente: python scripts/build_index.py"
            )
        if manifest.get("embedding_model") != self.embedding_model_id:
            raise IndexNotReadyError(
                f"Índice criado com '{manifest.get('embedding_model')}', mas o configurado é "
                f"'{self.embedding_model_id}'. Reindexe com: python scripts/build_index.py --rebuild"
            )
        documents = manifest.get("documents")
        if not isinstance(documents, dict) or not documents:
            raise IndexNotReadyError(
                f"Manifesto sem documentos válidos em {self.index_dir}. "
                "Reconstrua o índice: python scripts/build_index.py --rebuild"
            )
        missing = set(documents)
        ids = list(documents)
        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            offset = 0
            while True:
                page = self.store.get(
                    where={"doc_id": {"$in": batch}}, include=["metadatas"], limit=1000, offset=offset
                )
                for metadata in page["metadatas"] or []:
                    if metadata:
                        doc_id = metadata.get("doc_id")
                        if isinstance(doc_id, str) and doc_id in documents and metadata.get("content_hash") == documents[doc_id]:
                            missing.discard(doc_id)
                if len(page["ids"]) < 1000:
                    break
                offset += len(page["ids"])
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            raise IndexNotReadyError(
                f"Índice inconsistente em {self.index_dir}: documento(s) sem trecho válido: {sample}. "
                "Reconstrua explicitamente: python scripts/build_index.py --rebuild "
                "(acrescente --with-public-sources se utilizar o índice público)."
            )
        return manifest

    # ---- indexação ----
    def _chunks(self, doc: Document) -> list[Document]:
        pieces = self._splitter.split_documents([doc])
        for i, piece in enumerate(pieces):
            piece.metadata["chunk"] = i
        return pieces

    def _delete_doc(self, doc_id: str) -> None:
        ids = self.store.get(where={"doc_id": doc_id}).get("ids", [])
        if ids:
            self.store.delete(ids=ids)

    def sync(self, documents: list[Document], rebuild: bool = False) -> SyncReport:
        """Sincroniza o índice com os documentos-fonte (ação explícita do operador)."""
        if rebuild:
            self.store.delete_collection()
            self.store = Chroma(
                collection_name=COLLECTION,
                embedding_function=self.store.embeddings,
                persist_directory=str(self.index_dir),
                collection_metadata={"hnsw:space": "cosine"},
            )
            previous: dict[str, str] = {}
        else:
            manifest = self._read_manifest()
            if manifest is not None:
                self.validate()
            previous = (manifest or {}).get("documents", {})

        current = {d.metadata["doc_id"]: d.metadata["content_hash"] for d in documents}
        report = SyncReport(added=[], updated=[], removed=[], unchanged=[])

        for doc_id in set(previous) - set(current):
            self._delete_doc(doc_id)
            report.removed.append(doc_id)

        for doc in documents:
            doc_id, digest = doc.metadata["doc_id"], doc.metadata["content_hash"]
            if previous.get(doc_id) == digest:
                report.unchanged.append(doc_id)
                continue
            if doc_id in previous:
                self._delete_doc(doc_id)
                report.updated.append(doc_id)
            else:
                report.added.append(doc_id)
            chunks = self._chunks(doc)
            ids = [f"{doc_id}::{doc.metadata['version']}::{c.metadata['chunk']}" for c in chunks]
            self.store.add_documents(chunks, ids=ids)

        self._write_manifest(current)
        logger.info("index sync added=%s updated=%s removed=%s", report.added, report.updated, report.removed)
        return report

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Retorna (documento, similaridade de cosseno em [0,1]); Chroma devolve distância = 1 - cos."""
        results = self.store.similarity_search_with_score(query, k=k)
        return [(doc, max(0.0, min(1.0, 1.0 - float(dist)))) for doc, dist in results]
