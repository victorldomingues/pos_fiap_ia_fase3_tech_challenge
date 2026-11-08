"""Fixtures compartilhadas. Embeddings falsos e determinísticos evitam download de modelos nos testes."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.assistant.assistant import MedicalAssistant
from src.assistant.hitl import ApprovalQueue
from src.database.sqlite import ClinicDatabase
from src.logging.audit import AuditLogger
from src.models.llm import ExtractiveChatModel
from src.rag.indexer import ProtocolIndex
from src.rag.loader import load_protocols
from src.rag.retriever import ProtocolRetriever
from src.security.guardrails import normalize

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS_DIR = ROOT / "data" / "protocols"
STOPWORDS = {"para", "como", "qual", "quais", "que", "uma", "dos", "das", "com", "por", "sobre", "deve", "pode"}


class HashEmbeddings(Embeddings):
    """Bag-of-words com hashing (dim fixa, L2-normalizado): similaridade lexical determinística."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"\w{3,}", normalize(text)):
            if tok in STOPWORDS:
                continue
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture()
def embeddings() -> HashEmbeddings:
    return HashEmbeddings()


@pytest.fixture()
def index(tmp_path, embeddings) -> ProtocolIndex:
    idx = ProtocolIndex(tmp_path / "chroma", embeddings, "hash-embeddings-test", 600, 80)
    idx.sync(load_protocols(PROTOCOLS_DIR))
    return idx


@pytest.fixture()
def retriever(index) -> ProtocolRetriever:
    return ProtocolRetriever(index, top_k=3, min_confidence=0.15)


@pytest.fixture()
def make_assistant(tmp_path, retriever):
    def _make(llm=None) -> MedicalAssistant:
        return MedicalAssistant(
            retriever=retriever,
            llm=llm or ExtractiveChatModel(),
            audit=AuditLogger(tmp_path / "logs" / "audit.jsonl"),
            approvals=ApprovalQueue(tmp_path / "logs" / "approvals.jsonl"),
            model_label="test",
        )

    return _make


@pytest.fixture()
def assistant(make_assistant) -> MedicalAssistant:
    return make_assistant()


def fake_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


@pytest.fixture()
def clinic(tmp_path):
    database = ClinicDatabase(tmp_path / "clinic.sqlite3", bootstrap_token="bootstrap-test")
    doctor = database.provision_doctor(
        "SYN-D-01", "Doutora Demonstração", "Clínica geral", "bootstrap-test", synthetic_demo_confirmed=True
    )
    reviewer = database.provision_doctor(
        "SYN-D-02", "Médico Revisor", "Clínica geral", "bootstrap-test", synthetic_demo_confirmed=True
    )
    patient = database.register_patient(
        "Paciente Sintético",
        "111.444.777-35",
        "1980-01-02",
        "(11) 91234-5678",
        "paciente@example.test",
        synthetic_demo_confirmed=True,
    )
    return database, doctor, reviewer, patient
