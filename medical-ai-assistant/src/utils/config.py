"""Configuração central lida de variáveis de ambiente (sem segredos no código)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


@dataclass(frozen=True)
class Settings:
    """Parâmetros do assistente. Em Colab, aponte *_DIR para um destino persistente (Drive)."""

    protocols_dir: Path = field(default_factory=lambda: _env_path("MED_PROTOCOLS_DIR", PROJECT_ROOT / "data" / "protocols"))
    index_dir: Path = field(default_factory=lambda: _env_path("MED_INDEX_DIR", PROJECT_ROOT / "artifacts" / "chroma"))
    adapter_dir: Path = field(default_factory=lambda: _env_path("MED_ADAPTER_DIR", PROJECT_ROOT / "artifacts" / "qwen-qlora-adapter"))
    logs_dir: Path = field(default_factory=lambda: _env_path("MED_LOGS_DIR", PROJECT_ROOT / "logs"))
    database_path: Path = field(
        default_factory=lambda: _env_path("MED_DATABASE_PATH", PROJECT_ROOT / "data" / "sqlite" / "medical_assistant.sqlite3")
    )
    database_bootstrap_token: str | None = field(default_factory=lambda: os.getenv("MED_DB_BOOTSTRAP_TOKEN"))
    public_rag_enabled: bool = field(default_factory=lambda: _env_bool("MED_PUBLIC_RAG_ENABLED"))
    public_sources_path: Path = field(
        default_factory=lambda: _env_path("MED_PUBLIC_SOURCES_PATH", PROJECT_ROOT / "data" / "processed" / "public_sources.jsonl")
    )

    base_model_id: str = field(default_factory=lambda: os.getenv("MED_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    embedding_model_id: str = field(
        default_factory=lambda: os.getenv("MED_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )
    # "extractive" roda sem GPU (demo/testes); "hf" carrega Qwen (+ adaptador QLoRA se existir).
    llm_backend: str = field(default_factory=lambda: os.getenv("MED_LLM_BACKEND", "extractive"))
    use_adapter: bool = field(default_factory=lambda: _env_bool("MED_USE_ADAPTER", True))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MED_MAX_NEW_TOKENS", "384")))

    top_k: int = field(default_factory=lambda: int(os.getenv("MED_TOP_K", "3")))
    min_confidence: float = field(default_factory=lambda: float(os.getenv("MED_MIN_CONFIDENCE", "0.35")))
    chunk_size: int = 600
    chunk_overlap: int = 80

    langsmith_enabled: bool = field(default_factory=lambda: _env_bool("MED_LANGSMITH_ENABLED"))
    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "medical-ai-assistant"))

    @property
    def audit_log_path(self) -> Path:
        return self.logs_dir / "audit.jsonl"

    @property
    def public_index_dir(self) -> Path:
        return self.index_dir / "public_sources"

    @property
    def app_log_path(self) -> Path:
        return self.logs_dir / "app.log"

    @property
    def approvals_path(self) -> Path:
        return self.logs_dir / "approvals.jsonl"
