"""Auditoria local append-only em JSONL com encadeamento de hashes (rastreabilidade).

Cada entrada inclui o hash da anterior; `verify()` detecta edição ou remoção de linhas.
Isto é evidência de adulteração, NÃO imutabilidade: para produção use armazenamento WORM/gerenciado.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(prev_hash: str, body: dict[str, Any]) -> str:
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS_HASH
        last = GENESIS_HASH
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    def record(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Grava um evento. Passe apenas dados já redigidos (sem PII)."""
        with self._lock:
            body = {"timestamp": utc_now(), "event": event_type, **fields, "prev_hash": self._last_hash}
            entry = {**body, "hash": _digest(self._last_hash, body)}
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._last_hash = entry["hash"]
            return entry

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def verify(self) -> bool:
        prev = GENESIS_HASH
        for entry in self.read():
            body = {k: v for k, v in entry.items() if k != "hash"}
            if body.get("prev_hash") != prev or _digest(prev, body) != entry["hash"]:
                return False
            prev = entry["hash"]
        return True


def setup_app_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """Logger de aplicação (arquivo local). Nunca registre conteúdo clínico ou PII aqui."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("medical_assistant")
    logger.setLevel(level)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path.resolve()) for h in logger.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
