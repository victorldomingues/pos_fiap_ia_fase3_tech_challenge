"""Gerenciamento dos artefatos do fine-tuning (adaptador PEFT/QLoRA + metadados).

Sem dependências pesadas: pode ser usado para validar artefatos antes de carregar o modelo.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METADATA_NAME = "training_metadata.json"
ADAPTER_FILES = ("adapter_config.json",)
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


class AdapterError(RuntimeError):
    """Adaptador ausente, incompleto ou incompatível com o modelo-base configurado."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_training_metadata(adapter_dir: Path, base_model_id: str, hyperparams: dict[str, Any], dataset_path: Path | None = None) -> Path:
    """Registra proveniência do treino: base, hiperparâmetros e hash do dataset."""
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_model_id": base_model_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hyperparameters": hyperparams,
        "dataset_sha256": file_sha256(dataset_path) if dataset_path and Path(dataset_path).exists() else None,
        "purpose": "comportamento/formato/recusas; NÃO é fonte de conhecimento clínico (fonte = RAG versionado)",
    }
    out = adapter_dir / METADATA_NAME
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def validate_adapter(adapter_dir: Path, expected_base_model_id: str) -> dict[str, Any]:
    """Falha com instruções claras se o adaptador não existir ou for de outro modelo-base."""
    adapter_dir = Path(adapter_dir)
    hint = "Treine com a Parte 2 do notebook IADT 3 ou 'python -m src.finetuning.train_qlora' e aponte MED_ADAPTER_DIR."
    if not adapter_dir.is_dir():
        raise AdapterError(f"Adaptador não encontrado em {adapter_dir}. {hint}")
    for name in ADAPTER_FILES:
        if not (adapter_dir / name).exists():
            raise AdapterError(f"Arquivo '{name}' ausente em {adapter_dir}. {hint}")
    if not any((adapter_dir / w).exists() for w in ADAPTER_WEIGHTS):
        raise AdapterError(f"Pesos do adaptador ausentes em {adapter_dir}. {hint}")
    meta_path = adapter_dir / METADATA_NAME
    if not meta_path.exists():
        raise AdapterError(f"'{METADATA_NAME}' ausente em {adapter_dir}: proveniência desconhecida. {hint}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("base_model_id") != expected_base_model_id:
        raise AdapterError(
            f"Adaptador treinado sobre '{meta.get('base_model_id')}', mas o modelo-base configurado é "
            f"'{expected_base_model_id}'."
        )
    return meta
