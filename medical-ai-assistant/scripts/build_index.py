"""Constrói/atualiza o índice ChromaDB (ação explícita). Uso: python scripts/build_index.py [--rebuild]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assistant.factory import build_index, build_public_index  # noqa: E402
from src.rag.loader import load_protocols  # noqa: E402
from src.rag.public_sources import load_public_sources  # noqa: E402
from src.utils.config import Settings  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true", help="apaga e recria o índice (obrigatório se o modelo de embeddings mudou)")
    p.add_argument("--with-public-sources", action="store_true", help="indexa também o corpus público não curado em índice separado")
    a = p.parse_args()
    settings = Settings()
    docs = load_protocols(settings.protocols_dir)
    report = build_index(settings).sync(docs, rebuild=a.rebuild)
    print(f"Índice em {settings.index_dir}")
    print(f"  adicionados={report.added} atualizados={report.updated} removidos={report.removed} inalterados={report.unchanged}")
    if a.with_public_sources:
        public_docs = load_public_sources(settings.public_sources_path)
        public_index = build_public_index(settings)
        public_report = public_index.sync(public_docs, rebuild=a.rebuild)
        print(f"Índice público não curado em {settings.public_index_dir}: {len(public_docs)} documentos")
        print(f"  adicionados={len(public_report.added)} atualizados={len(public_report.updated)} removidos={len(public_report.removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
