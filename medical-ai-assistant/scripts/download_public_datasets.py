"""Baixa amostras de PubMedQA e MedQuAD do Hugging Face para data/raw (opcional, requer `datasets`).

Verifique licenças/termos de uso dos datasets e os identificadores no Hub antes de usar. A
disponibilidade pública não garante anonimização, segurança clínica nem permissão de redistribuição.
Os dados não são usados como respostas de fine-tuning. Uso: python scripts/download_public_datasets.py --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dump(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"salvo: {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    a = p.parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        print("Instale: pip install datasets", file=sys.stderr)
        return 2

    pubmed = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    _dump((pubmed[i] for i in range(min(a.limit, len(pubmed)))), a.out / "pubmedqa.jsonl")
    medquad = load_dataset("lavita/MedQuAD", split="train")
    _dump((medquad[i] for i in range(min(a.limit, len(medquad)))), a.out / "medquad.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
