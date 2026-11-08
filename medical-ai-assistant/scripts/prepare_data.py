"""Gera data/processed/{train,validation,test}.jsonl (limpeza → anonimização → validação → split).

Uso: python scripts/prepare_data.py [--pubmedqa data/raw/pubmedqa.jsonl] [--medquad data/raw/medquad.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.finetuning.dataset import (  # noqa: E402
    build_dataset,
    examples_from_faqs,
    examples_from_protocols,
    examples_from_refusals,
    normalize_public_sources,
    write_jsonl,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--protocols", type=Path, default=ROOT / "data" / "protocols")
    p.add_argument("--synthetic", type=Path, default=ROOT / "data" / "synthetic")
    p.add_argument("--pubmedqa", type=Path, default=None)
    p.add_argument("--medquad", type=Path, default=None)
    p.add_argument("--limit", type=int, default=200, help="máx. de exemplos por dataset público")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    a = p.parse_args()

    examples = examples_from_protocols(a.protocols)
    examples += examples_from_faqs(a.synthetic / "faqs.jsonl", a.protocols)
    examples += examples_from_refusals(a.synthetic / "refusals.jsonl")
    public_sources = []
    for path, source_name in ((a.pubmedqa, "pubmedqa"), (a.medquad, "medquad")):
        if path:
            public_sources += normalize_public_sources(path, source_name, a.limit)

    splits, report = build_dataset(examples)
    for name, rows in splits.items():
        write_jsonl(a.out / f"{name}.jsonl", rows)
    if public_sources:
        write_jsonl(a.out / "public_sources.jsonl", public_sources)
    print(json.dumps(report.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
