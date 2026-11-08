"""Avalia o assistente completo. Uso: python scripts/evaluate.py [--out logs/eval_report.json]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.factory import build_assistant  # noqa: E402
from src.finetuning.dataset import read_jsonl  # noqa: E402
from src.finetuning.evaluate import evaluate_assistant, save_report  # noqa: E402
from src.utils.config import Settings  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", type=Path, default=ROOT / "data" / "eval" / "eval_set.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "eval_report.json")
    a = p.parse_args()
    report = evaluate_assistant(build_assistant(Settings()), read_jsonl(a.eval_set))
    save_report(report, a.out)
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
