"""Preparação do dataset de fine-tuning: limpeza, validação, anonimização e conversão para chat.

Propósito do fine-tuning neste projeto: ensinar COMPORTAMENTO (formato, citação [DOC-n], recusas e
tom) — não memorizar conhecimento clínico. A fonte de verdade clínica é o RAG versionado.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.prompts.templates import SYSTEM_PROMPT, build_user_message
from src.models.schemas import SourceDocument
from src.rag.loader import load_protocols
from src.security.anonymizer import anonymize, find_pii
from src.security.guardrails import REFUSAL_MESSAGES, check_input, check_output

_TAGS = re.compile(r"<[^>]+>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SECTION = re.compile(r"^##\s+(.+?)\s*$\n(.*?)(?=^##\s|\Z)", re.MULTILINE | re.DOTALL)

MIN_CHARS, MAX_CHARS = 5, 4000


# ---------------------------------------------------------------- limpeza
def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = _TAGS.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- exemplos
def make_example(user: str, assistant: str, source: str, category: str = "answer") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {"source": source, "category": category},
    }


def validate_example(ex: dict[str, Any]) -> list[str]:
    """Retorna a lista de problemas (vazia = válido)."""
    errors: list[str] = []
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        return ["estrutura de mensagens inválida (esperado system/user/assistant)"]
    for m in msgs:
        content = m.get("content")
        if not isinstance(content, str) or not (MIN_CHARS <= len(content) <= MAX_CHARS + len(SYSTEM_PROMPT)):
            errors.append(f"conteúdo de '{m.get('role')}' vazio ou fora do limite de tamanho")
    if errors:
        return errors
    # PII só é checada em user/assistant (o system prompt é fixo e conhecido).
    for m in msgs[1:]:
        if find_pii(m["content"], None):
            errors.append(f"PII detectada em '{m['role']}'")
    answer = msgs[2]["content"]
    if not check_output(answer).allowed:
        errors.append("resposta viola guardrail de saída (diagnóstico/prescrição/dose/PII)")
    if ex.get("meta", {}).get("category") == "refusal" and check_input(msgs[1]["content"]).allowed:
        errors.append("exemplo de recusa cuja pergunta NÃO é bloqueada pelo guardrail de entrada")
    return errors


# ---------------------------------------------------------------- fontes
def examples_from_protocols(protocols_dir: Path) -> list[dict[str, Any]]:
    """Pergunta por seção do protocolo; resposta = trecho da própria seção com citação."""
    out: list[dict[str, Any]] = []
    for doc in load_protocols(protocols_dir):
        meta = doc.metadata
        src = SourceDocument(
            doc_id=meta["doc_id"], title=meta["title"], version=meta["version"], date=meta["date"],
            excerpt=clean_text(doc.page_content)[:1200], score=0.9,
        )
        for heading, body in _SECTION.findall(doc.page_content):
            sentences = [s for s in _SENTENCE.split(clean_text(body)) if s]
            if not sentences:
                continue
            question = f"O que o protocolo diz sobre {heading.strip().lower()}?"
            answer = f"Segundo o protocolo recuperado: {' '.join(sentences[:3])} [DOC-1]"
            out.append(make_example(build_user_message(question, [src]), answer, f"protocolo:{meta['doc_id']}"))
    return out


def examples_from_faqs(path: Path, protocols_dir: Path) -> list[dict[str, Any]]:
    docs = {d.metadata["doc_id"]: d for d in load_protocols(protocols_dir)}
    out = []
    for row in read_jsonl(path):
        doc = docs.get(row["doc_id"])
        if doc is None:
            continue
        m = doc.metadata
        src = SourceDocument(
            doc_id=m["doc_id"], title=m["title"], version=m["version"], date=m["date"],
            excerpt=clean_text(doc.page_content)[:1200], score=0.9,
        )
        out.append(make_example(build_user_message(clean_text(row["question"]), [src]), clean_text(row["answer"]), f"faq:{m['doc_id']}"))
    return out


def examples_from_refusals(path: Path) -> list[dict[str, Any]]:
    """Recusas ensinam o modelo a não diagnosticar/prescrever/dosar; texto = o do guardrail."""
    out = []
    for row in read_jsonl(path):
        q = clean_text(row["question"])
        res = check_input(q)
        answer = res.message if not res.allowed else REFUSAL_MESSAGES[row_category(row)]
        out.append(make_example(build_user_message(q, []), answer, "recusa", "refusal"))
    return out


def row_category(row: dict[str, Any]):
    from src.security.guardrails import BlockCategory

    return BlockCategory(row["category"])


def examples_from_pubmedqa(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    """Rejeita uso de PubMedQA em SFT; mantenha o corpus somente no RAG."""
    raise ValueError(
        "PubMedQA contém fatos clínicos variáveis e não deve gerar exemplos de fine-tuning. "
        "Use normalize_public_sources para disponibilizá-lo ao RAG."
    )


def examples_from_medquad(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    """Rejeita uso de MedQuAD em SFT; mantenha o corpus somente no RAG."""
    raise ValueError(
        "MedQuAD contém fatos clínicos variáveis e não deve gerar exemplos de fine-tuning. "
        "Use normalize_public_sources para disponibilizá-lo ao RAG."
    )


def normalize_public_sources(
    path: Path,
    source_name: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Limpa e anonimiza fontes públicas; não as marca como aprovadas clinicamente."""
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(_read_any(path), start=1):
        if source_name == "pubmedqa":
            context = " ".join((row.get("context") or {}).get("contexts") or [])
            answer = row.get("long_answer", "")
            source_id = f"PUBMEDQA-{row.get('pubid') or index}"
        elif source_name == "medquad":
            context = row.get("context", "")
            answer = row.get("answer", "")
            source_id = f"MEDQUAD-{index:05d}"
        else:
            raise ValueError(f"Fonte pública não suportada: {source_name}")

        question = anonymize(clean_text(row.get("question", "")), labels=None)
        cleaned_context = anonymize(clean_text(str(context)), labels=None)
        cleaned_answer = anonymize(clean_text(str(answer)), labels=None)
        if not question or not cleaned_answer:
            continue
        normalized.append(
            {
                "source": source_name,
                "source_id": source_id,
                "title": anonymize(clean_text(str(row.get("document_source", source_name))), labels=None),
                "question": question,
                "context": cleaned_context,
                "answer": cleaned_answer,
                "approved_for_rag": False,
                "purpose": "fonte pública não curada; não usar como resposta de fine-tuning",
            }
        )
        if len(normalized) >= limit:
            break
    return normalized


# ---------------------------------------------------------------- IO
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _read_any(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------- pipeline
@dataclass
class BuildReport:
    total_in: int = 0
    kept: int = 0
    dropped_invalid: int = 0
    dropped_duplicates: int = 0
    anonymized_fields: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=dict)


def dedupe_key(ex: dict[str, Any]) -> str:
    m = ex["messages"]
    return hashlib.sha256((m[1]["content"] + "\x00" + m[2]["content"]).encode()).hexdigest()


def build_dataset(examples: list[dict[str, Any]], seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple[dict[str, list[dict]], BuildReport]:
    """Limpa → anonimiza → valida → deduplica → divide (train/val/test, determinístico)."""
    report = BuildReport(total_in=len(examples))
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for ex in examples:
        # Anonimização defensiva dos turnos user/assistant antes da validação.
        for m in ex["messages"][1:]:
            cleaned = clean_text(m["content"]) if ex["meta"]["category"] != "refusal" else m["content"]
            anon = anonymize(cleaned)
            if anon != cleaned:
                report.anonymized_fields += 1
            m["content"] = anon
        problems = validate_example(ex)
        if problems:
            report.dropped_invalid += 1
            for p in problems:
                report.reasons[p] = report.reasons.get(p, 0) + 1
            continue
        key = dedupe_key(ex)
        if key in seen:
            report.dropped_duplicates += 1
            continue
        seen.add(key)
        kept.append(ex)
    rng = random.Random(seed)
    rng.shuffle(kept)
    n_test, n_val = int(len(kept) * test_ratio), int(len(kept) * val_ratio)
    splits = {"test": kept[:n_test], "validation": kept[n_test : n_test + n_val], "train": kept[n_test + n_val :]}
    report.kept = len(kept)
    report.splits = {k: len(v) for k, v in splits.items()}
    return splits, report
