import json

import pytest

from src.finetuning.artifacts import AdapterError, validate_adapter, write_training_metadata
from src.finetuning.dataset import (
    build_dataset,
    clean_text,
    examples_from_faqs,
    examples_from_medquad,
    examples_from_protocols,
    examples_from_pubmedqa,
    examples_from_refusals,
    make_example,
    validate_example,
)
from src.finetuning.train_qlora import IGNORE_INDEX, build_features
from src.prompts.templates import SYSTEM_PROMPT
from src.utils.patients import load_patients, pseudonymize
from tests.conftest import PROTOCOLS_DIR, ROOT

SYN = ROOT / "data" / "synthetic"


def test_clean_text_strips_html_and_whitespace():
    assert clean_text("<p>Olá&nbsp;&amp;  mundo</p>\n\t") == "Olá & mundo"


def test_validate_example_flags_pii_and_unsafe_answers():
    assert validate_example(make_example("Pergunta válida?", "Resposta válida. [DOC-1]", "t")) == []
    assert any("PII" in e for e in validate_example(make_example("CPF 111.444.777-35 ok?", "Resposta válida.", "t")))
    assert any("guardrail" in e for e in validate_example(make_example("Pergunta válida?", "Tome 500 mg ao dia.", "t")))
    assert validate_example({"messages": []})


def test_validate_refusal_requires_blocked_question():
    assert validate_example(make_example("Qual o horário de visitas?", "Não posso.", "r", "refusal"))


def test_sources_generate_valid_examples():
    protocols = examples_from_protocols(PROTOCOLS_DIR)
    faqs = examples_from_faqs(SYN / "faqs.jsonl", PROTOCOLS_DIR)
    refusals = examples_from_refusals(SYN / "refusals.jsonl")
    assert len(protocols) >= 16 and len(faqs) >= 10 and len(refusals) >= 10
    splits, report = build_dataset(protocols + faqs + refusals)
    assert report.dropped_invalid == 0, report.reasons
    assert sum(report.splits.values()) == report.kept and splits["train"]


def test_build_dataset_deduplicates_and_is_deterministic():
    ex = make_example("Pergunta válida?", "Resposta válida. [DOC-1]", "t")
    import copy

    s1, r1 = build_dataset([copy.deepcopy(ex), copy.deepcopy(ex)])
    assert r1.dropped_duplicates == 1 and r1.kept == 1
    many = [make_example(f"Pergunta {i} válida?", f"Resposta {i} válida. [DOC-1]", "t") for i in range(30)]
    a, _ = build_dataset(copy.deepcopy(many))
    b, _ = build_dataset(copy.deepcopy(many))
    assert a == b


def test_public_dataset_loaders(tmp_path):
    pubmed = tmp_path / "p.jsonl"
    pubmed.write_text(json.dumps({"pubid": 1, "question": "Does X help?", "context": {"contexts": ["Study of X."]}, "long_answer": "X may help in some settings."}) + "\n")
    medquad = tmp_path / "m.jsonl"
    medquad.write_text(json.dumps({"question": "What is Y?", "answer": "Y is a condition. It is common."}) + "\n")
    with pytest.raises(ValueError, match="não deve gerar exemplos de fine-tuning"):
        examples_from_pubmedqa(pubmed)
    with pytest.raises(ValueError, match="não deve gerar exemplos de fine-tuning"):
        examples_from_medquad(medquad)

    from src.finetuning.dataset import normalize_public_sources

    corpus = normalize_public_sources(pubmed, "pubmedqa")
    assert corpus[0]["approved_for_rag"] is False
    assert corpus[0]["answer"] == "X may help in some settings."


def test_checked_in_splits_use_current_prompt_without_public_dataset_refusals():
    from src.finetuning.dataset import read_jsonl

    assert "exclusivamente em português brasileiro (pt-BR)" in SYSTEM_PROMPT
    for split in ("train", "validation", "test"):
        rows = read_jsonl(ROOT / "data" / "processed" / f"{split}.jsonl")
        assert rows
        assert all(row["messages"][0]["content"] == SYSTEM_PROMPT for row in rows)
        assert all(row.get("meta", {}).get("category") != "no_source" for row in rows)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 255 for c in text]}


def test_build_features_masks_prompt_and_keeps_answer_labels():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "resposta"}]
    f = build_features(FakeTokenizer(), msgs, max_len=1000)
    n_prompt = sum(1 for x in f["labels"] if x == IGNORE_INDEX)
    assert n_prompt > 0 and len(f["labels"]) == len(f["input_ids"]) == len(f["attention_mask"])
    assert all(x != IGNORE_INDEX for x in f["labels"][n_prompt:])
    assert len(build_features(FakeTokenizer(), msgs, max_len=5)["input_ids"]) == 5


def test_adapter_validation(tmp_path):
    with pytest.raises(AdapterError, match="não encontrado"):
        validate_adapter(tmp_path / "nada", "base")
    d = tmp_path / "ad"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    with pytest.raises(AdapterError, match="Pesos"):
        validate_adapter(d, "base")
    (d / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(AdapterError, match="proveniência"):
        validate_adapter(d, "base")
    write_training_metadata(d, "base", {"r": 16})
    assert validate_adapter(d, "base")["base_model_id"] == "base"
    with pytest.raises(AdapterError, match="modelo-base"):
        validate_adapter(d, "outro")


def test_synthetic_patients_are_pseudonymized():
    patients = load_patients(SYN / "patients.json")
    assert len(patients) == 5
    out = pseudonymize(patients[0], salt="s")
    assert set(out) == {"pseudo_id", "birth_year", "notes"} and out["birth_year"] == "1980"
    assert out["pseudo_id"] != patients[0].patient_id
