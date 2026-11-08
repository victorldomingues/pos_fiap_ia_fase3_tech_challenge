"""Fine-tuning QLoRA do Qwen2.5-3B-Instruct (executar no Google Colab com GPU).

Uso:
    python -m src.finetuning.train_qlora --train data/processed/train.jsonl \
        --validation data/processed/validation.jsonl --output-dir /content/drive/MyDrive/medical-ai/adapter

Princípios: treino é ação explícita (nunca automático); a saída (adaptador + tokenizer + metadados)
deve ir para um destino persistente (ex.: Google Drive), pois /content é efêmero.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.finetuning.artifacts import write_training_metadata

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    base_model_id: str = DEFAULT_BASE_MODEL
    max_seq_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))
    epochs: float = 3.0
    learning_rate: float = 2e-4
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    warmup_ratio: float = 0.03
    seed: int = 42


def build_features(tokenizer: Any, messages: list[dict[str, str]], max_len: int) -> dict[str, list[int]]:
    """Tokeniza com o chat template e mascara o prompt: só a resposta do assistente gera loss."""
    prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    answer_text = messages[2]["content"] + "<|im_end|>\n"  # fecha o turno no formato ChatML do Qwen
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def train(cfg: TrainConfig, train_path: Path, val_path: Path | None, output_dir: Path, overwrite: bool = False) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requer GPU CUDA. No Colab: Ambiente de execução > Alterar tipo > GPU (T4 ou superior).")
    output_dir = Path(output_dir)
    if (output_dir / "adapter_config.json").exists() and not overwrite:
        raise FileExistsError(f"Já existe adaptador em {output_dir}. Use --overwrite para retreinar explicitamente.")

    set_seed(cfg.seed)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantização NF4 de 4 bits (o "Q" do QLoRA) com dupla quantização.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    def to_dataset(path: Path) -> Dataset:
        feats = [build_features(tokenizer, r["messages"], cfg.max_seq_len) for r in _read_rows(path)]
        return Dataset.from_list(feats)

    train_ds = to_dataset(train_path)
    eval_ds = to_dataset(val_path) if val_path and Path(val_path).exists() else None

    # transformers>=5 removeu warmup_ratio (warmup_steps aceita fração); <5 só aceita warmup_ratio
    warmup_kw = (
        {"warmup_ratio": cfg.warmup_ratio}
        if "warmup_ratio" in inspect.signature(TrainingArguments).parameters
        else {"warmup_steps": cfg.warmup_ratio}
    )
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        **warmup_kw,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=5,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",  # sem envio de métricas/conteúdo a terceiros
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX),
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_training_metadata(output_dir, cfg.base_model_id, asdict(cfg), train_path)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fine-tuning QLoRA (Qwen2.5-3B-Instruct)")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True, help="destino PERSISTENTE (ex.: pasta no Google Drive)")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    cfg = TrainConfig(
        base_model_id=a.base_model, epochs=a.epochs, learning_rate=a.lr, max_seq_len=a.max_seq_len,
        lora_r=a.lora_r, lora_alpha=a.lora_r * 2, per_device_batch_size=a.batch_size, grad_accum_steps=a.grad_accum,
    )
    out = train(cfg, a.train, a.validation, a.output_dir, a.overwrite)
    print(f"Adaptador salvo em: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
