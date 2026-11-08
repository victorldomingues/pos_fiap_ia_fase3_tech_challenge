"""Fábrica de LLMs LangChain: backend extrativo (sem GPU) ou Qwen2.5-3B-Instruct + adaptador QLoRA."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

from src.finetuning.artifacts import validate_adapter
from src.security.guardrails import normalize
from src.utils.config import Settings

_DOC_BLOCK = re.compile(r"\[DOC-(\d+)\][^\n]*\n(.*?)(?=\n\n\[DOC-|\Z)", re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w{4,}")
_HEADING = re.compile(r"^#+\s.*$", re.MULTILINE)


class ExtractiveChatModel(SimpleChatModel):
    """LLM determinístico: devolve as frases do contexto mais próximas da pergunta, com citação.

    Serve para demonstração offline e testes de integração; NÃO substitui o LLM fine-tunado.
    """

    @property
    def _llm_type(self) -> str:
        return "extractive-fallback"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        text = str(messages[-1].content)
        question = normalize(text.split("Pergunta:", 1)[-1])
        q_words = set(_WORD.findall(question)) - {"quais", "qual"}
        blocks = _DOC_BLOCK.findall(text.split("</contexto>")[0])
        if not blocks:
            return "Não há fonte confiável nos documentos recuperados para responder. Procure um profissional de saúde."
        candidates: list[tuple[int, int, str]] = []
        for number, body in blocks:
            sentences = _SENTENCE.split(" ".join(_HEADING.sub("", body).split()))
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or sentence.lower().startswith("pergunta original:"):
                    continue
                sentence = re.sub(r"^(?:contexto|resposta original):\s*", "", sentence, flags=re.IGNORECASE)
                overlap = q_words & set(_WORD.findall(normalize(sentence)))
                if overlap:
                    candidates.append((len(overlap), int(number), sentence))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen: list[tuple[int, str]] = []
        seen: set[str] = set()
        for _, number, sentence in candidates:
            key = normalize(sentence)
            if key not in seen:
                seen.add(key)
                chosen.append((number, sentence))
            if len(chosen) == 2:
                break
        if not chosen:
            return "Não encontrei uma afirmação diretamente relacionada à pergunta nas fontes recuperadas."
        findings = " ".join(f"{sentence} [DOC-{number}]" for number, sentence in chosen)
        return f"Segundo as fontes recuperadas: {findings}"


def _load_hf_chat_model(settings: Settings):
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    adapter_meta = None
    if settings.use_adapter:
        adapter_meta = validate_adapter(settings.adapter_dir, settings.base_model_id)

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model_id)
    quant = None
    if torch.cuda.is_available():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(settings.base_model_id, quantization_config=quant, device_map="auto")
    if adapter_meta is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(settings.adapter_dir))

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))


def load_chat_model(settings: Settings):
    backend = settings.llm_backend.lower()
    if backend == "extractive":
        return ExtractiveChatModel()
    if backend == "hf":
        return _load_hf_chat_model(settings)
    raise ValueError(f"MED_LLM_BACKEND inválido: '{settings.llm_backend}' (use 'extractive' ou 'hf').")
