"""Embeddings multilíngues (HuggingFaceEmbeddings). Import tardio para não exigir torch em testes."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


def get_embeddings(model_id: str) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    # Normalizar permite usar distância de cosseno de forma consistente no Chroma.
    return HuggingFaceEmbeddings(model_name=model_id, encode_kwargs={"normalize_embeddings": True})
