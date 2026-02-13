"""
Factories for LLM + Embeddings so you can switch between OpenAI and Ollama
using environment variables (see .env.example).
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .config import settings


def build_llm() -> BaseChatModel:
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env")
        from langchain_openai import ChatOpenAI  # type: ignore

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=settings.openai_api_key,
        )

    if settings.provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
            temperature=0.2,
        )

    raise RuntimeError(f"Unknown PROVIDER={settings.provider}. Use 'openai' or 'ollama'.")


def build_embeddings() -> Embeddings:
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env")
        from langchain_openai import OpenAIEmbeddings  # type: ignore

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )

    if settings.provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )

    raise RuntimeError(f"Unknown PROVIDER={settings.provider}. Use 'openai' or 'ollama'.")
