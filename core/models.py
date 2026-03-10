"""models.py — Modelos de Embeddings e LLM para o Pipeline RAG
==========================================================================
Usa Ollama local (qwen2.5:7b-instruct-q4_K_M) para todas as chamadas LLM.
Suporta system/user prompts via ollama.chat() para extração estruturada.
"""
import logging
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama as OllamaLangchain
from core.config import config

log = logging.getLogger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def get_llm() -> OllamaLangchain:
    """LangChain Ollama wrapper — mantido para backward compat (rag_engine.py)."""
    return OllamaLangchain(model=config.OLLAMA_MODEL, temperature=0.0)


def ollama_chat(system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
    """
    Chama Ollama local via ollama.chat() com system + user prompts.
    Usa qwen2.5:7b-instruct-q4_K_M por padrão (configurável via OLLAMA_MODEL).
    Retorna o texto da resposta.
    """
    import ollama

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": temperature if temperature is not None else config.OLLAMA_TEMPERATURE,
                "num_predict": config.OLLAMA_NUM_PREDICT,
                "num_ctx": config.OLLAMA_NUM_CTX,
            },
        )
        return response["message"]["content"]
    except Exception as e:
        log.error(f"Erro ao chamar Ollama ({config.OLLAMA_MODEL}): {e}")
        raise
