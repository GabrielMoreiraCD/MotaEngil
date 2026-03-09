'''models.py — Modelos de Embeddings e LLM para o Pipeline RAG
==========================================================================
Este módulo define os modelos de embeddings e LLM utilizados no pipeline RAG da Petrobras, utilizando as bibliotecas HuggingFace e Ollama.'''
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from core.config import config

def get_embeddings() -> HuggingFaceEmbeddings:
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def get_llm() -> Ollama:
    return Ollama(
        model=config.LLM_MODEL,
        temperature=0.0
    )

def get_claude_client():
    """
    Retorna cliente Anthropic para uso nos agentes que requerem extração estruturada
    de alta fidelidade (ex: NormasConsultationAgent na Etapa 3).
    Requer ANTHROPIC_API_KEY no .env.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "Pacote 'anthropic' não encontrado. Execute: pip install anthropic>=0.28.0"
        ) from e

    if not config.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY não configurada. Adicione ao .env ou use --use-llama para fallback local."
        )

    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)