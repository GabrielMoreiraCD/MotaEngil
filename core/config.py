"""
config.py — Configurações para o Pipeline RAG
==========================================================================
Este módulo define as configurações necessárias para o pipeline RAG da Petrobras.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # ── Pipeline BOM (Etapas 3-5) ─────────────────────────────────────────────
    COLLECTION_NORMAS = os.getenv("COLLECTION_NORMAS", "normas_tecnicas_publicas_v2")
    COLLECTION_MATERIAIS = os.getenv("COLLECTION_MATERIAIS", "catalogo_materiais_v1")

    # ── Ollama local (modelo principal — roda 100% offline) ─────────────────
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "4096"))
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

    # ── HuggingFace (Qwen2.5-VL para isométricos) ─────────────────────────────
    HF_TOKEN = os.getenv("HF_TOKEN")
    ISOMETRIC_MODEL = os.getenv("ISOMETRIC_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

config = Config()