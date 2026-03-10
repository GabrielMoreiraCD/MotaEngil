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
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    USE_CLAUDE = os.getenv("USE_CLAUDE", "false").lower() == "true"
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    COLLECTION_NORMAS = os.getenv("COLLECTION_NORMAS", "normas_tecnicas_publicas_v2")
    COLLECTION_MATERIAIS = os.getenv("COLLECTION_MATERIAIS", "catalogo_materiais_v1")

    # ── Gemini (substitui Claude — gratuito via google-generativeai) ──────────
    GEMINI_KEY = os.getenv("GEMINI_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    USE_GEMINI = bool(os.getenv("GEMINI_KEY"))  # True sempre que GEMINI_KEY presente

    # ── HuggingFace (Qwen2.5-VL para isométricos) ─────────────────────────────
    HF_TOKEN = os.getenv("HF_TOKEN")
    ISOMETRIC_MODEL = os.getenv("ISOMETRIC_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

config = Config()