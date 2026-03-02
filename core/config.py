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

config = Config()