'''
vectorstore.py — Gerenciamento do Vector Store para o Pipeline RAG
========================================================================== 
Este módulo implementa a classe `VectorStoreManager`, responsável por gerenciar a conexão e as operações com o Qdrant, incluindo a criação de índices de payload e a configuração de retrievers com filtros específicos para tipos de conteúdo.
'''
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType, Filter, FieldCondition, MatchValue
from langchain_community.vectorstores import Qdrant
from core.config import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, embeddings):
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )
        self.collection_name = config.COLLECTION_NAME
        self._ensure_payload_index()
        
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=embeddings,
        )

    def _ensure_payload_index(self):
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.tipo",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Índice de payload 'metadata.tipo' verificado/criado com sucesso.")
        except Exception as e:
            logger.debug(f"Índice possivelmente já existente: {e}")

    def get_retriever(self, filter_type: str = None, top_k: int = 50):
        search_kwargs = {"k": top_k}
        
        if filter_type and filter_type in ["imagem_descrita", "texto_extraido"]:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.tipo",
                        match=MatchValue(value=filter_type)
                    )
                ]
            )
            search_kwargs["filter"] = qdrant_filter
            
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)