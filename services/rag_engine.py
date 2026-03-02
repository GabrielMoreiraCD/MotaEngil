'''
rag_engine.py - Motor de Recuperação Augmentada (RAG) para responder perguntas técnicas com base em normas técnicas e documentos relacionados.
============================================================================================================
Este módulo implementa o RAGEngine, que integra um LLM com um sistema de recuperação de documentos baseado em vetores. O motor é projetado para responder perguntas técnicas utilizando informações extraídas de documentos normativos, imagens e diagramas.
'''
import logging
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from core.models import get_llm, get_embeddings
from core.vectorstore import VectorStoreManager
from services.router import route_query

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        logger.info("Inicializando Motor RAG e instanciando modelos...")
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.vector_manager = VectorStoreManager(self.embeddings)
        self.document_chain = self._build_document_chain()

    def _build_document_chain(self):
        system_prompt = (
            "Você é um assistente de engenharia especialista em normas técnicas. "
            "Responda EXCLUSIVAMENTE com base no contexto fornecido. O contexto pode conter "
            "texto extraído diretamente do documento ou descrições técnicas de imagens e diagramas.\n\n"
            "Contexto:\n{context}"
        )
        prompt = PromptTemplate.from_template(
            system_prompt + "\n\nPergunta: {input}\nResposta:"
        )
        return create_stuff_documents_chain(self.llm, prompt)

    def process_query(self, query: str) -> dict:
        logger.info("Avaliando roteamento da query...")
        tipo_filtro = route_query(query, self.llm)
        logger.info(f"Filtro vetorial aplicado: {tipo_filtro.upper()}")

        retriever = self.vector_manager.get_retriever(filter_type=tipo_filtro)
        rag_chain = create_retrieval_chain(retriever, self.document_chain)

        # Recuperação explícita apenas para logging/exibição
        retrieved_docs = retriever.invoke(query)
        
        logger.info("Gerando resposta técnica...")
        response = rag_chain.invoke({"input": query})
        
        return {
            "answer": response["answer"],
            "docs": retrieved_docs,
            "filter": tipo_filtro
        }