"""
ingest.py - Pipeline de Ingestão para Documentos Técnicos
============================================================================================================
Este módulo implementa a função `run_ingestion`, que orquestra o processo completo de ingestão de um documento técnico (PDF) para o sistema RAG. O pipeline inclui:
1. Extração multimodal de texto e imagens utilizando o PDFMultimodalExtractor.
2. Processamento de texto com chunking estrutural específico para normas técnicas.
3. Processamento de visão computacional para gerar descrições textuais de imagens e diagramas
4. Avaliação de qualidade (QA) para garantir a integridade e utilidade dos chunks gerados.
5. Formatação dos dados para LangChain e indexação vetorial no Qdrant.
"""
import sys
import logging
from core.extractors import PDFMultimodalExtractor
from core.chunkers import NormativeChunker
from services.vision_engine import LocalVisionEngine
from services.ingestion_qa import ChunkEvaluator
from core.models import get_embeddings
from core.vectorstore import VectorStoreManager
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("INGESTION_PIPELINE")

def run_ingestion(pdf_path: str, norma_id: str):
    logger.info(f"Iniciando processamento do documento: {pdf_path}")
    
    # 1. Extração
    extractor = PDFMultimodalExtractor(pdf_path)
    text_elements, image_elements = extractor.extract_elements()
    logger.info(f"Extraídos {len(text_elements)} blocos de texto e {len(image_elements)} imagens.")

    # 2. Processamento de Texto (Chunking Estrutural)
    chunker = NormativeChunker()
    text_chunks = chunker.process_text_elements(text_elements, norma_id)

    # 3. Processamento de Visão Computacional
    vision = LocalVisionEngine()
    image_chunks = []
    
    for img_data in image_elements:
        logger.info(f"Gerando transcrição para imagem da página {img_data['page']}...")
        description = vision.describe_image(img_data["image_obj"])
        logger.info(f"\n--- INÍCIO DA DESCRIÇÃO DA IMAGEM (Pág {img_data['page']}) ---\n{description}\n--- FIM DA DESCRIÇÃO ---\n")
        
        if description and "Falha" not in description:
            image_chunks.append({
                "content": description,
                "metadata": {
                    "tipo": "imagem_descrita",
                    "page": img_data["page"],
                    "norma_id": norma_id
                }
            })

    all_raw_chunks = text_chunks + image_chunks

    # 4. Avaliação de Qualidade (QA)
    qa_engine = ChunkEvaluator()
    approval_rate = qa_engine.evaluate_batch(text_chunks, sample_size=10)
    
    if approval_rate < 0.7:
        logger.error("Falha no Quality Assurance. Chunks muito fragmentados ou corrompidos. Abortando indexação.")
        sys.exit(1)

    # 5. Formatação para LangChain e Indexação Vetorial
    langchain_docs = [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in all_raw_chunks
    ]

    logger.info(f"Iniciando indexação de {len(langchain_docs)} vetores no Qdrant...")
    embeddings = get_embeddings()
    vector_manager = VectorStoreManager(embeddings)
    
    vector_manager.vector_store.add_documents(langchain_docs)
    logger.info("Ingestão concluída com sucesso e metadados estruturados gravados.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python ingest.py <caminho_do_pdf> <id_da_norma>")
        sys.exit(1)
        
    run_ingestion(sys.argv[1], sys.argv[2])