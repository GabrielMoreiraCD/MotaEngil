'''ingest_books.py — Pipeline de Ingestão de Livros Teóricos para o RAG da Petrobras
==========================================================================
Este módulo implementa a pipeline de ingestão de livros teóricos para o sistema RAG da Petrobras. Ele processa uma coleção de PDFs, extraindo texto com técnicas avançadas de OCR e layout-aware, segmentando o conteúdo em chunks ricos em contexto e armazenando os embeddings resultantes em um cluster Qdrant remoto. O módulo é projetado para lidar com as complexidades específicas dos livros técnicos, como layouts de duas colunas e a necessidade de preservar a estrutura semântica do conteúdo.
'''
import os
import fitz  
import pytesseract
import logging
from pathlib import Path
from PIL import Image
import io

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# =========================================================
# CONFIGURAÇÃO DE LOGGING
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Ingestao_Livros_Teoricos")

# =========================================================
# CONFIGURAÇÕES DO AMBIENTE E BANCO DE DADOS
# =========================================================
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Variáveis QDRANT_URL ou QDRANT_API_KEY não encontradas no arquivo .env")

BOOKS_DIR = Path(r"C:\luza_datasets\Rag-pipeline-main\Data\books_ref")
COLLECTION_NAME = "petrobras_rag_teoria"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Caminho do executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\gabriel.moreira\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# =========================================================
# METADADOS E REGRAS DE EXTRAÇÃO DOS LIVROS
# =========================================================
BOOKS_CONFIG = {
    "CienciaeEngehariaDeMateriaisUmaIntroducao.pdf": {
        "tema": "Fundamentos da estrutura atômica, cristalina e microestrutural. Mecanismos de deformação, diagramas de fase, tratamentos térmicos e comportamento mecânico.",
        "precisa_ocr": True,
        "duas_colunas": True,
        "pagina_inicio_colunas": 27
    },
    "DamageMechanismsAffectingFixed.pdf": {
        "tema": "Mecanismos de degradação: corrosão, fadiga, fluência, fragilização por hidrogênio. Variáveis operacionais e modos de falha estrutural em equipamentos industriais.",
        "precisa_ocr": False,
        "duas_colunas": False,
        "pagina_inicio_colunas": 0
    },
    "NonDestructiveTestingHandbook.pdf": {
        "tema": "Ensaios não destrutivos (END): ultrassom, radiografia, partículas magnéticas, líquido penetrante. Detecção de trincas e avaliação de integridade estrutural.",
        "precisa_ocr": False,
        "duas_colunas": True,
        "pagina_inicio_colunas": 19
    },
    "PressureVesselDesignManual.pdf": {
        "tema": "Projeto mecânico de vasos de pressão. Cálculo de tensões circunferenciais e longitudinais. Critérios ASME para espessura, bocais, flanges e modos de falha.",
        "precisa_ocr": False,
        "duas_colunas": True,
        "pagina_inicio_colunas": 2
    },
    "TubulacõesIndustriaisCalculo.pdf": {
        "tema": "Dimensionamento mecânico de tubulações sujeitas a pressão e temperatura. Equações para espessura, análise de tensões, flexibilidade e suportes.",
        "precisa_ocr": True,
        "duas_colunas": False,
        "pagina_inicio_colunas": 0
    },
    "WeldingHandbook.pdf": {
        "tema": "Processos de soldagem industrial (SMAW, MIG/MAG, TIG). Metalurgia da soldagem, ZTA, defeitos típicos de solda e tensões residuais.",
        "precisa_ocr": False,
        "duas_colunas": True,
        "pagina_inicio_colunas": 9
    }
}

# =========================================================
# MOTOR DE EXTRAÇÃO DE TEXTO
# =========================================================
def extract_text_from_pdf(pdf_path: Path, config: dict) -> list[Document]:
    doc = fitz.open(pdf_path)
    documents = []
    
    precisa_ocr = config["precisa_ocr"]
    duas_colunas = config["duas_colunas"]
    pagina_inicio_colunas = config["pagina_inicio_colunas"]
    tema_contexto = config["tema"]

    for page_num in range(len(doc)):
        page = doc[page_num]
        pagina_real = page_num + 1
        page_text = ""

        if precisa_ocr:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img, lang='por')
            
        else:
            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[6] == 0]

            if duas_colunas and pagina_real >= pagina_inicio_colunas:
                page_width = page.rect.width
                center_x = page_width / 2.0
                
                left_column = []
                right_column = []
                
                for b in text_blocks:
                    x0, y0 = b[0], b[1]
                    if x0 < center_x:
                        left_column.append(b)
                    else:
                        right_column.append(b)
                
                left_column.sort(key=lambda b: b[1])
                right_column.sort(key=lambda b: b[1])
                
                sorted_blocks = left_column + right_column
            else:
                sorted_blocks = sorted(text_blocks, key=lambda b: b[1])
                
            page_text = "\n\n".join([b[4].strip() for b in sorted_blocks if b[4].strip()])

        if len(page_text.strip()) > 50:
            doc_metadata = {
                "source": pdf_path.name,
                "page": pagina_real,
                "tipo_documento": "livro_texto",
                "contexto_teorico": tema_contexto
            }
            documents.append(Document(page_content=page_text, metadata=doc_metadata))
            
    doc.close()
    return documents

# =========================================================
# ORQUESTRADOR PRINCIPAL
# =========================================================
def main():
    logger.info("A iniciar a pipeline de ingestão de livros teóricos.")
    
    if not BOOKS_DIR.exists():
        raise FileNotFoundError(f"O diretório especificado não foi encontrado: {BOOKS_DIR}")

    # 1. Configuração do Modelo de Embeddings
    logger.info(f"A carregar o modelo de embeddings: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Inicialização do Qdrant Client e Verificação da Coleção
    logger.info("A conectar ao cluster Qdrant remoto...")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Validação rigorosa: Cria a coleção se não existir
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        logger.info(f"A coleção '{COLLECTION_NAME}' não existe. Criando agora com 1024 dimensões...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        logger.info("Coleção criada com sucesso.")
    
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings 
    )

    # 3. Configuração do Chunker Teórico
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n\n", "\n\n", "\n", ".", " "]
    )

    # 4. Processamento iterativo
    for pdf_filename, config in BOOKS_CONFIG.items():
        pdf_path = BOOKS_DIR / pdf_filename
        
        if not pdf_path.exists():
            logger.warning(f"Ficheiro não encontrado, a saltar: {pdf_filename}")
            continue

        logger.info(f"A iniciar processamento do ficheiro: {pdf_filename}")
        
        try:
            raw_documents = extract_text_from_pdf(pdf_path, config)
            logger.info(f"[{pdf_filename}] Extração concluída. {len(raw_documents)} páginas úteis encontradas.")
            
            chunks = text_splitter.split_documents(raw_documents)
            logger.info(f"[{pdf_filename}] Dividido em {len(chunks)} fragmentos teóricos.")
            
            vector_store.add_documents(chunks)
            logger.info(f"[{pdf_filename}] Ingestão no Qdrant remoto efetuada com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro crítico ao processar o ficheiro {pdf_filename}: {e}")

    logger.info("Ingestão de todos os livros concluída.")

if __name__ == "__main__":
    main()