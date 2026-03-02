"""
ingest_standards.py — Pipeline de Ingestão de Normas Técnicas Petrobras
=======================================================================
Este módulo implementa a pipeline completa de ingestão de normas técnicas em formato PDF para um sistema de Retrieval-Augmented Generation (RAG), 
com persistência vetorial em um cluster remoto Qdrant. O pipeline realiza extração layout-aware do conteúdo, preservando a estrutura espacial
e semântica dos documentos, incluindo texto, tabelas e referências a imagens técnicas.
O processo utiliza um extrator especializado (StandardPDFExtractor) para capturar elementos estruturados e um chunker espacial (SpatialChunker) 
para segmentar o conteúdo em unidades semanticamente coerentes, mantendo vínculos com seções, páginas e coordenadas originais.
As imagens são extraídas e armazenadas localmente, com metadados associados ao chunk correspondente, permitindo futura expansão para processamento multimodal.
Cada chunk é enriquecido com metadados estruturados, incluindo identificadores únicos determinísticos, hashes de integridade, 
localização no documento e contexto semântico. Os embeddings são gerados por um modelo configurado e inseridos em lotes no Qdrant,
utilizando índices otimizados para filtragem eficiente por atributos como tipo de conteúdo, norma, página e presença de imagens.
O módulo inclui mecanismos de controle de integridade e consistência, como validação de cobertura textual, fallback automático para extração de imagens via PyMuPDF,
controle incremental baseado em hash MD5 e verificação interativa do payload antes da indexação. Esses recursos garantem alta fidelidade semântica, reprodutibilidade 
e robustez na ingestão de documentos técnicos complexos."""

import os
import sys
import logging
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.extractors import StandardPDFExtractor
from core.chunkers import SpatialChunker
from core.models import get_embeddings
from core.config import config

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

import fitz

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest_standards.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Ingestao_Normas")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "normas_tecnicas_publicas"
STANDARDS_DIR: Path = Path(r"C:\luza_datasets\Rag-pipeline-main\Data\normas_locais")
IMAGES_DIR: Path = Path(r"C:\luza_datasets\Rag-pipeline-main\Data\images_normas")

BATCH_SIZE: int = 64
MIN_CHUNK_CHARS: int = 15
COVERAGE_THRESHOLD: float = 0.85
IMAGE_FALLBACK_ENABLED: bool = True
STATE_FILE: Path = Path("ingest_state.json")

# [FIX 1] Limites para o fallback (devem espelhar StandardPDFExtractor)
FALLBACK_Y_MIN: float = 76.0
FALLBACK_Y_MAX: float = 805.0
FALLBACK_MIN_PX: int = 80


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def compute_file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def make_point_id(norma_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{norma_id}::{chunk_index}".encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def is_valid_chunk(content: str) -> bool:
    stripped = content.strip()
    if not stripped or len(stripped) < MIN_CHUNK_CHARS:
        return False
    return any(c.isalnum() or c in "°%/()[]{}\"'" for c in stripped)


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Validação de cobertura
# ---------------------------------------------------------------------------

def compute_coverage(pdf_path: str, documents: list[Document]) -> tuple[float, dict]:
    doc = fitz.open(pdf_path)
    chars_per_page: dict[int, int] = {}
    total_pdf_chars = 0
    for i in range(len(doc)):
        text = " ".join(doc[i].get_text("text").split())
        chars_per_page[i + 1] = len(text)
        total_pdf_chars += len(text)
    doc.close()

    if total_pdf_chars == 0:
        return 1.0, {}

    covered_per_page: dict[int, int] = {}
    for d in documents:
        pg = d.metadata.get("page_number", 0)
        covered_per_page[pg] = covered_per_page.get(pg, 0) + len(d.page_content)

    total_covered = sum(covered_per_page.values())
    pages_without_chunks = [
        pg for pg, chars in chars_per_page.items()
        if chars > 50 and covered_per_page.get(pg, 0) == 0
    ]
    coverage = min(total_covered / total_pdf_chars, 1.0)
    return coverage, {
        "total_pdf_chars":      total_pdf_chars,
        "total_covered_chars":  total_covered,
        "coverage_pct":         round(coverage * 100, 1),
        "pages_without_chunks": pages_without_chunks,
    }


# ---------------------------------------------------------------------------
# [FIX 1] Fallback de imagens com filtro correto de posição
# ---------------------------------------------------------------------------

def extract_image_chunks_fallback(
    pdf_path: str,
    images_dir: Path,
    norma_id: str,
    file_md5: str,
    start_index: int,
) -> tuple[list[Document], list[str]]:
    """
    Fallback via PyMuPDF — ativado apenas se o extrator principal não emitiu
    nenhum chunk image_ref.

    [FIX 1] Filtra imagens no cabeçalho (y1 <= FALLBACK_Y_MIN) e rodapé
    (y0 >= FALLBACK_Y_MAX) antes de qualquer processamento, eliminando a
    captura indevida do logo "-PÚBLICA- N-47" em todas as páginas.
    """
    documents: list[Document] = []
    point_ids: list[str] = []
    idx = start_index

    pdf_doc = fitz.open(pdf_path)
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                if not rects:
                    continue

                bbox = tuple(rects[0])
                x0, y0, x1, y1 = bbox

                # [FIX 1] Filtra cabeçalho e rodapé
                if y1 <= FALLBACK_Y_MIN or y0 >= FALLBACK_Y_MAX:
                    continue

                base_image = pdf_doc.extract_image(xref)
                img_w = base_image["width"]
                img_h = base_image["height"]

                if img_w < FALLBACK_MIN_PX or img_h < FALLBACK_MIN_PX:
                    continue

                img_filename = f"{norma_id}_p{page_num+1}_fb{img_index}.{base_image['ext']}"
                img_path = images_dir / img_filename
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])

                # Texto ao redor (abaixo do filtro de y_min)
                rect_surr = fitz.Rect(
                    0,
                    max(FALLBACK_Y_MIN, y0 - 65),
                    page.rect.width,
                    min(FALLBACK_Y_MAX, y1 + 65),
                )
                surrounding = " ".join(page.get_text("text", clip=rect_surr).split())

                content = (
                    f"[IMAGEM TÉCNICA — FALLBACK] {norma_id} — Página {page_num+1}. "
                    f"Contexto: {surrounding[:400] if surrounding else 'não capturado'}."
                )
                metadata = {
                    "source_type":       "norma_tecnica",
                    "norma_id":          norma_id,
                    "chunk_type":        "image_ref",
                    "page_number":       page_num + 1,
                    "has_image":         True,
                    "chunk_index":       idx,
                    "file_md5":          file_md5,
                    "ingested_at":       datetime.now(timezone.utc).isoformat(),
                    "image_path":        str(img_path),
                    "image_width":       img_w,
                    "image_height":      img_h,
                    "surrounding_text":  surrounding,
                    "image_type":        "raster",
                    "extraction_method": "pymupdf_fallback",
                }
                documents.append(Document(page_content=content, metadata=metadata))
                point_ids.append(make_point_id(norma_id, idx))
                idx += 1

            except Exception as e:
                logger.warning(f"  [Fallback] xref={xref} pág {page_num+1}: {e}")

    pdf_doc.close()
    return documents, point_ids


# ---------------------------------------------------------------------------
# Enriquecimento de metadados
# ---------------------------------------------------------------------------

def build_metadata(
    chunk: dict[str, Any],
    norma_id: str,
    chunk_index: int,
    file_md5: str,
) -> dict[str, Any]:
    raw: dict = chunk.get("metadata", {})
    chunk_type = raw.get("chunk_type", "text")
    if chunk_type not in {"text", "table", "image_ref"}:
        chunk_type = "text"

    metadata: dict[str, Any] = {
        "source_type":   "norma_tecnica",
        "norma_id":      norma_id,
        "chunk_type":    chunk_type,
        "page_number":   int(raw.get("page_number", 0)),
        "has_image":     chunk_type == "image_ref",
        "chunk_index":   chunk_index,
        "file_md5":      file_md5,
        "ingested_at":   datetime.now(timezone.utc).isoformat(),
        "section_title": raw.get("section_title", ""),
        "bounding_box":  raw.get("bounding_box"),
    }

    if chunk_type == "image_ref":
        metadata["image_path"]       = raw.get("image_path", "")
        metadata["image_width"]      = raw.get("image_width")
        metadata["image_height"]     = raw.get("image_height")
        metadata["surrounding_text"] = raw.get("surrounding_text", "")
        metadata["image_type"]       = raw.get("image_type", "raster")  # [FIX 2]

    return {k: v for k, v in metadata.items() if v is not None and v != ""}


# ---------------------------------------------------------------------------
# Índices de payload no Qdrant
# ---------------------------------------------------------------------------

def ensure_payload_indexes(client: QdrantClient) -> None:
    fields = {
        "norma_id":    PayloadSchemaType.KEYWORD,
        "chunk_type":  PayloadSchemaType.KEYWORD,
        "source_type": PayloadSchemaType.KEYWORD,
        "page_number": PayloadSchemaType.INTEGER,
        "has_image":   PayloadSchemaType.BOOL,
        "image_type":  PayloadSchemaType.KEYWORD,  # [FIX 2] índice para filtrar por tipo de imagem
    }
    for field, schema in fields.items():
        try:
            client.create_payload_index(COLLECTION_NAME, field, schema)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Upsert em lotes
# ---------------------------------------------------------------------------

def upsert_in_batches(
    vector_store: QdrantVectorStore,
    documents: list[Document],
    ids: list[str],
) -> int:
    total = 0
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_ids  = ids[i : i + BATCH_SIZE]
        try:
            vector_store.add_documents(batch_docs, ids=batch_ids)
            total += len(batch_docs)
            logger.debug(f"  Lote {i // BATCH_SIZE + 1}: {len(batch_docs)} OK.")
        except Exception as e:
            logger.error(f"  Falha no lote {i // BATCH_SIZE + 1}: {e}")
    return total


# ---------------------------------------------------------------------------
# Verificação interativa
# ---------------------------------------------------------------------------

def _print_payload_preview(
    documents: list[Document],
    point_ids: list[str],
    filename: str,
    coverage_details: dict,
) -> None:
    SEP_MAJOR = "═" * 72
    SEP_MINOR = "─" * 72

    print(f"\n{SEP_MAJOR}")
    print(f"  VERIFICAÇÃO DE PAYLOAD — {filename}")
    print(f"  Total de chunks prontos para inserção: {len(documents)}")
    print(SEP_MAJOR)

    cov_pct       = coverage_details.get("coverage_pct", 0)
    pdf_chars     = coverage_details.get("total_pdf_chars", 0)
    covered_chars = coverage_details.get("total_covered_chars", 0)
    missing_pages = coverage_details.get("pages_without_chunks", [])
    ok_icon       = "✅" if cov_pct >= COVERAGE_THRESHOLD * 100 else "⚠️ "

    print(f"\n  {ok_icon} COBERTURA DE TEXTO:")
    print(f"    PDF bruto      : {pdf_chars:>8,} chars")
    print(f"    Nos chunks     : {covered_chars:>8,} chars")
    print(f"    Cobertura      : {cov_pct:>7.1f}%  (mínimo: {COVERAGE_THRESHOLD*100:.0f}%)")

    if missing_pages:
        print(f"\n  PÁGINAS SEM NENHUM CHUNK: {missing_pages}")
    else:
        print(f"\n    Todas as páginas com conteúdo têm ao menos um chunk.")

    # Resumo por tipo
    type_counts: dict[str, int] = {}
    img_type_counts: dict[str, int] = {}
    for doc in documents:
        ct = doc.metadata.get("chunk_type", "text")
        type_counts[ct] = type_counts.get(ct, 0) + 1
        if ct == "image_ref":
            it = doc.metadata.get("image_type", "raster")
            img_type_counts[it] = img_type_counts.get(it, 0) + 1

    print(f"\n  RESUMO POR TIPO DE CHUNK:")
    for ct, count in sorted(type_counts.items()):
        label = {"text": "📄 text", "table": "📊 table", "image_ref": "🖼️  image_ref"}.get(ct, ct)
        print(f"    {label:<20} {count:>4}x   {'█' * min(count, 50)}")

    if img_type_counts:
        print(f"\n  SUBTIPO DE IMAGENS:")
        for it, count in sorted(img_type_counts.items()):
            icon = "🔷" if it == "vector_render" else "🔶"
            print(f"    {icon} {it:<18} {count:>4}x")

    if type_counts.get("image_ref", 0) == 0:
        print(f"\n  NENHUMA IMAGEM INDEXADA")

    print(f"\n{SEP_MINOR}")
    print(f"  DETALHES CHUNK A CHUNK:")
    print(SEP_MINOR)

    for i, (doc, pid) in enumerate(zip(documents, point_ids)):
        meta       = doc.metadata
        chunk_type = meta.get("chunk_type", "text")
        type_icon  = {
            "text":      "📄 TEXTO",
            "table":     "📊 TABELA",
            "image_ref": "🖼️  IMAGEM_REF",
        }.get(chunk_type, f"❓ {chunk_type.upper()}")

        print(f"\n  [{i+1:>4}/{len(documents)}]  {type_icon}")
        print(f"  {'─' * 68}")
        print(f"  {'point_id':<22} {pid}")
        print(f"  {'norma_id':<22} {meta.get('norma_id', 'N/A')}")
        print(f"  {'page_number':<22} {meta.get('page_number', 'N/A')}")
        print(f"  {'section_title':<22} {meta.get('section_title', '') or '(sem título)'}")
        print(f"  {'chunk_type':<22} {chunk_type}")
        print(f"  {'has_image':<22} {meta.get('has_image', False)}")
        print(f"  {'bounding_box':<22} {meta.get('bounding_box', 'N/A')}")
        print(f"  {'chunk_index':<22} {meta.get('chunk_index', 'N/A')}")
        print(f"  {'file_md5':<22} {meta.get('file_md5', 'N/A')[:16]}…")

        if chunk_type == "image_ref":
            img_type = meta.get("image_type", "raster")
            icon = "🔷" if img_type == "vector_render" else "🔶"
            print(f"  {'image_type':<22} {icon} {img_type}")
            print(f"  {'image_path':<22} {meta.get('image_path', '⚠️ AUSENTE')}")
            print(f"  {'image_size':<22} {meta.get('image_width', '?')}×{meta.get('image_height', '?')} px")
            surr = meta.get("surrounding_text", "")
            if surr:
                print(f"  {'surrounding_text':<22} \"{surr[:120].replace(chr(10), ' ')}{'…' if len(surr) > 120 else ''}\"")
            else:
                print(f"  {'surrounding_text':<22} ⚠️  VAZIO")

        content = doc.page_content
        preview = content[:200].replace("\n", " ").strip()
        if len(content) > 200:
            preview += "…"
        print(f"\n  CONTEÚDO ({len(content)} chars):")
        print(f"  \"{preview}\"")

    print(f"\n{SEP_MAJOR}\n")


def _ask_confirmation(coverage_ok: bool) -> bool:
    if not coverage_ok:
        print(f"  ╔══════════════════════════════════════════════════════════════╗")
        print(f"  ║  COBERTURA ABAIXO DE {COVERAGE_THRESHOLD*100:.0f}% — DADOS INCOMPLETOS      ║")
        print(f"  ╚══════════════════════════════════════════════════════════════╝\n")

    print("  Checklist antes de confirmar:")
    print("    • chunk_type correto em todos os chunks?")
    print("    • section_title preenchido nos chunks de texto?")
    print("    • image_ref com surrounding_text preenchido?")
    print("    • Figuras vetoriais (vector_render) nas páginas corretas?")
    print("    • Nenhum logo de cabeçalho nas imagens fallback?\n")

    while True:
        try:
            answer = input("  ➤  'sim' para inserir no Qdrant | 'nao' para cancelar: ")
            answer = answer.strip().lower()
            if answer in {"sim", "s", "yes", "y"}:
                print()
                logger.info("Verificação aprovada. Iniciando ingestão completa...")
                return True
            elif answer in {"nao", "não", "n", "no"}:
                print()
                logger.warning("Ingestão cancelada pelo usuário.")
                return False
            else:
                print("  Resposta inválida — use 'sim' ou 'nao'.")
        except (KeyboardInterrupt, EOFError):
            print()
            logger.warning("Ingestão interrompida (Ctrl+C).")
            return False


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 70)
    logger.info("PIPELINE DE INGESTÃO — NORMAS TÉCNICAS PETROBRAS")
    logger.info("=" * 70)

    if not STANDARDS_DIR.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {STANDARDS_DIR}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(STANDARDS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"Nenhum PDF encontrado em: {STANDARDS_DIR}")
        return

    logger.info(f"Normas encontradas: {len(pdf_files)}")

    logger.info("Carregando modelo de embeddings...")
    embeddings  = get_embeddings()
    vector_size = len(embeddings.embed_query("validação de dimensão"))
    logger.info(f"Dimensão do vetor: {vector_size}")

    qdrant_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        logger.info(f"Criando coleção '{COLLECTION_NAME}' (dim={vector_size}, COSINE)...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        logger.info(f"Coleção '{COLLECTION_NAME}' já existe — reutilizando.")

    ensure_payload_indexes(qdrant_client)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    chunker = SpatialChunker(target_chunk_size=1200, split_on_section_headers=True)
    state   = load_state()

    stats = {
        "total_normas":           len(pdf_files),
        "processadas":            0,
        "ignoradas_sem_mudanca":  0,
        "chunks_inseridos":       0,
        "chunks_descartados":     0,
        "imagens_fallback":       0,
        "cobertura_insuficiente": [],
        "erros":                  [],
    }

    first_norma_verified: bool = False

    for pdf_file in pdf_files:
        norma_id = pdf_file.stem
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Norma: {pdf_file.name}")

        current_md5 = compute_file_md5(pdf_file)
        if state.get(norma_id) == current_md5:
            logger.info("  ↳ Sem mudanças. Pulando.")
            stats["ignoradas_sem_mudanca"] += 1
            continue

        try:
            logger.info("  [1/5] Extração espacial...")
            extractor       = StandardPDFExtractor(str(pdf_file), str(IMAGES_DIR))
            extracted_pages = extractor.extract_all()
            logger.info(f"  ↳ {len(extracted_pages)} páginas extraídas.")

            logger.info("  [2/5] Chunking e vinculação espacial...")
            chunks_data = chunker.link_and_chunk(extracted_pages, norma_id)
            logger.info(f"  ↳ {len(chunks_data)} chunks brutos.")

            logger.info("  [3/5] Validação e enriquecimento de metadados...")
            documents:  list[Document] = []
            point_ids:  list[str]      = []
            discarded  = 0

            for idx, chunk in enumerate(chunks_data):
                content = chunk.get("content", "").strip()
                if not is_valid_chunk(content):
                    discarded += 1
                    continue
                meta = build_metadata(chunk, norma_id, idx, current_md5)
                documents.append(Document(page_content=content, metadata=meta))
                point_ids.append(make_point_id(norma_id, idx))

            stats["chunks_descartados"] += discarded

            # Fallback de imagens — só se extrator não emitiu nenhum image_ref
            n_image_chunks = sum(
                1 for d in documents if d.metadata.get("chunk_type") == "image_ref"
            )
            if IMAGE_FALLBACK_ENABLED and n_image_chunks == 0:
                logger.info("  ↳ Sem image_ref nativo — ativando fallback PyMuPDF...")
                fb_docs, fb_ids = extract_image_chunks_fallback(
                    str(pdf_file), IMAGES_DIR, norma_id, current_md5,
                    start_index=len(chunks_data),
                )
                if fb_docs:
                    documents.extend(fb_docs)
                    point_ids.extend(fb_ids)
                    stats["imagens_fallback"] += len(fb_docs)
                    logger.info(f"  ↳ Fallback: {len(fb_docs)} imagens capturadas.")

            n_images = sum(
                1 for d in documents if d.metadata.get("chunk_type") == "image_ref"
            )
            logger.info(
                f"  ↳ {len(documents)} chunks válidos | "
                f"{discarded} descartados | {n_images} imagens."
            )

            logger.info("  [4/5] Validando cobertura de texto...")
            coverage, cov_details = compute_coverage(str(pdf_file), documents)
            coverage_ok = coverage >= COVERAGE_THRESHOLD
            logger.info(
                f"  ↳ Cobertura: {cov_details.get('coverage_pct', 0):.1f}% "
                f"({'✅ OK' if coverage_ok else '⚠️ INSUFICIENTE'})"
            )
            if cov_details.get("pages_without_chunks"):
                logger.warning(f"  ↳ Páginas sem chunks: {cov_details['pages_without_chunks']}")
            if not coverage_ok:
                stats["cobertura_insuficiente"].append(
                    f"{norma_id}: {cov_details.get('coverage_pct', 0):.1f}%"
                )

            if not first_norma_verified:
                _print_payload_preview(documents, point_ids, pdf_file.name, cov_details)
                if not _ask_confirmation(coverage_ok):
                    return
                first_norma_verified = True

            if documents:
                logger.info(f"  [5/5] Enviando ao Qdrant (lotes de {BATCH_SIZE})...")
                inserted = upsert_in_batches(vector_store, documents, point_ids)
                stats["chunks_inseridos"] += inserted
                logger.info(f"  ↳ {inserted}/{len(documents)} chunks sincronizados.")
            else:
                logger.warning(f"  Nenhum chunk válido para {pdf_file.name}.")

            state[norma_id] = current_md5
            save_state(state)
            stats["processadas"] += 1

        except Exception as e:
            logger.error(f"  ERRO em {pdf_file.name}: {e}", exc_info=True)
            stats["erros"].append(f"{norma_id}: {type(e).__name__}: {e}")

    logger.info(f"\n{'=' * 70}")
    logger.info("RELATÓRIO FINAL")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Normas encontradas            : {stats['total_normas']}")
    logger.info(f"  Processadas com sucesso       : {stats['processadas']}")
    logger.info(f"  Ignoradas (sem mudanças)      : {stats['ignoradas_sem_mudanca']}")
    logger.info(f"  Chunks inseridos no Qdrant    : {stats['chunks_inseridos']}")
    logger.info(f"  Chunks descartados (ruído)    : {stats['chunks_descartados']}")
    logger.info(f"  Imagens via fallback          : {stats['imagens_fallback']}")
    if stats["cobertura_insuficiente"]:
        logger.warning(f"  Cobertura insuficiente ({len(stats['cobertura_insuficiente'])}):")
        for item in stats["cobertura_insuficiente"]:
            logger.warning(f"    - {item}")
    else:
        logger.info("  Cobertura: todas OK")
    if stats["erros"]:
        logger.error(f"  Erros ({len(stats['erros'])}):")
        for err in stats["erros"]:
            logger.error(f"    - {err}")
    else:
        logger.info("  Erros: nenhum")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
