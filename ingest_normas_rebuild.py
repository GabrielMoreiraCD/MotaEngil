# ingest_normas_rebuild.py
"""
Script de reingestão completa para normas técnicas Petrobras.
ATENÇÃO: apaga e recria a coleção normas_tecnicas_publicas.

Uso:
    python ingest_normas_rebuild.py --normas_dir ./data/normas --dry_run
    python ingest_normas_rebuild.py --normas_dir ./data/normas
"""
import argparse
import logging
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    FieldCondition, MatchValue, Filter
)
from sentence_transformers import SentenceTransformer

from core.extractors_normas import extract_norma_chunks
from core.chunkers_normas import chunk_norma_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

COLLECTION_NAME = "normas_tecnicas_publicas_v2"  # nova coleção — não sobrescreve a antiga
EMBEDDER_MODEL  = "BAAI/bge-m3"                  # MESMO embedder que petrobras_rag_teoria
VECTOR_DIM      = 1024
BATCH_SIZE      = 32

_RE_NORMA_FROM_FILENAME = re.compile(r'(N-\d{3,4}|NR-\d+)', re.IGNORECASE)

def norma_id_from_path(path: Path) -> str:
    m = _RE_NORMA_FROM_FILENAME.search(path.stem)
    return m.group(1).upper() if m else path.stem.upper()

def main(normas_dir: str, dry_run: bool):
    import os
    from dotenv import load_dotenv
    load_dotenv()

    qdrant_url    = os.environ["QDRANT_URL"]
    qdrant_apikey = os.environ.get("QDRANT_API_KEY")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_apikey)

    # 1. Embedder — BAAI/bge-m3 para ambas as coleções (consistência)
    log.info(f"Carregando embedder: {EMBEDDER_MODEL}")
    embedder = SentenceTransformer(EMBEDDER_MODEL)

    if not dry_run:
        # Recria coleção com payload indexado para filtros eficientes
        if client.collection_exists(COLLECTION_NAME):
            log.warning(f"Coleção {COLLECTION_NAME} existe — deletando para reingestão limpa")
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

        # Índices de payload para filtro O(log n) no RAG engine
        for field in ["norma_id", "secao", "tipo"]:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema="keyword",
            )
        log.info(f"Coleção {COLLECTION_NAME} criada com índices em norma_id/secao/tipo")

    # 2. Processa cada PDF
    normas_path = Path(normas_dir)
    pdf_files = sorted(normas_path.glob("**/*.pdf"))
    log.info(f"Encontrados {len(pdf_files)} PDFs em {normas_dir}")

    all_points: list[PointStruct] = []

    for pdf_path in pdf_files:
        norma_id = norma_id_from_path(pdf_path)
        log.info(f"Processando {pdf_path.name} → norma_id={norma_id}")

        try:
            raw_chunks = extract_norma_chunks(str(pdf_path), norma_id)
        except Exception as e:
            log.error(f"Falha ao extrair {pdf_path.name}: {e}")
            continue

        structured = chunk_norma_chunks(raw_chunks)
        log.info(f"  {len(raw_chunks)} blocos extraídos → {len(structured)} chunks estruturados")

        if dry_run:
            # Imprime amostra para validação antes de comitar
            for c in structured[:3]:
                print(f"\n  --- AMOSTRA [{norma_id}] secao={c['secao']} tipo={c['tipo']} ---")
                print(f"  EMBED_TEXT: {c['texto_embedding'][:200]}")
                print(f"  STORE_TEXT: {c['texto'][:200]}")
            continue

        # 3. Embed em batch usando texto_embedding (com contexto injetado)
        embed_texts = [c["texto_embedding"] for c in structured]

        for batch_start in range(0, len(embed_texts), BATCH_SIZE):
            batch_texts = embed_texts[batch_start : batch_start + BATCH_SIZE]
            batch_chunks = structured[batch_start : batch_start + BATCH_SIZE]

            vectors = embedder.encode(
                batch_texts,
                normalize_embeddings=True,  # cosine → produto interno equivalente
                show_progress_bar=False,
            ).tolist()

            for vec, chunk in zip(vectors, batch_chunks):
                payload = {k: v for k, v in chunk.items() if k != "texto_embedding"}
                # texto_embedding NÃO vai para o payload — economiza espaço
                all_points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload,
                ))

        log.info(f"  Total de pontos acumulados: {len(all_points)}")

    if dry_run:
        log.info(f"[DRY RUN] {len(all_points)} pontos seriam ingeridos (nenhuma escrita no Qdrant)")
        return

    # 4. Upsert em batches
    log.info(f"Iniciando upsert de {len(all_points)} pontos...")
    for i in range(0, len(all_points), 100):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=all_points[i : i + 100],
            wait=True,
        )
    log.info(f"✓ Ingestão completa. {len(all_points)} pontos em {COLLECTION_NAME}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normas_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    main(args.normas_dir, args.dry_run)