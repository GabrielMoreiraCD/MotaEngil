# ingest_livros_rebuild.py
"""
Reingestão completa de livros técnicos de referência para Qdrant.
Detecta automaticamente PDFs nativos vs rasterizados.
Cria coleção petrobras_rag_teoria_v2 (não destrói a original).

Uso:
    # Validação sem escrita:
    python ingest_livros_rebuild.py --livros_dir ./Data/books_ref/ --dry_run

    # Ingestão real:
    python ingest_livros_rebuild.py --livros_dir ./Data/books_ref/

    # Ignora PDFs majoritariamente em inglês:
    python ingest_livros_rebuild.py --livros_dir ./Data/books_ref/ --skip_english
"""

import argparse
import logging
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from core.extractors_livros import LivroChunk, extract_livro_chunks_with_ocr_fallback

# ─── Configuração ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

COLLECTION_NAME = "petrobras_rag_teoria_v2"
EMBEDDER_MODEL  = "BAAI/bge-m3"
VECTOR_DIM      = 1024
BATCH_SIZE      = 32    # chunks por batch de embedding
UPSERT_BATCH    = 100   # pontos por batch de upsert no Qdrant

CHUNK_MAX_WORDS = 350   # palavras máximas por chunk (sliding window)
CHUNK_OVERLAP   = 70    # palavras de sobreposição entre chunks adjacentes

# Mapeamento nome-de-arquivo → fonte_id legível
# Adicione entradas conforme necessário
FONTE_MAP: dict[str, str] = {
    "pedro_telles"          : "TELLES_TUBULACOES_INDUSTRIAIS",
    "tubulacoes"            : "TELLES_TUBULACOES_INDUSTRIAIS",
    "tubulac"               : "TELLES_TUBULACOES_INDUSTRIAIS",
    "pressure_vessel"       : "PRESSURE_VESSEL_DESIGN_MANUAL",
    "pressurevess"          : "PRESSURE_VESSEL_DESIGN_MANUAL",
    "damage_mechanism"      : "API_RP_571_DAMAGE_MECHANISMS",
    "damagemech"            : "API_RP_571_DAMAGE_MECHANISMS",
    "api_rp_571"            : "API_RP_571_DAMAGE_MECHANISMS",
    "nondestructive"        : "NDT_HANDBOOK",
    "ndt"                   : "NDT_HANDBOOK",
    "welding"               : "WELDING_HANDBOOK_AWS",
    "ciencia"               : "CIENCIA_ENGENHARIA_MATERIAIS",
    "materiais"             : "CIENCIA_ENGENHARIA_MATERIAIS",
}

_RE_NORMA_FILENAME = re.compile(r'(N-\d{3,4}|NR-\d+)', re.IGNORECASE)


def fonte_id_from_path(path: Path) -> str:
    """Resolve fonte_id a partir do nome do arquivo via FONTE_MAP."""
    stem_lower = path.stem.lower().replace(" ", "_").replace("-", "_")
    for key, val in FONTE_MAP.items():
        if key in stem_lower:
            return val
    # Fallback: usa stem em uppercase
    return re.sub(r'[^A-Z0-9_]', '', path.stem.upper())


# ─── Contextual embedding ─────────────────────────────────────────────────────

def build_contextual_text(chunk: LivroChunk) -> str:
    """
    Injeta contexto hierárquico no início do texto antes de embedar.
    O vetor resultante captura tanto a localização estrutural (capítulo)
    quanto o conteúdo semântico — aumenta recall em queries compostas.
    O prefixo NÃO é salvo no payload, apenas usado para embedding.
    """
    return (
        f"[Fonte: {chunk.fonte}] "
        f"[Capítulo: {chunk.capitulo}] "
        f"[Seção: {chunk.subtitulo}] "
        f"[Idioma: {chunk.idioma}]\n"
        f"{chunk.texto_limpo}"
    )


# ─── Sliding window ───────────────────────────────────────────────────────────

def sliding_window_split(chunk: LivroChunk) -> list[LivroChunk]:
    """
    Quebra chunks longos em janelas sobrepostas.
    Chunks <= CHUNK_MAX_WORDS palavras são retornados intactos.
    """
    words = chunk.texto_limpo.split()
    if len(words) <= CHUNK_MAX_WORDS:
        return [chunk]

    parts: list[LivroChunk] = []
    step = CHUNK_MAX_WORDS - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        window = " ".join(words[i : i + CHUNK_MAX_WORDS])
        if len(window) < 80:   # descarta janela residual muito pequena
            continue
        parts.append(LivroChunk(
            fonte=chunk.fonte,
            capitulo=chunk.capitulo,
            subtitulo=chunk.subtitulo,
            texto=chunk.texto,
            texto_limpo=window,
            pagina=chunk.pagina,
            idioma=chunk.idioma,
            qualidade_ocr=chunk.qualidade_ocr,
        ))
    return parts if parts else [chunk]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(livros_dir: str, dry_run: bool, skip_english: bool) -> None:
    load_dotenv()

    qdrant_url    = os.environ["QDRANT_URL"]
    qdrant_apikey = os.environ.get("QDRANT_API_KEY")

    client   = QdrantClient(url=qdrant_url, api_key=qdrant_apikey)
    embedder = SentenceTransformer(EMBEDDER_MODEL)

    # ── Cria coleção (somente em modo real) ───────────────────────────────────
    if not dry_run:
        if client.collection_exists(COLLECTION_NAME):
            log.warning(f"Coleção {COLLECTION_NAME} existe — deletando para reingestão limpa")
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

        # Índices de payload para filtros eficientes no RAG engine
        for field_name, schema in [
            ("fonte",         "keyword"),
            ("capitulo",      "keyword"),
            ("idioma",        "keyword"),
            ("qualidade_ocr", "float"),
        ]:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
            )
        log.info(f"Coleção {COLLECTION_NAME} criada com índices de payload")

    # ── Processa PDFs ─────────────────────────────────────────────────────────
    pdf_files = sorted(Path(livros_dir).glob("**/*.pdf"))
    log.info(f"{len(pdf_files)} PDFs encontrados em {livros_dir}")

    stats = {
        "total_raw"       : 0,
        "descartados_en"  : 0,
        "final_chunks"    : 0,
        "pontos_ingeridos": 0,
    }

    all_points: list[PointStruct] = []

    for pdf_path in pdf_files:
        fonte_id = fonte_id_from_path(pdf_path)
        log.info(f"Processando: {pdf_path.name} → fonte_id={fonte_id}")

        # Extração com fallback OCR automático
        try:
            raw_chunks = extract_livro_chunks_with_ocr_fallback(
                str(pdf_path),
                fonte_id,
                ocr_lang="por+eng",
            )
        except Exception as exc:
            log.error(f"  ERRO ao extrair {pdf_path.name}: {exc}")
            continue

        stats["total_raw"] += len(raw_chunks)

        if not raw_chunks:
            log.warning(f"  0 chunks extraídos — PDF pode estar corrompido ou vazio")
            continue

        # Filtro de idioma: pula PDFs majoritariamente em inglês se solicitado
        if skip_english:
            en_count = sum(1 for c in raw_chunks if c.idioma == "en")
            en_ratio = en_count / len(raw_chunks)
            if en_ratio > 0.60:
                log.warning(
                    f"  {pdf_path.name}: {en_ratio:.0%} chunks em EN "
                    f"— pulando (--skip_english ativo)"
                )
                stats["descartados_en"] += len(raw_chunks)
                continue

        # Sliding window
        final_chunks: list[LivroChunk] = []
        for rc in raw_chunks:
            final_chunks.extend(sliding_window_split(rc))

        stats["final_chunks"] += len(final_chunks)
        log.info(
            f"  {len(raw_chunks)} blocos extraídos "
            f"→ {len(final_chunks)} chunks após sliding window"
        )

        # Dry run: imprime estatísticas de qualidade OCR e amostra
        if dry_run:
            scores = [c.qualidade_ocr for c in final_chunks]
            avg_q  = sum(scores) / len(scores) if scores else 0.0
            below  = sum(1 for s in scores if s < 0.45)
            en_ct  = sum(1 for c in final_chunks if c.idioma == "en")
            pt_ct  = len(final_chunks) - en_ct

            print(f"\n  ─── [{fonte_id}] ───")
            print(f"  Chunks finais      : {len(final_chunks)}")
            print(f"  Qualidade OCR média: {avg_q:.3f}")
            print(f"  Abaixo threshold   : {below}/{len(final_chunks)}")
            print(f"  Idioma PT / EN     : {pt_ct} / {en_ct}")
            if final_chunks:
                amostra = final_chunks[0].texto_limpo[:250].replace('\n', ' ')
                print(f"  Amostra chunk[0]   : {amostra}...")
            continue

        # ── Embedding em batches ──────────────────────────────────────────────
        embed_texts = [build_contextual_text(c) for c in final_chunks]

        for batch_start in range(0, len(embed_texts), BATCH_SIZE):
            batch_texts  = embed_texts[batch_start : batch_start + BATCH_SIZE]
            batch_chunks = final_chunks[batch_start : batch_start + BATCH_SIZE]

            vectors = embedder.encode(
                batch_texts,
                normalize_embeddings=True,   # cosine distance = produto interno
                show_progress_bar=False,
            ).tolist()

            for vec, chunk in zip(vectors, batch_chunks):
                all_points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "texto"       : chunk.texto_limpo,
                        "fonte"       : chunk.fonte,
                        "capitulo"    : chunk.capitulo,
                        "subtitulo"   : chunk.subtitulo,
                        "pagina"      : chunk.pagina,
                        "idioma"      : chunk.idioma,
                        "qualidade_ocr": chunk.qualidade_ocr,
                    },
                ))

        log.info(f"  Pontos acumulados até agora: {len(all_points)}")

    # ── Dry run: sumário final ────────────────────────────────────────────────
    if dry_run:
        print(f"\n{'='*60}")
        print(f"[DRY RUN] Sumário final:")
        print(f"  PDFs processados    : {len(pdf_files)}")
        print(f"  Chunks brutos       : {stats['total_raw']}")
        print(f"  Chunks finais       : {stats['final_chunks']}")
        print(f"  Descartados (EN)    : {stats['descartados_en']}")
        print(f"  Seriam ingeridos    : {stats['final_chunks']}")
        print(f"  Coleção destino     : {COLLECTION_NAME}")
        print(f"{'='*60}")
        return

    # ── Upsert em batches ─────────────────────────────────────────────────────
    log.info(f"Iniciando upsert de {len(all_points)} pontos em {COLLECTION_NAME}...")

    for i in range(0, len(all_points), UPSERT_BATCH):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=all_points[i : i + UPSERT_BATCH],
            wait=True,
        )
        if (i // UPSERT_BATCH) % 10 == 0:
            log.info(f"  Upsert: {min(i + UPSERT_BATCH, len(all_points))}/{len(all_points)}")

    stats["pontos_ingeridos"] = len(all_points)
    log.info(f"✓ Ingestão concluída. Stats: {stats}")
    log.info(f"✓ Coleção: {COLLECTION_NAME} | Pontos: {len(all_points)}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reingestão de livros técnicos no Qdrant com fallback OCR"
    )
    parser.add_argument(
        "--livros_dir",
        required=True,
        help="Diretório contendo os PDFs dos livros de referência",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Valida extração sem escrever no Qdrant",
    )
    parser.add_argument(
        "--skip_english",
        action="store_true",
        help="Ignora PDFs com > 60%% de chunks em inglês",
    )
    args = parser.parse_args()
    main(args.livros_dir, args.dry_run, args.skip_english)