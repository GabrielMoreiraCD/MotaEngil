# ingest_materials.py
"""
Ingestão de catálogos de materiais Petrobras (219 XLSX) → Qdrant Cloud.
Coleção destino: catalogo_materiais_v1

ESTRATÉGIA DE SCHEMA HETEROGÊNEO:
  Cada XLSX tem colunas diferentes. O módulo normaliza via mapa de aliases
  (COLUMN_ALIASES) que cobre sinônimos conhecidos do padrão Petrobras.
  Colunas sem match são preservadas com prefixo 'extra_' no payload.
  O texto de embedding é construído priorizando campos técnicos canônicos
  (descricao, especificacao, diametro, pressao) e usa todos os valores
  da linha como fallback se nenhum campo canônico for encontrado.

PADRÃO DE ID: SHA256 determinístico = {stem_do_arquivo}::{indice_da_linha}
  Garante idempotência: re-ingerir o mesmo XLSX não cria duplicatas.

USO:
  # Validação sem escrita no Qdrant:
  python ingest_materials.py --materiais_dir ./Data/Materiais --dry_run

  # Ingestão real:
  python ingest_materials.py --materiais_dir ./Data/Materiais

  # Reprocessa mesmo sem mudança de MD5:
  python ingest_materials.py --materiais_dir ./Data/Materiais --force

  # Limpa a coleção antes de reingerir:
  python ingest_materials.py --materiais_dir ./Data/Materiais --recreate_collection
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Logging — mesmo padrão de ingest_standards.py (stdout + arquivo)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest_materials.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("Ingestao_Materiais")

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "catalogo_materiais_v1"
EMBEDDER_MODEL  = "BAAI/bge-m3"
VECTOR_DIM      = 1024
BATCH_EMBED     = 32    # linhas por batch de embedding (RAM bound para bge-m3 1024-dim)
BATCH_UPSERT    = 100   # pontos por upsert no Qdrant
STATE_FILE      = Path("ingest_materials_state.json")
REPORT_FILE     = Path("ingest_materials_report.json")

# ─────────────────────────────────────────────────────────────────────────────
# MAPA DE ALIASES → CAMPO CANÔNICO
#
# Cada chave é o nome canônico interno.
# Cada lista contém todos os sinônimos conhecidos no padrão Petrobras.
# Ordem de prioridade: alias mais específico deve vir primeiro.
# Adicione novos aliases conforme XLSXs com schema desconhecido forem encontrados.
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES: dict[str, list[str]] = {
    # ── Identificação ─────────────────────────────────────────────────────────
    "codigo": [
        "cod_material", "cod_mat", "codigo_material", "código_material",
        "codigo", "código", "cod.", "part_number", "part number",
        "referencia", "referência", "ref.", "num_item", "numero_item",
        "n_item", "item",
    ],
    "descricao": [
        "denominacao", "denominação", "descricao_completa", "descrição_completa",
        "descrição", "descricao", "description", "desc.", "denominac",
        "nome_material", "nome", "produto", "especif_resumida",
        "designacao", "designação",
    ],
    "especificacao": [
        "especificacao_tecnica", "especificação_técnica",
        "especificacao", "especificação", "especificacoes", "especificações",
        "spec", "norma_aplicavel", "norma_aplicável", "norma_fabricacao",
        "norma_fabricação", "standard", "revision", "especif",
    ],
    # ── Dimensões ─────────────────────────────────────────────────────────────
    "diametro": [
        "diametro_nominal", "diâmetro_nominal", "diametro", "diâmetro",
        "dn", "nps", "nominal_pipe_size", "nominal pipe size",
        "diametro_externo", "diâmetro_externo", "diam", "bitola", "polegadas",
    ],
    "espessura": [
        "espessura_parede", "espessura_da_parede", "espessura", "thickness",
        "sch", "schedule", "wall_thickness", "esp",
    ],
    "comprimento": [
        "comprimento", "length", "comp.", "compr",
    ],
    # ── Condições de operação ─────────────────────────────────────────────────
    "pressao": [
        "classe_de_pressao", "classe de pressão", "pressao_maxima",
        "pressão_máxima", "pressao", "pressão", "pressure", "rating",
        "pn", "class", "pressure_class", "classe_pressao",
    ],
    "temperatura": [
        "temperatura_maxima", "temperatura_max", "temp_max", "temperatura",
        "temperature", "temp", "t_max", "faixa_temperatura",
    ],
    # ── Material e metalurgia ─────────────────────────────────────────────────
    "material_base": [
        "material_de_construcao", "material_construcao", "material_construção",
        "material_base", "material base", "liga", "alloy",
        "composicao", "composição", "matl", "material", "grau_material",
        "grau", "grade",
    ],
    # ── Conexões ─────────────────────────────────────────────────────────────
    "conexao": [
        "tipo_extremidade", "tipo_de_extremidade", "end_connection",
        "end connection", "tipo_conexao", "tipo_de_conexão",
        "conexao", "conexão", "extremidades", "acabamento",
    ],
    "face_flange": [
        "face_flange", "face_do_flange", "face", "acabamento_face",
        "flange_facing", "rf", "rtj", "ff",
    ],
    # ── Normas ───────────────────────────────────────────────────────────────
    "norma": [
        "norma_projeto", "norma_de_projeto", "norma_referencia",
        "norma_de_referência", "norma", "standard_aplicavel",
        "asme", "api", "astm", "codigo_projeto", "codigo_de_projeto",
    ],
    # ── Fornecimento ─────────────────────────────────────────────────────────
    "unidade": [
        "unidade_fornecimento", "unidade_de_fornecimento", "und_forn",
        "un", "und", "unidade", "unit", "uom", "um",
    ],
    "fabricante": [
        "fabricante_aprovado", "fabricante_homologado", "fabricante",
        "manufacturer", "fornecedor", "supplier", "marca", "brand",
    ],
    # ── Notas ────────────────────────────────────────────────────────────────
    "observacao": [
        "observacoes", "observações", "observacao", "observação",
        "obs", "nota", "notes", "comentario", "comentário",
        "remarks", "notas_complementares",
    ],
}

# Campos em ordem de prioridade para o texto de embedding.
# Lógica: dimensões técnicas primeiro → dominam o espaço semântico →
# queries como "curva 2" DN 2" ASTM A234 GrB SW" acertam o item correto.
EMBED_PRIORITY: list[str] = [
    "categoria",        # nome do arquivo = tipo de material
    "descricao",
    "especificacao",
    "codigo",
    "material_base",
    "diametro",
    "espessura",
    "pressao",
    "norma",
    "conexao",
    "temperatura",
    "comprimento",
    "face_flange",
    "observacao",
]

# Campos canônicos que vão para o payload (e são indexados no Qdrant)
PAYLOAD_CANONICAL: list[str] = [
    "categoria", "codigo", "descricao", "especificacao",
    "diametro", "espessura", "pressao", "material_base",
    "norma", "unidade", "fabricante", "temperatura",
    "conexao", "face_flange", "observacao",
]

# Valores que representam "vazio" em diferentes convenções
_EMPTY_SENTINELS = {"", "nan", "none", "n/a", "na", "-", "--", "s/i", "si", "nd"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses de relatório (para auditoria e rastreabilidade)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileReport:
    filename: str
    status: str          # "ok" | "fail" | "empty" | "skipped"
    rows_total: int = 0
    rows_indexed: int = 0
    rows_skipped: int = 0
    error: str = ""
    columns_mapped: dict = field(default_factory=dict)


@dataclass
class IngestReport:
    collection: str = COLLECTION_NAME
    embedder: str = EMBEDDER_MODEL
    timestamp: str = ""
    dry_run: bool = False
    total_files: int = 0
    ok: int = 0
    empty: int = 0
    failed: int = 0
    skipped: int = 0
    total_points: int = 0
    duration_seconds: float = 0.0
    files: list = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def compute_md5(path: Path) -> str:
    """MD5 do arquivo para detecção de mudança (skip incremental)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def make_point_id(file_stem: str, row_index: int) -> str:
    """
    ID SHA256 determinístico por (arquivo, linha).
    Garante idempotência: upsert de mesmo ponto não cria duplicata no Qdrant.
    Formato: 8-4-4-4-12 (UUID-like, mas deterministicamente derivado).
    """
    digest = hashlib.sha256(f"{file_stem}::{row_index}".encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def load_state(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Normalização de colunas (schema agnóstico)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_str(s: str) -> str:
    """
    Remove acentos e caracteres especiais para comparação de alias.
    Não usa unicodedata para evitar dependência — substituições manuais
    cobrem o universo do português + inglês técnico.
    """
    s = s.strip().lower()
    for src, tgt in [
        ("ã", "a"), ("â", "a"), ("á", "a"), ("à", "a"),
        ("ê", "e"), ("é", "e"), ("è", "e"),
        ("î", "i"), ("í", "i"),
        ("ô", "o"), ("ó", "o"), ("õ", "o"), ("ò", "o"),
        ("û", "u"), ("ú", "u"), ("ù", "u"),
        ("ç", "c"),
    ]:
        s = s.replace(src, tgt)
    return re.sub(r"[^a-z0-9]", "_", s)


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Mapeia colunas reais do DataFrame → schema canônico via alias.

    Retorna:
        (df_renomeado, {nome_original: nome_canônico})

    Colunas sem match recebem prefixo 'extra_' — dados não se perdem.
    Um canonical só é usado uma vez (first-match wins).

    Complexidade: O(C × A) onde C = colunas, A = total de aliases.
    Com 30 colunas × 120 aliases = 3600 comparações → negligível.
    """
    rename_map: dict[str, str] = {}
    canonical_used: set[str] = set()

    for col in df.columns:
        col_clean = _normalize_str(str(col))
        matched = False

        for canonical, aliases in COLUMN_ALIASES.items():
            if canonical in canonical_used:
                continue
            for alias in aliases:
                alias_clean = _normalize_str(alias)
                # Match exato OU alias está contido no nome da coluna real
                if alias_clean == col_clean or alias_clean in col_clean:
                    rename_map[col] = canonical
                    canonical_used.add(canonical)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            safe = _normalize_str(str(col))[:40].strip("_") or f"col_{len(rename_map)}"
            rename_map[col] = f"extra_{safe}"

    return df.rename(columns=rename_map), rename_map


# ─────────────────────────────────────────────────────────────────────────────
# Construção do texto de embedding
# ─────────────────────────────────────────────────────────────────────────────

def _is_empty_val(val: str) -> bool:
    return val.strip().lower() in _EMPTY_SENTINELS


def build_embed_text(row: dict, categoria: str) -> str:
    """
    Constrói o texto de embedding para uma linha do XLSX.

    Design rationale:
    - Campos canônicos em EMBED_PRIORITY: concentra dimensões técnicas
      no início do string → exploits positional bias do attention do bge-m3.
      Um item "CURVA DE AÇO | DESCRICAO: Curva 90° BW | DIAMETRO: 2" | 
      PRESSAO: SCH 80 | MATERIAL BASE: ASTM A234 GrWPB" terá alta similaridade
      com a query "curva 90 graus 2 polegadas SCH80 carbono".
    - Fallback full-row: cobre XLSXs com schema 100% desconhecido.
    - Limite de 20 campos no fallback: evita inputs que extrapolam
      o contexto do tokenizer do bge-m3 (max 8192 tokens).
    """
    row_with_cat = {"categoria": categoria, **row}
    parts: list[str] = []

    for field_name in EMBED_PRIORITY:
        val = str(row_with_cat.get(field_name, ""))
        if not _is_empty_val(val):
            label = field_name.upper().replace("_", " ")
            parts.append(f"{label}: {val.strip()}")

    if not parts:
        # Fallback: campos extra_ + qualquer valor disponível
        extra_parts = [
            f"{k.replace('extra_', '').upper()}: {str(v).strip()}"
            for k, v in row.items()
            if not _is_empty_val(str(v)) and len(str(v).strip()) > 2
        ]
        parts = extra_parts[:20]

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Leitura robusta do XLSX
# ─────────────────────────────────────────────────────────────────────────────

def read_xlsx_robust(path: Path) -> pd.DataFrame:
    """
    Lê XLSX com openpyxl.

    Limpeza aplicada:
    1. Remove colunas "Unnamed: N" (colunas sem header no XLSX)
    2. Remove linhas completamente vazias
    3. Preenche NaN com "" para serialização segura no payload Qdrant
    4. Converte tudo para str — evita erros de tipo float no Qdrant payload

    Não usa xlrd: xlrd >= 2.0 não suporta formato xlsx (apenas xls legado).
    Arquivo corrompido ou protegido por senha → RuntimeError com mensagem clara.
    """
    try:
        df = pd.read_excel(path, engine="openpyxl", dtype=str, header=0)
    except Exception as e:
        raise RuntimeError(
            f"Falha ao abrir '{path.name}' com openpyxl: {e}. "
            "Verifique se o arquivo não está corrompido ou protegido por senha."
        ) from e

    # Remove colunas sem header real
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+$")]

    # Remove linhas totalmente vazias
    df = df.dropna(how="all").reset_index(drop=True)

    # Preenche NaN e converte para string
    df = df.fillna("").astype(str)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Setup Qdrant (idempotente)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_collection(
    client: QdrantClient, name: str, dim: int, recreate: bool
) -> None:
    """
    Cria a collection se não existir.
    Se recreate=True deleta e recria (reingestão limpa).
    Idempotente quando recreate=False.
    """
    exists = client.collection_exists(name)

    if exists and recreate:
        log.warning(f"[Qdrant] Deletando '{name}' para reingestão limpa (--recreate_collection)")
        client.delete_collection(name)
        exists = False

    if not exists:
        log.info(f"[Qdrant] Criando '{name}' (dim={dim}, COSINE)...")
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info(f"[Qdrant] Coleção '{name}' criada.")

    _ensure_payload_indexes(client, name)


def _ensure_payload_indexes(client: QdrantClient, name: str) -> None:
    """
    Cria índices keyword nos campos usados em filtros do RAG engine.
    Idempotente: ignora "already exists".

    Sem índice → Qdrant faz full-scan O(N) no payload.
    Com índice → O(log N). Crítico para coleção com >100k pontos.
    """
    keyword_fields = [
        "categoria",      # filtro por tipo de material (ex: "VÁLVULA ESFERA METÁLICA")
        "codigo",         # lookup por código Petrobras
        "diametro",       # filtro por DN/NPS
        "pressao",        # filtro por classe de pressão / rating
        "material_base",  # filtro por liga metálica (ex: "ASTM A105")
        "norma",          # filtro por norma de referência (ex: "ASME B16.5")
        "source_file",    # rastreabilidade até o arquivo XLSX de origem
    ]
    for field_name in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            err_lower = str(e).lower()
            if "already exists" in err_lower or "conflict" in err_lower:
                pass  # Esperado em coleção pré-existente — não é erro
            else:
                log.warning(f"[Qdrant] Índice '{field_name}' não criado: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline por arquivo XLSX
# ─────────────────────────────────────────────────────────────────────────────

def process_xlsx(
    xlsx_path: Path,
    embedder: SentenceTransformer,
    client: QdrantClient,
    dry_run: bool,
) -> FileReport:
    """
    Pipeline completo para um único XLSX:
        lê → normaliza colunas → constrói textos → embeda em batch → upsert

    Retorna FileReport com métricas para auditoria.
    """
    report = FileReport(filename=xlsx_path.name, status="fail")
    categoria = xlsx_path.stem  # ex: "VÁLVULA ESFERA METÁLICA"

    # ── 1. Leitura ────────────────────────────────────────────────────────────
    try:
        df = read_xlsx_robust(xlsx_path)
    except RuntimeError as e:
        report.error = str(e)
        log.error(f"  [LEITURA] {e}")
        return report

    report.rows_total = len(df)

    if df.empty:
        report.status = "empty"
        log.warning("  DataFrame vazio após limpeza — sem dados válidos.")
        return report

    # ── 2. Normalização de colunas ────────────────────────────────────────────
    df, col_map = normalize_columns(df)
    report.columns_mapped = col_map

    mapped_canonical = {v: k for k, v in col_map.items() if not v.startswith("extra_")}
    log.info(f"  Colunas canônicas: {sorted(mapped_canonical.keys())}")

    if not mapped_canonical:
        log.warning("  Nenhuma coluna canônica mapeada — usando fallback full-row para embedding.")

    # ── 3. Construção de pontos ───────────────────────────────────────────────
    points: list[PointStruct] = []
    rows_skipped = 0

    for row_idx, row in enumerate(df.to_dict(orient="records")):
        embed_text = build_embed_text(row, categoria)

        if not embed_text.strip():
            rows_skipped += 1
            continue

        # Payload: campos canônicos disponíveis + extras + metadados de origem
        payload: dict = {
            "source_type": "material_catalog",
            "source_file": xlsx_path.name,
            "categoria":   categoria,
            "row_index":   row_idx,
            "embed_text":  embed_text,  # mantido para debug e auditoria
        }

        for field_name in PAYLOAD_CANONICAL:
            val = str(row.get(field_name, "")).strip()
            if not _is_empty_val(val):
                payload[field_name] = val

        # Preserva campos não mapeados com prefixo extra_
        for k, v in row.items():
            if k.startswith("extra_") and not _is_empty_val(str(v)):
                payload[k] = str(v).strip()

        point_id = make_point_id(xlsx_path.stem, row_idx)
        # vector preenchido no batch abaixo — PointStruct aceita lista vazia temporariamente
        points.append(PointStruct(id=point_id, vector=[], payload=payload))

    if not points:
        report.status = "empty"
        report.rows_skipped = rows_skipped
        log.warning("  0 pontos válidos gerados após filtragem de linhas vazias.")
        return report

    # ── 4. Embedding em batches ───────────────────────────────────────────────
    embed_texts = [p.payload["embed_text"] for p in points]

    for batch_start in range(0, len(embed_texts), BATCH_EMBED):
        batch_texts = embed_texts[batch_start : batch_start + BATCH_EMBED]
        vectors = embedder.encode(
            batch_texts,
            normalize_embeddings=True,   # L2-norm → cosine = produto interno
            show_progress_bar=False,
            batch_size=BATCH_EMBED,
        ).tolist()

        for i, vec in enumerate(vectors):
            points[batch_start + i].vector = vec

    # ── 5. Upsert no Qdrant ───────────────────────────────────────────────────
    if not dry_run:
        for batch_start in range(0, len(points), BATCH_UPSERT):
            batch = points[batch_start : batch_start + BATCH_UPSERT]
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True,
            )

    report.status = "ok"
    report.rows_indexed = len(points)
    report.rows_skipped = rows_skipped
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Dry run: amostra sem escrever no Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def _print_dry_run_sample(xlsx_path: Path) -> None:
    """Exibe estatísticas e 3 linhas de amostra para validação visual."""
    try:
        df = read_xlsx_robust(xlsx_path)
        df, col_map = normalize_columns(df)
        categoria = xlsx_path.stem
        mapped = [v for v in col_map.values() if not v.startswith("extra_")]

        print(f"\n  ─── [{categoria}] ───")
        print(f"  Total de linhas      : {len(df)}")
        print(f"  Colunas canônicas    : {sorted(set(mapped))}")
        print(f"  Colunas extra        : {sum(1 for v in col_map.values() if v.startswith('extra_'))}")

        for i, row in enumerate(df.head(3).to_dict(orient="records")):
            text = build_embed_text(row, categoria)
            preview = text[:200] + ("..." if len(text) > 200 else "")
            print(f"  Linha {i} embed_text  : {preview}")

    except Exception as e:
        print(f"  [ERRO ao amostrar '{xlsx_path.name}'] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    materiais_dir: str,
    dry_run: bool,
    force: bool,
    recreate_collection: bool,
) -> None:
    t_start = time.time()
    load_dotenv()

    qdrant_url    = os.environ.get("QDRANT_URL", "").strip()
    qdrant_apikey = os.environ.get("QDRANT_API_KEY", "").strip()

    if not qdrant_url or not qdrant_apikey:
        log.error("QDRANT_URL e QDRANT_API_KEY são obrigatórios no .env")
        sys.exit(1)

    # ── Embedder ──────────────────────────────────────────────────────────────
    log.info(f"Carregando embedder: {EMBEDDER_MODEL}")
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    log.info("Embedder pronto.")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    client = QdrantClient(url=qdrant_url, api_key=qdrant_apikey)

    if not dry_run:
        ensure_collection(client, COLLECTION_NAME, VECTOR_DIM, recreate_collection)

    # ── State incremental ─────────────────────────────────────────────────────
    state = load_state(STATE_FILE)

    # ── Scan dos XLSXs (case-insensitive: .xlsx e .XLSX) ─────────────────────
    materiais_path = Path(materiais_dir)
    all_xlsx = sorted(materiais_path.glob("*.xlsx")) + sorted(materiais_path.glob("*.XLSX"))
    seen: set[str] = set()
    xlsx_files: list[Path] = []
    for f in all_xlsx:
        if f.name not in seen:
            seen.add(f.name)
            xlsx_files.append(f)

    log.info("=" * 70)
    log.info("PIPELINE DE INGESTÃO — CATÁLOGO DE MATERIAIS PETROBRAS")
    log.info("=" * 70)
    log.info(f"  Diretório    : {materiais_path.resolve()}")
    log.info(f"  Arquivos     : {len(xlsx_files)} XLSXs")
    log.info(f"  Coleção      : {COLLECTION_NAME}")
    log.info(f"  Embedder     : {EMBEDDER_MODEL}")
    log.info(f"  Modo         : {'DRY RUN — sem escrita no Qdrant' if dry_run else 'PRODUÇÃO'}")
    log.info("=" * 70)

    # ── Relatório ─────────────────────────────────────────────────────────────
    ingest_report = IngestReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dry_run=dry_run,
        total_files=len(xlsx_files),
    )

    # ── Loop principal ────────────────────────────────────────────────────────
    for xlsx_path in xlsx_files:
        log.info(f"\n[{xlsx_path.name}]")

        # Skip incremental por MD5
        current_md5 = compute_md5(xlsx_path)
        if not force and state.get(xlsx_path.name) == current_md5:
            log.info("  ↳ Sem mudanças desde a última ingestão. Pulando. (use --force para forçar)")
            file_report = FileReport(filename=xlsx_path.name, status="skipped")
            ingest_report.files.append(asdict(file_report))
            ingest_report.skipped += 1
            continue

        if dry_run:
            _print_dry_run_sample(xlsx_path)
            ingest_report.ok += 1
            continue

        # Processamento real
        try:
            file_report = process_xlsx(xlsx_path, embedder, client, dry_run=False)
        except Exception as e:
            # Captura exceções não previstas dentro do processo (defensivo)
            log.error(f"  ERRO inesperado em '{xlsx_path.name}': {e}", exc_info=True)
            file_report = FileReport(
                filename=xlsx_path.name,
                status="fail",
                error=f"{type(e).__name__}: {e}",
            )

        ingest_report.files.append(asdict(file_report))

        if file_report.status == "ok":
            ingest_report.ok += 1
            ingest_report.total_points += file_report.rows_indexed
            state[xlsx_path.name] = current_md5
            save_state(state, STATE_FILE)
            log.info(
                f"  ✓ {file_report.rows_indexed} itens indexados | "
                f"{file_report.rows_skipped} linhas ignoradas (vazias)"
            )
        elif file_report.status == "empty":
            ingest_report.empty += 1
            log.warning("  ⚠ Arquivo vazio ou sem dados válidos.")
        else:
            ingest_report.failed += 1
            log.error(f"  ✗ Falha: {file_report.error}")

    # ── Relatório final ───────────────────────────────────────────────────────
    ingest_report.duration_seconds = round(time.time() - t_start, 2)

    log.info("\n" + "=" * 70)
    log.info("RELATÓRIO FINAL")
    log.info("=" * 70)
    log.info(f"  Arquivos encontrados   : {ingest_report.total_files}")
    log.info(f"  Ingeridos com sucesso  : {ingest_report.ok}")
    log.info(f"  Pulados (sem mudança)  : {ingest_report.skipped}")
    log.info(f"  Vazios                 : {ingest_report.empty}")
    log.info(f"  Falhas                 : {ingest_report.failed}")
    log.info(f"  Total de pontos Qdrant : {ingest_report.total_points}")
    log.info(f"  Duração total          : {ingest_report.duration_seconds}s")
    log.info(f"  Coleção destino        : {COLLECTION_NAME}")

    if not dry_run:
        ingest_report.to_json(REPORT_FILE)
        log.info(f"  Relatório JSON         : {REPORT_FILE}")
        log.info(f"  State file             : {STATE_FILE}")

    if ingest_report.failed > 0:
        log.error(f"\n  ⚠  {ingest_report.failed} arquivo(s) falharam. Verifique ingest_materials.log.")

    if dry_run:
        log.info("\n  [DRY RUN] Nenhuma escrita foi feita no Qdrant.")

    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingestão de catálogos XLSX de materiais Petrobras → Qdrant Cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--materiais_dir",
        required=True,
        help="Diretório contendo os arquivos XLSX de materiais.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Processa e exibe amostras sem escrever no Qdrant.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocessa todos os arquivos mesmo sem mudança de MD5.",
    )
    parser.add_argument(
        "--recreate_collection",
        action="store_true",
        help="Deleta e recria a coleção antes de ingerir. Use com cautela.",
    )
    args = parser.parse_args()

    main(
        materiais_dir=args.materiais_dir,
        dry_run=args.dry_run,
        force=args.force,
        recreate_collection=args.recreate_collection,
    )