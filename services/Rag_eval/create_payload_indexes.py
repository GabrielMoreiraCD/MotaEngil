"""
create_payload_indexes.py
=========================
Cria payload indexes (tipo keyword) nas coleções do cluster Qdrant.

O Qdrant exige índice explícito para qualquer campo usado em FieldCondition.
Sem o índice, toda tentativa de filtro retorna:
  "Bad request: Index required but not found for <campo>"

Coleções alvo (cluster Mota_engil_test):
  - petrobras_rag_teoria
  - normas_tecnicas_publicas

Campos indexados por padrão neste script:
  keyword: norma, tipo, source, disciplina, capitulo, chunk_id
  integer: page_number

Execute UMA VEZ por coleção. Idempotente: re-executar não causa erro.

Uso:
  python create_payload_indexes.py
  python create_payload_indexes.py --collections petrobras_rag_teoria
  python create_payload_indexes.py --dry-run   # lista o que seria criado
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType


# ---------------------------------------------------------------------------
# Campos a indexar por tipo
# Adicione qualquer campo que você usa em metadata_filter nos ground truths
# ---------------------------------------------------------------------------

KEYWORD_FIELDS = [
    "norma",          # ex: "N-115", "N-279"
    "tipo",           # ex: "texto", "tabela", "figura"
    "source",         # nome do arquivo de origem
    "disciplina",     # ex: "tubulacao", "estrutural"
    "capitulo",       # ex: "4.2", "Anexo A"
    "chunk_id",       # ID único do chunk (usado nos ground truths)
    "doc_type",       # ex: "norma", "catalogo", "procedimento"
]

INTEGER_FIELDS = [
    "page_number",    # página do PDF de origem
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env() -> tuple[str, str]:
    """Carrega QDRANT_URL e QDRANT_API_KEY do .env."""
    for candidate in [Path.cwd() / ".env", Path.cwd().parent / ".env"]:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            print(f"[ENV] .env carregado: {candidate.resolve()}")
            break

    url = os.environ.get("QDRANT_URL", "").strip()
    key = os.environ.get("QDRANT_API_KEY", "").strip()

    if not url or not key:
        print("ERRO: QDRANT_URL e QDRANT_API_KEY são obrigatórios no .env")
        sys.exit(1)

    return url, key


def _get_existing_indexes(client: QdrantClient, collection: str) -> set[str]:
    """
    Retorna o conjunto de campos já indexados em uma coleção.
    Usa get_collection() → payload_schema.
    """
    info = client.get_collection(collection)
    # payload_schema é um dict {field_name: PayloadIndexInfo} ou None
    schema = getattr(info.payload_schema, "__root__", None) or {}
    if hasattr(info, "payload_schema") and info.payload_schema:
        try:
            schema = dict(info.payload_schema)
        except Exception:
            schema = {}
    return set(schema.keys())


def create_indexes(
    client: QdrantClient,
    collection: str,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """
    Cria os índices ausentes na coleção.

    Returns:
        {"created": [...], "skipped": [...], "failed": [...]}
    """
    result: dict[str, list[str]] = {"created": [], "skipped": [], "failed": []}

    print(f"\n[Indexer] Coleção: {collection}")

    try:
        existing = _get_existing_indexes(client, collection)
        print(f"  Campos já indexados: {sorted(existing) or '(nenhum)'}")
    except Exception as e:
        print(f"  AVISO: não foi possível listar índices existentes: {e}")
        existing = set()

    # Campos keyword
    for field in KEYWORD_FIELDS:
        if field in existing:
            print(f"  [SKIP] {field} (keyword) — já existe")
            result["skipped"].append(field)
            continue

        if dry_run:
            print(f"  [DRY-RUN] Criaria índice keyword: {field}")
            result["created"].append(field)
            continue

        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"  [OK] Índice keyword criado: {field}")
            result["created"].append(field)
        except Exception as e:
            err_str = str(e)
            # Idempotência: "already exists" não é erro real
            if "already exists" in err_str.lower():
                print(f"  [SKIP] {field} — já existe (confirmado pela API)")
                result["skipped"].append(field)
            else:
                print(f"  [ERRO] {field}: {err_str}")
                result["failed"].append(field)

    # Campos integer
    for field in INTEGER_FIELDS:
        if field in existing:
            print(f"  [SKIP] {field} (integer) — já existe")
            result["skipped"].append(field)
            continue

        if dry_run:
            print(f"  [DRY-RUN] Criaria índice integer: {field}")
            result["created"].append(field)
            continue

        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.INTEGER,
            )
            print(f"  [OK] Índice integer criado: {field}")
            result["created"].append(field)
        except Exception as e:
            err_str = str(e)
            if "already exists" in err_str.lower():
                print(f"  [SKIP] {field} — já existe")
                result["skipped"].append(field)
            else:
                print(f"  [ERRO] {field}: {err_str}")
                result["failed"].append(field)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cria payload indexes no Qdrant Cloud")
    p.add_argument(
        "--collections", type=str, nargs="+",
        default=["petrobras_rag_teoria", "normas_tecnicas_publicas"],
        help="Coleções alvo (padrão: ambas do cluster Mota_engil_test)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Lista o que seria criado sem executar.",
    )
    p.add_argument(
        "--extra-keyword-fields", type=str, nargs="*", default=[],
        metavar="FIELD",
        help="Campos keyword adicionais para indexar (além dos padrão).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Adiciona campos extras se fornecidos via CLI
    if args.extra_keyword_fields:
        for f in args.extra_keyword_fields:
            if f not in KEYWORD_FIELDS:
                KEYWORD_FIELDS.append(f)
        print(f"[Indexer] Campos extras adicionados: {args.extra_keyword_fields}")

    qdrant_url, qdrant_key = _load_env()

    print(f"\n[Indexer] Conectando: {qdrant_url}")
    if args.dry_run:
        print("[Indexer] MODO DRY-RUN — nenhuma alteração será feita\n")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # Valida coleções antes de processar
    available = [c.name for c in client.get_collections().collections]
    print(f"[Indexer] Coleções disponíveis no cluster: {available}")

    not_found = [c for c in args.collections if c not in available]
    if not_found:
        print(f"\nERRO: Coleções não encontradas: {not_found}")
        print(f"Disponíveis: {available}")
        sys.exit(1)

    # Cria índices em cada coleção
    summary: dict[str, dict] = {}
    for collection in args.collections:
        summary[collection] = create_indexes(client, collection, dry_run=args.dry_run)

    # Sumário final
    print("\n" + "=" * 55)
    print("  SUMÁRIO")
    print("=" * 55)
    for collection, res in summary.items():
        print(
            f"  {collection}:\n"
            f"    Criados : {len(res['created'])}  {res['created']}\n"
            f"    Pulados : {len(res['skipped'])}\n"
            f"    Falhas  : {len(res['failed'])}  {res['failed']}"
        )
    print()

    has_failures = any(res["failed"] for res in summary.values())
    if has_failures:
        print("⚠  Alguns índices falharam. Verifique as permissões da API key.")
        sys.exit(1)
    else:
        mode = "simulados" if args.dry_run else "criados"
        print(f"✓  Todos os índices {mode} com sucesso.")


if __name__ == "__main__":
    main()