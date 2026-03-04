"""
build_ground_truth.py
=====================
Ferramenta de anotação de ground truth para avaliação RAG.

Fluxo:
  1. Lê as queries reais definidas em QUERIES (ou arquivo externo --queries-file).
  2. Para cada query + coleção, chama o Qdrant e exibe os top-K documentos
     retornados com seus chunk_ids e trechos de texto.
  3. Abre prompt interativo: você marca quais docs são relevantes.
  4. Salva ground_truth.json pronto para uso no run_evaluation.py.

Modo não-interativo (--auto-top1):
  Marca automaticamente o doc de rank-1 como relevante.
  Útil para smoke-test rápido quando você quer apenas verificar se o
  pipeline está funcionando end-to-end.

Uso:
  # Interativo (recomendado para avaliação real)
  python build_ground_truth.py --output services/Rag_eval/ground_truth_real.json

  # Não-interativo, marca top-1 como relevante automaticamente
  python build_ground_truth.py --auto-top1 --output services/Rag_eval/ground_truth_auto.json

  # Limita a K documentos exibidos por query
  python build_ground_truth.py --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Queries reais derivadas da saída do triage_agent.py (LC-029)
# Edite esta lista para refletir as queries do seu projeto atual.
# Você também pode passar --queries-file com um JSON de queries.
# ---------------------------------------------------------------------------

DEFAULT_QUERIES: list[dict] = [
    # Formato: {query_id, query_text, metadata_filter (opcional)}
    {
        "query_id": "q001",
        "query_text": "espessura mínima tubulação carbono linha 3/4 polegadas pressão N-115",
        "metadata_filter": None,
    },
    {
        "query_id": "q002",
        "query_text": "TIE-IN procedimento conexão tubulação existente linha gás exportação",
        "metadata_filter": None,
    },
    {
        "query_id": "q003",
        "query_text": "fabricação montagem spool linha gás combustível alta pressão isométrico",
        "metadata_filter": None,
    },
    {
        "query_id": "q004",
        "query_text": "suporte tubulação offshore carga axial ASTM A36 fixação estrutural",
        "metadata_filter": None,
    },
    {
        "query_id": "q005",
        "query_text": "estimativa homem-hora HH tubulação 3/4 polegadas fabricação montagem offshore",
        "metadata_filter": None,
    },
    {
        "query_id": "q006",
        "query_text": "painel amostrador gás instalação isométrico IS-3010 tubulação",
        "metadata_filter": None,
    },
    {
        "query_id": "q007",
        "query_text": "N-858 requisitos preservação pintura conexões tubulação offshore",
        "metadata_filter": None,
    },
]

# Mapa canônico dimensão → modelo (espelho do rag_metrics.py)
_DIM_TO_MODEL: dict[int, str] = {
    384:  "sentence-transformers/all-MiniLM-L6-v2",
    768:  "sentence-transformers/all-mpnet-base-v2",
    1024: "BAAI/bge-m3",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env() -> dict[str, str]:
    for candidate in [Path.cwd() / ".env", Path.cwd().parent / ".env",
                      Path.cwd().parent.parent / ".env"]:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            print(f"[ENV] .env carregado: {candidate.resolve()}")
            break

    cfg = {
        "qdrant_url":    os.environ.get("QDRANT_URL", "").strip(),
        "qdrant_key":    os.environ.get("QDRANT_API_KEY", "").strip(),
        "collections":   os.environ.get("COLLECTION_NAMES",
                         os.environ.get("COLLECTION_NAME", "")).strip(),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        print(f"ERRO: variáveis ausentes no .env: {missing}")
        sys.exit(1)
    cfg["collection_list"] = [c.strip() for c in cfg["collections"].split(",") if c.strip()]
    return cfg


def _detect_dim(client: QdrantClient, collection: str) -> int:
    info = client.get_collection(collection)
    vec  = info.config.params.vectors
    if hasattr(vec, "size"):
        return int(vec.size)
    if isinstance(vec, dict) and vec:
        return int(vec[next(iter(vec))].size)
    return 0


def _get_embedder(dim: int) -> tuple[SentenceTransformer, str]:
    import warnings
    # Override por dim via .env
    model_name = os.environ.get(f"EMBEDDING_MODEL_{dim}", "").strip()
    if not model_name:
        model_name = _DIM_TO_MODEL.get(dim, os.environ.get("EMBEDDING_MODEL", ""))
    if not model_name:
        raise ValueError(f"Não foi possível resolver modelo para dimensão {dim}d.")
    print(f"  [Embedder] {model_name} ({dim}d)")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        embedder = SentenceTransformer(model_name)
    return embedder, model_name


def _embed(embedder: SentenceTransformer, text: str) -> list[float]:
    return embedder.encode(text, normalize_embeddings=True).tolist()


def _excerpt(payload: dict, max_chars: int = 200) -> str:
    """Extrai trecho de texto do payload para exibição."""
    for field in ("text", "content", "page_content", "chunk_text", "body", "texto"):
        val = payload.get(field, "")
        if val and isinstance(val, str):
            clean = " ".join(val.split())
            return clean[:max_chars] + ("…" if len(clean) > max_chars else "")
    return "(sem campo de texto no payload)"


def _display_hit(rank: int, doc_id: str, score: float, payload: dict) -> None:
    """Imprime um resultado de busca formatado."""
    norma    = payload.get("norma",    payload.get("norma_id",   ""))
    source   = payload.get("source",   "")
    page     = payload.get("page_number", "")
    chunk_tp = payload.get("chunk_type", payload.get("tipo", ""))

    meta_parts = [p for p in [norma, source, f"p.{page}" if page else "", chunk_tp] if p]
    meta_str   = "  |  ".join(meta_parts)

    excerpt = _excerpt(payload)
    print(f"\n  [{rank:>2}] chunk_id: {doc_id}")
    print(f"       score : {score:.4f}  |  {meta_str}")
    print(f"       texto : {textwrap.fill(excerpt, width=70, subsequent_indent='              ')}")


def _ask_relevant(num_hits: int) -> list[int]:
    """
    Prompt interativo: usuário informa quais ranks são relevantes.
    Aceita: "1 3 5" ou "1,3,5" ou "1-3" ou "none" ou Enter (pula).
    Retorna lista de índices base-0.
    """
    while True:
        raw = input(
            f"\n  Quais são relevantes? (ex: 1 3 5 | 1-3 | none | Enter=pular): "
        ).strip().lower()

        if raw in ("", "none", "n", "skip"):
            return []

        # Expande ranges: "1-3" → [1,2,3]
        tokens = re.split(r"[,\s]+", raw)
        ranks: list[int] = []
        valid = True
        for tok in tokens:
            if not tok:
                continue
            if "-" in tok:
                parts = tok.split("-")
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    ranks += list(range(int(parts[0]), int(parts[1]) + 1))
                else:
                    valid = False; break
            elif tok.isdigit():
                ranks.append(int(tok))
            else:
                valid = False; break

        if not valid:
            print("  Entrada inválida. Use números separados por espaço ou vírgula.")
            continue

        out_of_range = [r for r in ranks if r < 1 or r > num_hits]
        if out_of_range:
            print(f"  Ranks fora do intervalo [1, {num_hits}]: {out_of_range}")
            continue

        # Converte para índice base-0
        return [r - 1 for r in ranks]


# ---------------------------------------------------------------------------
# Core: busca + anotação por coleção
# ---------------------------------------------------------------------------

import re   # noqa: E402 (necessário para _ask_relevant antes da definição)


def harvest_one_collection(
    client: QdrantClient,
    collection: str,
    queries: list[dict],
    top_k: int,
    auto_top1: bool,
    chunk_id_field: str,
) -> list[dict]:
    """
    Executa as queries na coleção e retorna entries anotadas para ground truth.
    """
    dim      = _detect_dim(client, collection)
    embedder, model_name = _get_embedder(dim)

    print(f"\n{'='*65}")
    print(f"  Coleção : {collection}")
    print(f"  Embedder: {model_name} ({dim}d)")
    print(f"  Top-K   : {top_k}")
    if auto_top1:
        print(f"  Modo    : AUTO-TOP1 (rank-1 marcado automaticamente)")
    print(f"{'='*65}")

    gt_entries: list[dict] = []

    for entry in queries:
        qid   = entry["query_id"]
        qtext = entry["query_text"]
        mfilter = entry.get("metadata_filter")

        print(f"\n{'─'*65}")
        print(f"  [{collection}] Query {qid}: \"{qtext}\"")
        print(f"{'─'*65}")

        # Busca no Qdrant
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant_filter = None
        if mfilter:
            qdrant_filter = Filter(must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in mfilter.items()
            ])

        hits = client.search(
            collection_name=collection,
            query_vector=_embed(embedder, qtext),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        if not hits:
            print("  (nenhum resultado retornado)")
            gt_entries.append({
                "query_id":        qid,
                "query_text":      qtext,
                "collection":      collection,
                "relevant_doc_ids": [],
                "metadata_filter": mfilter,
                "_embedding_model": model_name,
            })
            continue

        # Exibe os hits
        doc_ids = []
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit.payload.get(chunk_id_field, str(hit.id))
            doc_ids.append(doc_id)
            _display_hit(rank, doc_id, hit.score, hit.payload)

        # Anotação
        if auto_top1:
            relevant_indices = [0]
            print(f"\n  [AUTO] Marcado como relevante: rank-1 → {doc_ids[0]}")
        else:
            relevant_indices = _ask_relevant(len(hits))

        relevant_ids = [doc_ids[i] for i in relevant_indices]
        if relevant_ids:
            print(f"  ✓ Marcados: {relevant_ids}")
        else:
            print(f"  ○ Nenhum marcado (query sem relevante nesta coleção)")

        gt_entries.append({
            "query_id":         qid,
            "query_text":       qtext,
            "collection":       collection,
            "relevant_doc_ids": relevant_ids,
            "metadata_filter":  mfilter,
            "_embedding_model": model_name,
        })

    return gt_entries


# ---------------------------------------------------------------------------
# Merge: gera ground truth unificado (sem campo collection — padrão eval)
# ou separado por coleção
# ---------------------------------------------------------------------------

def merge_to_gt_format(
    all_entries: list[dict],
    split_by_collection: bool = False,
) -> list[dict]:
    """
    Consolida anotações de múltiplas coleções em um único ground truth.

    Estratégia de merge:
      - Para cada query_id, une os relevant_doc_ids de todas as coleções.
      - metadata_filter é o da primeira entrada (deve ser igual entre coleções).
      - Remove campo interno `collection` e `_embedding_model`.
    """
    if split_by_collection:
        # Mantém entries separadas por coleção (uso avançado)
        return all_entries

    merged: dict[str, dict] = {}
    for entry in all_entries:
        qid = entry["query_id"]
        if qid not in merged:
            merged[qid] = {
                "query_id":         entry["query_id"],
                "query_text":       entry["query_text"],
                "relevant_doc_ids": list(entry["relevant_doc_ids"]),
                "metadata_filter":  entry.get("metadata_filter"),
            }
        else:
            # Une IDs relevantes de coleções diferentes sem duplicatas
            existing = set(merged[qid]["relevant_doc_ids"])
            for rid in entry["relevant_doc_ids"]:
                if rid not in existing:
                    merged[qid]["relevant_doc_ids"].append(rid)
                    existing.add(rid)

    return list(merged.values())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Construtor de Ground Truth para avaliação RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", type=str,
        default="services/Rag_eval/ground_truth_real.json",
        help="Caminho do arquivo ground truth a gerar.",
    )
    p.add_argument(
        "--queries-file", type=str, default=None,
        help="JSON com lista de queries (mesmo formato de DEFAULT_QUERIES).",
    )
    p.add_argument(
        "--collections", type=str, nargs="+", default=None,
        help="Override das coleções (padrão: COLLECTION_NAMES do .env).",
    )
    p.add_argument(
        "--top-k", type=int, default=10,
        help="Top-K documentos exibidos por query para anotação.",
    )
    p.add_argument(
        "--chunk-id-field", type=str, default="chunk_id",
        help="Campo no payload Qdrant com o ID do chunk.",
    )
    p.add_argument(
        "--auto-top1", action="store_true",
        help="Marca automaticamente rank-1 como relevante (sem interação).",
    )
    p.add_argument(
        "--split-by-collection", action="store_true",
        help="Gera um ground truth por coleção em vez de um unificado.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = _load_env()

    # Queries
    if args.queries_file:
        with open(args.queries_file, encoding="utf-8") as f:
            queries = json.load(f)
        print(f"[GT Builder] {len(queries)} queries carregadas de: {args.queries_file}")
    else:
        queries = DEFAULT_QUERIES
        print(f"[GT Builder] Usando {len(queries)} queries padrão (LC-029 / Mota-Engil)")

    # Coleções
    collections = args.collections or cfg["collection_list"]
    print(f"[GT Builder] Coleções: {collections}")

    # Conecta ao Qdrant
    client = QdrantClient(url=cfg["qdrant_url"], api_key=cfg["qdrant_key"])
    available = [c.name for c in client.get_collections().collections]
    print(f"[GT Builder] Coleções disponíveis: {available}")

    missing = [c for c in collections if c not in available]
    if missing:
        print(f"ERRO: coleções não encontradas: {missing}")
        sys.exit(1)

    # Coleta anotações por coleção
    all_entries: list[dict] = []
    for col in collections:
        entries = harvest_one_collection(
            client=client,
            collection=col,
            queries=queries,
            top_k=args.top_k,
            auto_top1=args.auto_top1,
            chunk_id_field=args.chunk_id_field,
        )
        all_entries.extend(entries)

    # Merge e salva
    gt = merge_to_gt_format(all_entries, split_by_collection=args.split_by_collection)

    if args.split_by_collection:
        # Salva um arquivo por coleção
        output_base = Path(args.output).with_suffix("")
        for col in collections:
            col_entries = [e for e in gt if e.get("collection") == col]
            out_path = Path(f"{output_base}_{col}.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(col_entries, f, ensure_ascii=False, indent=2)
            print(f"\n[GT Builder] Ground truth '{col}' salvo em: {out_path}")
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)
        print(f"\n[GT Builder] Ground truth salvo em: {out_path}")
        print(f"  Queries anotadas : {len(gt)}")
        total_relevant = sum(len(e["relevant_doc_ids"]) for e in gt)
        queries_sem_rel = sum(1 for e in gt if not e["relevant_doc_ids"])
        print(f"  Docs relevantes  : {total_relevant}")
        print(f"  Queries sem relev: {queries_sem_rel} "
              f"{'⚠ considere revisar' if queries_sem_rel > 0 else '✓'}")

    print(f"\nPróximo passo:")
    print(f"  python run_evaluation.py --ground-truth {args.output}")


if __name__ == "__main__":
    main()