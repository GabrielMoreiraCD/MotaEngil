"""
run_evaluation.py
=================
CLI para avaliação RAG multi-coleção contra o Qdrant Cloud.
Toda configuração primária vem do .env via Settings.from_env().

─── .env esperado ────────────────────────────────────────────────
QDRANT_API_KEY=eyJh...
QDRANT_URL=https://b9b4...qdrant.io
COLLECTION_NAMES=petrobras_rag_teoria,normas_tecnicas_publicas   ← multi
COLLECTION_NAME=petrobras_rag_teoria                             ← fallback (uma só)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_TOKEN=hf_...
LLM_MODEL=llama3:latest
──────────────────────────────────────────────────────────────────

Exemplos de uso:
  # Avalia as coleções definidas em COLLECTION_NAMES do .env
  python run_evaluation.py --ground-truth data/ground_truth.json

  # Override: avalia somente uma coleção específica
  python run_evaluation.py --ground-truth data/gt.json \\
    --collections petrobras_rag_teoria

  # Override: avalia duas coleções específicas (ignora .env COLLECTION_NAMES)
  python run_evaluation.py --ground-truth data/gt.json \\
    --collections petrobras_rag_teoria normas_tecnicas_publicas

  # Gera template de ground truth vazio
  python run_evaluation.py --create-template data/gt_template.json
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG Metrics Evaluator — Delineador Técnico Petrobras/Mota-Engil",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    p.add_argument(
        "--ground-truth", type=str, default=None,
        help="Caminho para JSON de ground truth.",
    )
    p.add_argument(
        "--create-template", type=str, default=None, metavar="OUTPUT_PATH",
        help="Gera template de ground truth vazio e encerra.",
    )

    # Config overrides (opcional — padrão vem do .env)
    p.add_argument(
        "--env-file", type=str, default=None,
        help="Caminho explícito para .env (padrão: auto-localiza).",
    )
    p.add_argument(
        "--collections", type=str, nargs="+", default=None,
        metavar="COLLECTION",
        help=(
            "Override de COLLECTION_NAMES do .env. "
            "Aceita uma ou mais coleções: --collections col_a col_b"
        ),
    )
    p.add_argument(
        "--embedding-model", type=str, default=None,
        help="Override de EMBEDDING_MODEL do .env.",
    )

    # Parâmetros de avaliação
    p.add_argument(
        "--k", type=int, nargs="+", default=[1, 3, 5, 10],
        help="Cortes K (ex: --k 1 3 5 10).",
    )
    p.add_argument(
        "--top-k-retrieve", type=int, default=10,
        help="Docs buscados no Qdrant por query (deve ser >= max(K)).",
    )
    p.add_argument(
        "--chunk-id-field", type=str, default="chunk_id",
        help="Campo no payload Qdrant com o ID do chunk.",
    )

    # Saída
    p.add_argument(
        "--output-dir", type=str, default="reports/",
        help="Diretório para relatórios JSON e CSV.",
    )
    p.add_argument(
        "--no-verbose", action="store_true",
        help="Suprime logs por query.",
    )

    return p.parse_args()


def _safe_name(s: str) -> str:
    """Sanitiza nome de coleção para uso em nomes de arquivo."""
    return s.replace(" ", "_").replace("/", "-")


def main() -> None:
    args = parse_args()

    from rag_metrics import (
        Settings,
        MultiCollectionEvaluator,
        GroundTruthDataset,
        MetricsReporter,
    )

    # ----------------------------------------------------------------
    # Modo template — sem I/O externo
    # ----------------------------------------------------------------
    if args.create_template:
        GroundTruthDataset.create_template(args.create_template, num_examples=5)
        print(
            f"\nPreencha o arquivo gerado:\n"
            f"  query_text       → pergunta em linguagem natural\n"
            f"  relevant_doc_ids → IDs dos chunks (campo '{args.chunk_id_field}' no Qdrant)\n"
            f"  metadata_filter  → ex: {{\"norma\": \"N-115\"}} ou null\n"
        )
        sys.exit(0)

    if not args.ground_truth:
        print("ERRO: --ground-truth é obrigatório.\n"
              "Use --create-template para gerar um template vazio.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Settings: carrega .env, aplica overrides CLI
    # ----------------------------------------------------------------
    cfg = Settings.from_env(env_path=args.env_file)

    # Override de embedding model via CLI
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model
        cfg = Settings.from_env(env_path=args.env_file)  # recarrega com override

    # Determina lista final de coleções:
    #   1. --collections CLI  >  2. COLLECTION_NAMES .env  >  3. COLLECTION_NAME .env
    target_collections: list[str] = args.collections or list(cfg.collection_names)

    print(f"\n[CLI] Coleções alvo: {target_collections}")
    print(f"[CLI] Embedding    : {cfg.embedding_model}")
    print(f"[CLI] K values     : {args.k}\n")

    # ----------------------------------------------------------------
    # Ground truth (compartilhado entre todas as coleções)
    # ----------------------------------------------------------------
    gt = GroundTruthDataset(args.ground_truth)
    print(f"[CLI] Ground truth : {len(gt)} queries\n")

    # ----------------------------------------------------------------
    # Avaliação multi-coleção
    # ----------------------------------------------------------------
    evaluator = MultiCollectionEvaluator(
        collection_names=target_collections,
        settings=cfg,
        k_values=args.k,
        top_k_retrieve=args.top_k_retrieve,
        chunk_id_field=args.chunk_id_field,
    )

    comparison = evaluator.evaluate_all(gt, verbose=not args.no_verbose)

    # ----------------------------------------------------------------
    # Relatórios individuais + comparação
    # ----------------------------------------------------------------
    output_dir = Path(args.output_dir)
    ts = comparison.timestamp.replace(":", "-").replace(".", "-")

    # Sumário individual por coleção
    for col, report in comparison.reports.items():
        MetricsReporter.print_summary(report)
        safe = _safe_name(col)
        MetricsReporter.save_json(report, output_dir / f"eval_{safe}_{ts}.json")
        MetricsReporter.save_csv(report,  output_dir / f"eval_{safe}_{ts}.csv")

    # Comparação side-by-side (sempre, mesmo com N > 2)
    MetricsReporter.print_comparison(comparison)
    MetricsReporter.save_comparison_json(
        comparison, output_dir / f"comparison_{ts}.json"
    )
    MetricsReporter.save_comparison_csv(
        comparison, output_dir / f"comparison_{ts}.csv"
    )

    # ----------------------------------------------------------------
    # Diagnóstico: queries onde as coleções divergem mais (apenas 2 cols)
    # ----------------------------------------------------------------
    if len(target_collections) == 2:
        _print_divergent_queries(comparison, target_collections, k=args.k[-1])


def _print_divergent_queries(
    comp,
    cols: list[str],
    k: int,
    top_n: int = 5,
) -> None:
    """
    Imprime as top_n queries com maior divergência de nDCG@K entre coleções.
    Útil para inspecionar onde cada coleção tem vantagem qualitativa.

    Divergência = |nDCG_A - nDCG_B|
    """
    from rag_metrics import QueryResult

    rep_a = comp.reports[cols[0]]
    rep_b = comp.reports[cols[1]]

    # Alinha por query_id
    results_a: dict[str, QueryResult] = {r.query_id: r for r in rep_a.per_query_results}
    results_b: dict[str, QueryResult] = {r.query_id: r for r in rep_b.per_query_results}

    common_ids = set(results_a.keys()) & set(results_b.keys())
    divergences = []
    for qid in common_ids:
        ra = results_a[qid]
        rb = results_b[qid]
        ndcg_a = ra.ndcg_at_k.get(k, 0.0)
        ndcg_b = rb.ndcg_at_k.get(k, 0.0)
        divergences.append((abs(ndcg_a - ndcg_b), qid, ndcg_a, ndcg_b, ra.query_text))

    divergences.sort(reverse=True)

    sep = "-" * 80
    print(f"\n  TOP-{top_n} QUERIES COM MAIOR DIVERGÊNCIA (nDCG@{k})")
    print(f"  {'Query':<8}  {cols[0]:>18}  {cols[1]:>18}  {'Δ':>8}  Texto")
    print(f"  {sep}")
    for diff, qid, a, b, text in divergences[:top_n]:
        winner = "←" if a > b else ("→" if b > a else "=")
        print(
            f"  {qid:<8}  {a:>18.4f}  {b:>18.4f}  "
            f"{diff:>+8.4f} {winner}  '{text[:45]}'"
        )
    print()


if __name__ == "__main__":
    main()