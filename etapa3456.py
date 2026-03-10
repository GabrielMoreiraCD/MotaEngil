"""
etapa3456.py — Orquestrador: Triage → Normas → Materiais → BOM
===============================================================
Entry point único que encadeia as Etapas 2→3→4→5 para gerar a
Lista de Materiais de um serviço de engenharia automaticamente.

Cada etapa persiste seu resultado como JSON na pasta de saída,
permitindo retomar o pipeline a partir de qualquer ponto após uma falha
(usar --resume-from para pular etapas já concluídas).

USO:
  # Execução completa com arquivo MD específico (recomendado para SS-93)
  python etapa3456.py \\
    --csv "Data/projetos/SS-93 Reparo com Solda/Documento de Referencia/DE - 14770545-1-1.csv" \\
    --md "Data/projetos/SS-93 Reparo com Solda/PARTE 1 - RelatorioSS.pdf" \\
    --output-dir Data/etapa_bom_output/SS-93 \\
    --format both

  # Com diretório MD (usa o primeiro arquivo .pdf/.md/.txt encontrado)
  python etapa3456.py --csv ... --md-dir "Data/projetos/SS-93 Reparo com Solda"

  # Com Llama3 local (sem API Anthropic)
  python etapa3456.py --csv ... --md ... --use-llama

  # Retomar da Etapa 4 (Etapas 2 e 3 já concluídas)
  python etapa3456.py --csv ... --md ... --resume-from 4

  # Dry-run: mostra apenas o que seria gerado
  python etapa3456.py --csv ... --md ... --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Pipeline_BOM")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de persistência
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_md_path(md_path: str | None, md_dir: str | None) -> str:
    """
    Resolve o caminho do Memorial Descritivo.
    Se md_path fornecido: usa direto.
    Se md_dir fornecido: busca o primeiro arquivo .pdf, .md ou .txt no diretório.
    """
    if md_path:
        p = Path(md_path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo MD não encontrado: {md_path}")
        return str(p)

    if md_dir:
        d = Path(md_dir)
        for ext in ("*.pdf", "*.md", "*.txt"):
            candidates = sorted(d.rglob(ext))
            if candidates:
                log.info(f"  MD auto-selecionado: {candidates[0].name}")
                return str(candidates[0])
        raise FileNotFoundError(f"Nenhum arquivo .pdf/.md/.txt encontrado em: {md_dir}")

    raise ValueError("Forneça --md (arquivo) ou --md-dir (diretório) com o Memorial Descritivo.")


def run_pipeline(
    csv_path: str,
    md_path: str | None = None,
    md_dir: str | None = None,
    ieis_path: str | None = None,
    ebp_path: str | None = None,
    isometrico_paths: list[str] | None = None,
    output_dir: str = "Data/etapa_bom_output",
    use_claude: bool = False,
    use_gemini: bool = True,
    export_format: str = "xlsx",
    resume_from: int = 2,
    dry_run: bool = False,
) -> dict:
    """
    Executa o pipeline completo de geração de BOM.

    Args:
        csv_path:     Caminho para o CSV do formulário de serviço (DE-*.csv)
        md_path:      Caminho para o arquivo do Memorial Descritivo (PDF/MD/TXT)
        md_dir:       Alternativa: diretório com PDFs (usa o primeiro encontrado)
        output_dir:   Diretório de saída para JSONs e arquivo final
        use_claude:   Se True, usa claude-sonnet-4-6 na Etapa 3 (requer ANTHROPIC_API_KEY)
        export_format: "xlsx" | "pdf" | "both"
        resume_from:  Etapa a partir da qual retomar (2=triage, 3=normas, 4=materiais, 5=bom)
        dry_run:      Se True, executa tudo mas não salva arquivos finais

    Returns:
        dict com caminhos dos arquivos gerados e estatísticas
    """
    resolved_md = _resolve_md_path(md_path, md_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Arquivos intermediários
    f_triage   = output_path / "etapa2_triage_result.json"
    f_normas   = output_path / "etapa3_normas_result.json"
    f_materiais = output_path / "etapa4_materiais_result.json"
    f_bom_json = output_path / "etapa5_bom_result.json"

    log.info("=" * 70)
    log.info("PIPELINE BOM — GERAÇÃO AUTOMÁTICA DE LISTA DE MATERIAIS")
    log.info("=" * 70)
    from core.config import config as _cfg
    _llm_label = (
        f"Gemini ({_cfg.GEMINI_MODEL})" if _cfg.USE_GEMINI
        else (f"Claude ({_cfg.CLAUDE_MODEL})" if use_claude and _cfg.USE_CLAUDE
              else "Llama3 (local)")
    )
    log.info(f"  CSV:          {csv_path}")
    log.info(f"  MD:           {resolved_md}")
    log.info(f"  IEIS:         {ieis_path or '(não fornecido)'}")
    log.info(f"  EBP:          {ebp_path or '(não fornecido)'}")
    log.info(f"  Isométrico(s):{' | '.join(isometrico_paths) if isometrico_paths else '(não fornecido)'}")
    log.info(f"  Output:       {output_dir}")
    log.info(f"  LLM:          {_llm_label}")
    log.info(f"  Formato:      {export_format}")
    log.info(f"  Retomar de:   Etapa {resume_from}")
    log.info("=" * 70)

    # ─── ETAPA 2: Triage ─────────────────────────────────────────────────────
    from core.schemas import EscopoTriagemUnificado

    if resume_from <= 2 or not f_triage.exists():
        log.info("\n[ETAPA 2] Extração de escopo via Triage Agent (Four-Pass)...")
        from services.triage_agent import UnifiedTriageAgent
        agent = UnifiedTriageAgent()
        triage_raw = agent.process_project_files(
            csv_path=csv_path,
            md_path=resolved_md,
            ieis_path=ieis_path,
            ebp_path=ebp_path,
        )

        # Pass 5: Leitura visual de isométricos (Qwen2.5-VL) — se fornecidos
        if isometrico_paths:
            from core.config import config as _cfg2
            if _cfg2.HF_TOKEN:
                log.info(f"\n[ETAPA 2 / Pass 5] Lendo {len(isometrico_paths)} isométrico(s) via Qwen2.5-VL...")
                from services.isometric_reader import IsometricReader
                reader = IsometricReader(hf_token=_cfg2.HF_TOKEN)
                context = {
                    "tag_linha": triage_raw.get("tag_linha_principal", ""),
                    "piping_class": triage_raw.get("piping_class_referencia", ""),
                }
                iso_specs = reader.extract_specs_batch(isometrico_paths, context)
                if iso_specs:
                    triage_raw["isometric_specs"] = [s.model_dump() for s in iso_specs]
                    log.info(f"  Pass 5 concluído: {len(iso_specs)} specs do isométrico adicionadas.")
                else:
                    log.warning("  Pass 5: Nenhuma spec extraída do isométrico.")
            else:
                log.warning("  Pass 5 ignorado: HF_TOKEN não configurado no .env")

        _save_json(triage_raw, f_triage)
        log.info(f"  Triage concluído. Salvo em: {f_triage}")
    else:
        log.info(f"\n[ETAPA 2] Carregando triage do cache: {f_triage}")
        triage_raw = _load_json(f_triage)

    # Valida com Pydantic — campos são Optional, então resultado parcial é aceito
    try:
        escopo = EscopoTriagemUnificado.model_validate(triage_raw)
    except Exception as e:
        log.warning(f"Validação Pydantic falhou ({e}). Usando escopo com campos padrão.")
        escopo = EscopoTriagemUnificado.model_validate({
            k: v for k, v in (triage_raw or {}).items()
            if v is not None and v != [] and v != ""
        })

    # ── DIAGNÓSTICO: log do escopo validado ──────────────────────────────────
    log.info(f"  Serviço: {escopo.id_servico} | {escopo.titulo_servico}")
    log.info(f"  Plataforma: {escopo.plataforma} | Tag: {escopo.tag_linha_principal}")
    log.info(f"  LP/ZE: {escopo.numero_ze}")
    log.info(f"  Normas aplicáveis: {escopo.normas_petrobras_aplicaveis}")
    ieis = getattr(escopo, "especificacoes_soldagem", None)
    if ieis:
        log.info(f"  IEIS: eletrodo={getattr(ieis,'metal_adicao',None)} | NDT={getattr(ieis,'ndt_requerido',[])}")
    spool_list = getattr(escopo, "spool_list", None) or []
    log.info(f"  Spools: {len(spool_list)} | Isométrico specs: {len(getattr(escopo,'isometric_specs',[]))}")

    # ─── ETAPA 3: Consulta de Normas ─────────────────────────────────────────
    from core.schemas import NormasConsultaResult

    if resume_from <= 3 or not f_normas.exists():
        log.info("\n[ETAPA 3] Consultando normas técnicas no Qdrant + injeção direta IEIS/EBP...")
        from services.normas_agent import NormasConsultationAgent
        normas_agent = NormasConsultationAgent()
        normas_result = normas_agent.process(escopo, output_path=f_normas)
    else:
        log.info(f"\n[ETAPA 3] Carregando resultado de normas do cache: {f_normas}")
        normas_raw = _load_json(f_normas)
        normas_result = NormasConsultaResult.model_validate(normas_raw)

    # ── DIAGNÓSTICO: breakdown por fonte ─────────────────────────────────────
    n_specs = len(normas_result.especificacoes_extraidas)
    by_fonte = {}
    for s in normas_result.especificacoes_extraidas:
        src = s.norma_origem or "?"
        by_fonte[src] = by_fonte.get(src, 0) + 1
    log.info(
        f"  {n_specs} specs extraídas de {len(normas_result.normas_consultadas)} normas | "
        f"{len(normas_result.normas_sem_resultado)} sem resultado no Qdrant"
    )
    log.info(f"  Breakdown por fonte: {by_fonte}")

    ieis_count = by_fonte.get("IEIS", 0) + by_fonte.get("IEIS_NDT", 0)
    ebp_count  = by_fonte.get("EBP_SPOOL", 0) + by_fonte.get("ESCOPO_TAG", 0)
    iso_count  = sum(v for k, v in by_fonte.items() if k.startswith("ISOMETRICO:"))
    qdrant_count = n_specs - ieis_count - ebp_count - iso_count
    log.info(
        f"  Fontes: IEIS={ieis_count} | EBP/spool={ebp_count} | "
        f"Isométrico={iso_count} | Qdrant={qdrant_count}"
    )

    if not normas_result.especificacoes_extraidas:
        log.warning(
            "Nenhuma especificação extraída. Verifique se as normas estão indexadas no Qdrant "
            f"(collection: {__import__('core.config', fromlist=['config']).config.COLLECTION_NORMAS})."
        )

    # ─── ETAPA 4: Mapeamento de Materiais ────────────────────────────────────
    from core.schemas import MateriaisRequisitados

    if resume_from <= 4 or not f_materiais.exists():
        log.info("\n[ETAPA 4] Mapeando especificações para o catálogo de materiais...")
        from services.materiais_agent import MateriaisMappingAgent
        materiais_agent = MateriaisMappingAgent()
        materiais_result = materiais_agent.process(normas_result, escopo, output_path=f_materiais)
    else:
        log.info(f"\n[ETAPA 4] Carregando mapeamento do cache: {f_materiais}")
        materiais_raw = _load_json(f_materiais)
        materiais_result = MateriaisRequisitados.model_validate(materiais_raw)

    log.info(
        f"  {materiais_result.total_mapeados} mapeados | "
        f"{materiais_result.total_nao_mapeados} não mapeados | "
        f"total: {materiais_result.total_requisitos}"
    )

    # ─── ETAPA 5: Geração da BOM ─────────────────────────────────────────────
    log.info("\n[ETAPA 5] Gerando Lista de Materiais...")
    from services.bom_generator import BOMGenerator
    generator = BOMGenerator()
    bom = generator.generate(
        materiais=materiais_result,
        escopo=escopo,
        normas_base=normas_result.normas_consultadas,
    )

    output_files: dict[str, str] = {}

    if not dry_run:
        # Persiste JSON
        generator.save_json(bom, f_bom_json)
        output_files["json"] = str(f_bom_json)

        # Exporta XLSX
        if export_format in ("xlsx", "both"):
            xlsx_path = output_path / f"BOM_{escopo.id_servico}.xlsx"
            generator.export_xlsx(bom, xlsx_path)
            output_files["xlsx"] = str(xlsx_path)

        # Exporta PDF
        if export_format in ("pdf", "both"):
            pdf_path = output_path / f"BOM_{escopo.id_servico}.pdf"
            generator.export_pdf(bom, pdf_path)
            output_files["pdf"] = str(pdf_path)
    else:
        log.info("  [DRY RUN] Nenhum arquivo exportado.")

    # ─── Relatório final ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PIPELINE BOM — CONCLUÍDO")
    log.info("=" * 70)
    log.info(f"  Serviço:              {bom.id_servico}")
    log.info(f"  Total de itens:       {bom.total_itens}")
    log.info(f"  Itens com código:     {bom.total_itens - bom.itens_sem_codigo}")
    log.info(f"  Itens sem código:     {bom.itens_sem_codigo} (requerem revisão)")
    log.info(f"  Advertências:         {len(bom.advertencias)}")
    if output_files:
        log.info(f"  Arquivos gerados:")
        for fmt, path in output_files.items():
            log.info(f"    {fmt.upper()}: {path}")
    log.info("=" * 70)

    if bom.advertencias:
        log.warning("\nADVERTÊNCIAS PARA O ENGENHEIRO REVISOR:")
        for adv in bom.advertencias:
            log.warning(f"  ⚠ {adv}")

    return {
        "bom": bom.model_dump(),
        "output_files": output_files,
        "stats": {
            "total_itens": bom.total_itens,
            "itens_com_codigo": bom.total_itens - bom.itens_sem_codigo,
            "itens_sem_codigo": bom.itens_sem_codigo,
            "normas_consultadas": len(normas_result.normas_consultadas),
            "specs_extraidas": len(normas_result.especificacoes_extraidas),
            "mapeados": materiais_result.total_mapeados,
            "nao_mapeados": materiais_result.total_nao_mapeados,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline automático de Lista de Materiais para serviços de engenharia Petrobras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Caminho para o CSV do formulário de serviço (DE-*.csv)",
    )
    parser.add_argument(
        "--md",
        default=None,
        help="Caminho para o arquivo do Memorial Descritivo (PDF, MD ou TXT)",
    )
    parser.add_argument(
        "--md-dir",
        default=None,
        help="Diretório com os arquivos do projeto (usa o primeiro PDF/MD/TXT encontrado). Use --md ou --md-dir.",
    )
    parser.add_argument(
        "--ieis",
        default=None,
        help="Caminho para o IEIS (Instrução para Execução e Inspeção de Solda). "
             "Fornece material base, eletrodo, processo de soldagem e NDT diretamente.",
    )
    parser.add_argument(
        "--ebp",
        default=None,
        help="Caminho para o EBP/Planejamento Executivo (Book of Planning). "
             "Fornece spool list com comprimentos reais, isométricos e referência de piping class.",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/etapa_bom_output",
        help="Diretório de saída para JSONs intermediários e arquivo final (padrão: Data/etapa_bom_output)",
    )
    parser.add_argument(
        "--format",
        choices=["xlsx", "pdf", "both"],
        default="xlsx",
        help="Formato de exportação da BOM (padrão: xlsx)",
    )
    parser.add_argument(
        "--isometrico",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Caminho(s) para imagem(ns) JPG/PNG de isométrico. "
             "Usa Qwen2.5-VL-72B via HF Inference API para extrair specs visuais. "
             "Exemplo: --isometrico 'Figura 1 atualizada.jpg'",
    )
    parser.add_argument(
        "--use-llama",
        action="store_true",
        help="Usa Llama3 local (Ollama) em vez de Gemini na Etapa 3. Para uso offline.",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        choices=[2, 3, 4, 5],
        default=2,
        help="Etapa a partir da qual retomar (2=triage, 3=normas, 4=materiais, 5=bom). Usa cache das etapas anteriores.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa o pipeline sem exportar arquivos finais.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if not args.md and not args.md_dir:
        parser.error("Forneça --md (arquivo) ou --md-dir (diretório) com o Memorial Descritivo.")

    result = run_pipeline(
        csv_path=args.csv,
        md_path=args.md,
        md_dir=args.md_dir,
        ieis_path=args.ieis,
        ebp_path=args.ebp,
        isometrico_paths=args.isometrico,
        output_dir=args.output_dir,
        use_claude=False,           # Claude desabilitado — usar Gemini ou Llama3
        use_gemini=not args.use_llama,
        export_format=args.format,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
    )

    # Imprime sumário JSON para integração com outros scripts
    print("\n=== SUMÁRIO JSON ===")
    print(json.dumps(result["stats"], indent=2, ensure_ascii=False))
