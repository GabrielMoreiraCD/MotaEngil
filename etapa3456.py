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
    output_dir: str = "Data/etapa_bom_output",
    use_claude: bool = True,
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
    log.info(f"  CSV:          {csv_path}")
    log.info(f"  MD:           {resolved_md}")
    log.info(f"  Output:       {output_dir}")
    log.info(f"  LLM Etapa 3:  {'Claude (' + __import__('core.config', fromlist=['config']).config.CLAUDE_MODEL + ')' if use_claude else 'Llama3 (local)'}")
    log.info(f"  Formato:      {export_format}")
    log.info(f"  Retomar de:   Etapa {resume_from}")
    log.info("=" * 70)

    # ─── ETAPA 2: Triage ─────────────────────────────────────────────────────
    from core.schemas import EscopoTriagemUnificado

    if resume_from <= 2 or not f_triage.exists():
        log.info("\n[ETAPA 2] Extração de escopo via Triage Agent...")
        from services.triage_agent import UnifiedTriageAgent
        agent = UnifiedTriageAgent(use_claude=use_claude)
        triage_raw = agent.process_project_files(
            csv_path=csv_path,
            md_path=resolved_md,
        )
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
        # Garante que pelo menos o que foi extraído seja preservado
        escopo = EscopoTriagemUnificado.model_validate({
            k: v for k, v in (triage_raw or {}).items()
            if v is not None and v != [] and v != ""
        })

    log.info(f"  Serviço: {escopo.id_servico} | {escopo.titulo_servico}")
    log.info(f"  Normas aplicáveis: {escopo.normas_petrobras_aplicaveis}")

    # ─── ETAPA 3: Consulta de Normas ─────────────────────────────────────────
    from core.schemas import NormasConsultaResult

    if resume_from <= 3 or not f_normas.exists():
        log.info("\n[ETAPA 3] Consultando normas técnicas no Qdrant...")
        from services.normas_agent import NormasConsultationAgent
        normas_agent = NormasConsultationAgent(use_claude=use_claude)
        normas_result = normas_agent.process(escopo, output_path=f_normas)
    else:
        log.info(f"\n[ETAPA 3] Carregando resultado de normas do cache: {f_normas}")
        normas_raw = _load_json(f_normas)
        normas_result = NormasConsultaResult.model_validate(normas_raw)

    log.info(
        f"  {len(normas_result.especificacoes_extraidas)} specs extraídas de "
        f"{len(normas_result.normas_consultadas)} normas | "
        f"{len(normas_result.normas_sem_resultado)} sem resultado no Qdrant"
    )

    if not normas_result.especificacoes_extraidas:
        log.warning(
            "Nenhuma especificação extraída das normas. "
            "Verifique se as normas estão indexadas no Qdrant "
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
        "--use-llama",
        action="store_true",
        help="Usa Llama3 local (Ollama) em vez de Claude na Etapa 3. Recomendado apenas se não houver ANTHROPIC_API_KEY.",
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
        output_dir=args.output_dir,
        use_claude=not args.use_llama,
        export_format=args.format,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
    )

    # Imprime sumário JSON para integração com outros scripts
    print("\n=== SUMÁRIO JSON ===")
    print(json.dumps(result["stats"], indent=2, ensure_ascii=False))
