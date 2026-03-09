"""
etapa1_inspect_normas.py
═══════════════════════════════════════════════════════════════════════════════
ETAPA 1 — Inspeção de Normas e Mapeamento de Isométricos

Responsabilidades:
  1. Lê o DescricaodeDocumentos.csv e identifica todas as referências ET,
     piping_classes e documentos relevantes para os 4 projetos SS do POC.
  2. Para cada ET encontrada nas normas privadas (PDF local):
       - Lista todas as imagens por página com dimensões e posição Y
       - Captura o texto adjacente (± 50px no eixo Y) de cada imagem
       - Classifica candidatos a isométrico por tamanho e keywords no contexto
       - Gera um mapa page_num → [imagens_candidatas]
  3. Consulta o Qdrant (normas públicas já indexadas) buscando chunks que
     mencionem piping_class + isométrico/esquema/figura para identificar
     quais normas públicas também contêm imagens relevantes.
  4. Gera dois arquivos de saída:
       - inspection_report.json  → dados estruturados para image_ingestor.py
       - inspection_report.md    → relatório legível para revisão manual

Uso:
  python etapa1_inspect_normas.py
  python etapa1_inspect_normas.py --csv custom_path.csv
  python etapa1_inspect_normas.py --skip-qdrant   (se Qdrant não estiver acessível)
  python etapa1_inspect_normas.py --pdf-only Z_I-ET-3010.68-1200-200-JUR-001_F 3 para tubulação.pdf

Dependências:
  pip install pymupdf pandas qdrant-client rich
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse, csv, json, re, sys, hashlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ─── Dependências opcionais com fallback ─────────────────────────────────────
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("[AVISO] PyMuPDF não instalado. Instale com: pip install pymupdf")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchText
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    print("[AVISO] qdrant-client não instalado. Busca Qdrant desabilitada.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    console = Console()
    RICH = True
except ImportError:
    RICH = False
    class _FakeConsole:
        def print(self, *a, **k): print(*[str(x) for x in a])
        def rule(self, t=""): print(f"\n{'─'*60} {t} {'─'*60}")
    console = _FakeConsole()

# ─── Configuração de paths ────────────────────────────────────────────────────
DATA_ROOT       = Path(r"C:\luza_datasets\Rag-pipeline-main\Data")
CSV_PATH        = DATA_ROOT / "DescricaodeDocumentos.csv"
NORMAS_PRIVADAS = DATA_ROOT / "normas_privadas"
OUTPUT_DIR      = DATA_ROOT / "etapa1_output"

# Qdrant config (ajustar conforme ambiente)
QDRANT_URL      = "https://b9b4abf7-5b65-4adf-b06a-5cd6c409bcde.us-west-2-0.aws.cloud.qdrant.io"   # ou URL Cloud
QDRANT_API_KEY  = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.L4BNhAponLF7felJu0rgsGGJLpiu_P7MwHSpsMtUx50"
COLLECTION_NAME = "normas_tecnicas_publicas_v2"         # coleção já existente

# ─── Regex de identificação técnica ──────────────────────────────────────────
RE_PIPING_CLASS = re.compile(r'\b([A-Z][0-9]{1,2}[A-Z]?S?)\b')
RE_ET_REF       = re.compile(r'I-ET-[\d\.]+-[\w-]+|DR-ENGP[\w\.-]+', re.I)
RE_ISO_KEYWORD  = re.compile(
    r'isométric|isometric|esquema|piping\s*class|linha\s*de\s*processo|'
    r'material\s*line|flow\s*line|tubulação|pipe\s*spec|classe\s*de\s*linha',
    re.I
)
RE_DN           = re.compile(r'DN\s*(\d+)|(\d+)\s*["\']', re.I)
RE_SCH          = re.compile(r'SCH(?:EDULE)?\s*(XS|STD|\d+)', re.I)
RE_PIPING_SPEC  = re.compile(r'\d+["\'][-–]\s*[A-Z][-–]\s*([A-Z0-9]+)[-–]\s*\d+')

# Tamanho mínimo para considerar uma imagem como "isométrico" (pixels)
# Ajustar após inspeção visual — valores iniciais conservadores
IMG_MIN_WIDTH  = 300   # px
IMG_MIN_HEIGHT = 200   # px
IMG_MIN_AREA   = 80_000  # px²  (~283×283 px mínimo)

# Janela de texto adjacente (pontos PDF acima/abaixo da imagem)
TEXT_WINDOW_PT = 60  # pontos (1pt = 1/72 polegada ≈ 0.35mm)

# ─── Estruturas de dados ──────────────────────────────────────────────────────
@dataclass
class ImageCandidate:
    """Uma imagem encontrada em um PDF de norma, candidata a isométrico."""
    pdf_name:         str
    page_num:         int           # 1-indexed
    img_index:        int           # índice na página
    width_px:         int
    height_px:        int
    area_px:          int
    bbox_pt:          list[float]   # [x0, y0, x1, y1] em pontos PDF
    y_center_pt:      float         # posição Y central (para ordenar)
    context_text:     str           # texto adjacente extraído
    piping_classes:   list[str]     # classes identificadas no contexto
    et_refs:          list[str]     # ET refs no contexto
    iso_score:        float         # 0–1, probabilidade de ser isométrico
    iso_keywords:     list[str]     # keywords que geraram o score
    is_multipage:     bool = False  # True se isométrico continua na próxima pág
    uid:              str = ""

    def __post_init__(self):
        if not self.uid:
            h = hashlib.md5(f"{self.pdf_name}:{self.page_num}:{self.img_index}".encode())
            self.uid = h.hexdigest()[:10]


@dataclass
class PageSummary:
    """Resumo de uma página de PDF com suas imagens."""
    page_num:    int
    n_images:    int
    n_large:     int           # imagens acima do threshold
    n_candidates:int           # candidatos a isométrico
    text_sample: str           # primeiros 300 chars do texto da página
    piping_classes: list[str]
    image_sizes: list[tuple]   # [(w,h), ...]


@dataclass
class PDFInspectionResult:
    """Resultado completo da inspeção de um PDF de norma."""
    pdf_name:         str
    pdf_path:         str
    n_pages:          int
    n_images_total:   int
    n_images_large:   int
    n_iso_candidates: int
    candidates:       list[ImageCandidate] = field(default_factory=list)
    page_summaries:   list[PageSummary]    = field(default_factory=list)
    piping_class_map: dict = field(default_factory=dict)  # class → [page_nums]
    et_refs_found:    list[str] = field(default_factory=list)
    inspection_notes: list[str] = field(default_factory=list)


@dataclass
class QdrantNormHit:
    """Resultado de busca no Qdrant (normas públicas)."""
    collection:  str
    score:       float
    text:        str
    source:      str
    page:        int
    piping_classes: list[str]
    iso_keywords:   list[str]


# ─── PARTE 1: Leitura do CSV ──────────────────────────────────────────────────
def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Lê o DescricaodeDocumentos.csv com fallback de encoding.
    Tenta detectar automaticamente separador e encoding.
    """
    for enc in ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']:
        try:
            df = pd.read_csv(csv_path, sep=None, engine='python',
                             encoding=enc, dtype=str)
            df.fillna('', inplace=True)
            console.print(f"[green]CSV lido:[/green] {csv_path.name} | "
                          f"encoding={enc} | shape={df.shape}" if RICH else
                          f"CSV lido: {csv_path.name} | {df.shape}")
            return df
        except Exception:
            continue
    raise RuntimeError(f"Não foi possível ler o CSV: {csv_path}")


def extract_from_csv(df: pd.DataFrame) -> dict:
    """
    Varre todas as colunas do CSV buscando:
    - Referências de ET (I-ET-..., DR-ENGP-...)
    - Piping classes (B10S, B10, B3S, P4X...)
    - Números de SS (093, 116, 312, 522)
    - Tags de linha (LP-448, LP-445...)
    Retorna dict estruturado para uso no pipeline.
    """
    et_refs      = set()
    pipe_classes = set()
    ss_numbers   = set()
    line_tags    = set()
    raw_text     = " ".join(df.astype(str).values.flatten())

    # ET refs
    for m in RE_ET_REF.finditer(raw_text):
        et_refs.add(m.group().strip())

    # Piping classes — apenas as que fazem sentido (2–5 chars, começa com letra)
    for m in RE_PIPING_CLASS.finditer(raw_text):
        pc = m.group()
        if 2 <= len(pc) <= 5:
            pipe_classes.add(pc)

    # SS numbers
    for m in re.finditer(r'SS[-\s]*(\d{2,4})', raw_text, re.I):
        ss_numbers.add(m.group(1).zfill(3))

    # Line tags
    for m in re.finditer(r'(?:LP|LC|LG)-\s*(\d{3,4})', raw_text, re.I):
        line_tags.add(f"{m.group()}")

    # Piping specs completas (2"-F-B10S-200)
    for m in RE_PIPING_SPEC.finditer(raw_text):
        pipe_classes.add(m.group(1))

    result = {
        "et_refs":      sorted(et_refs),
        "piping_classes": sorted(pipe_classes),
        "ss_numbers":   sorted(ss_numbers),
        "line_tags":    sorted(line_tags),
    }

    console.print(f"\n[bold]CSV → Extraído:[/bold]" if RICH else "\nCSV → Extraído:")
    for k, v in result.items():
        console.print(f"  {k}: {v}" if not RICH else f"  [cyan]{k}[/cyan]: {v}")

    return result


# ─── PARTE 2: Inspeção dos PDFs de norma ─────────────────────────────────────
def _score_iso_candidate(img_w: int, img_h: int,
                         context: str, page_text: str) -> tuple[float, list[str]]:
    """
    Calcula score 0–1 de probabilidade de uma imagem ser um isométrico.

    Critérios (pesos):
      0.30 — tamanho: imagens grandes têm mais chance de ser isométrico
      0.30 — keywords ISO no contexto adjacente
      0.20 — piping class identificada no contexto
      0.10 — keywords ISO no texto geral da página
      0.10 — aspect ratio próximo de 4:3 ou 16:9 (isométricos tendem a ser landscape)
    """
    score    = 0.0
    keywords = []
    area     = img_w * img_h

    # Tamanho
    if area > 500_000:   score += 0.30; keywords.append(f"large({img_w}×{img_h})")
    elif area > IMG_MIN_AREA: score += 0.15; keywords.append(f"medium({img_w}×{img_h})")

    # Keywords no contexto adjacente (peso alto)
    ctx_lower = context.lower()
    iso_hits = RE_ISO_KEYWORD.findall(ctx_lower)
    if iso_hits:
        score += min(0.30, len(iso_hits) * 0.10)
        keywords.extend(iso_hits[:3])

    # Piping class no contexto
    if RE_PIPING_CLASS.search(context):
        score += 0.20
        keywords.append("piping_class_found")

    # Keywords na página inteira
    page_hits = RE_ISO_KEYWORD.findall(page_text.lower())
    if page_hits:
        score += min(0.10, len(page_hits) * 0.03)

    # Aspect ratio (isométricos são tipicamente > 1.2 : 1)
    if img_w > 0 and img_h > 0:
        ratio = img_w / img_h
        if 1.2 <= ratio <= 4.0:
            score += 0.10
            keywords.append(f"ratio_ok({ratio:.1f})")

    return min(1.0, score), keywords


def inspect_pdf(pdf_path: Path, pipe_classes_of_interest: set[str]) -> PDFInspectionResult:
    """
    Inspeciona um PDF de norma privada página a página.
    Extrai todas as imagens, mede dimensões, captura contexto textual e
    classifica candidatos a isométrico.
    """
    if not HAS_FITZ:
        console.print(f"[red]PyMuPDF não disponível. Pulando {pdf_path.name}[/red]" if RICH
                      else f"[ERRO] PyMuPDF não disponível. Pulando {pdf_path.name}")
        return PDFInspectionResult(
            pdf_name=pdf_path.name, pdf_path=str(pdf_path),
            n_pages=0, n_images_total=0, n_images_large=0, n_iso_candidates=0,
            inspection_notes=["PyMuPDF não instalado — instale com: pip install pymupdf"]
        )

    result = PDFInspectionResult(
        pdf_name=pdf_path.name,
        pdf_path=str(pdf_path),
        n_pages=0, n_images_total=0, n_images_large=0, n_iso_candidates=0
    )

    console.print(f"\n[bold blue]Inspecionando:[/bold blue] {pdf_path.name}" if RICH
                  else f"\nInspecionando: {pdf_path.name}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        result.inspection_notes.append(f"ERRO ao abrir PDF: {e}")
        return result

    result.n_pages = len(doc)
    piping_class_map: dict[str, list[int]] = defaultdict(list)

    pages_iter = track(range(len(doc)), description=f"  Páginas {pdf_path.name}") \
                 if RICH else range(len(doc))

    for page_idx in pages_iter:
        page      = doc[page_idx]
        page_num  = page_idx + 1
        page_h    = page.rect.height
        page_text = page.get_text("text")

        # Detecta piping classes na página
        page_pipe_classes = list(set(RE_PIPING_CLASS.findall(page_text)))
        relevant_classes  = [pc for pc in page_pipe_classes
                             if pc in pipe_classes_of_interest or len(pc) >= 3]

        for pc in relevant_classes:
            piping_class_map[pc].append(page_num)

        # ET refs na página
        et_in_page = RE_ET_REF.findall(page_text)
        result.et_refs_found.extend(et_in_page)

        # Extrai imagens da página
        img_list = page.get_images(full=True)
        page_candidates = []

        for img_idx, img_info in enumerate(img_list):
            xref = img_info[0]
            result.n_images_total += 1

            # Obtém dimensões da imagem
            try:
                img_dict  = doc.extract_image(xref)
                img_w     = img_dict.get("width", 0)
                img_h_img = img_dict.get("height", 0)
            except Exception:
                img_w, img_h_img = 0, 0

            area = img_w * img_h_img

            # Obtém bounding box da imagem na página
            img_rects = page.get_image_rects(xref)
            if img_rects:
                bbox = list(img_rects[0])
                y_center = (bbox[1] + bbox[3]) / 2
            else:
                # Fallback: estima posição pelo índice
                bbox     = [0, page_h * img_idx / max(len(img_list), 1),
                            page.rect.width,
                            page_h * (img_idx + 1) / max(len(img_list), 1)]
                y_center = (bbox[1] + bbox[3]) / 2

            # Filtra imagens pequenas
            if area < IMG_MIN_AREA or img_w < IMG_MIN_WIDTH or img_h_img < IMG_MIN_HEIGHT:
                continue

            result.n_images_large += 1

            # Extrai texto adjacente na janela Y ± TEXT_WINDOW_PT
            y0_win = max(0,       y_center - TEXT_WINDOW_PT)
            y1_win = min(page_h,  y_center + TEXT_WINDOW_PT)

            # Texto acima da imagem (legenda/identificação)
            blocks_above = []
            blocks_below = []
            for blk in page.get_text("blocks"):
                bx0, by0, bx1, by1, btext, *_ = blk
                if by1 < bbox[1] and by1 > (bbox[1] - TEXT_WINDOW_PT * 2):
                    blocks_above.append(btext.strip())
                if by0 > bbox[3] and by0 < (bbox[3] + TEXT_WINDOW_PT * 2):
                    blocks_below.append(btext.strip())

            context = " | ".join(filter(None, blocks_above[-3:] + blocks_below[:3]))
            if not context:
                # Fallback: texto da janela Y completa
                context = page.get_text("text", clip=fitz.Rect(0, y0_win,
                                                                 page.rect.width, y1_win))
            context = context[:400]

            # Piping classes no contexto
            ctx_classes = list(set(RE_PIPING_CLASS.findall(context)))

            # Score de isométrico
            score, kw = _score_iso_candidate(img_w, img_h_img, context, page_text)

            candidate = ImageCandidate(
                pdf_name      = pdf_path.name,
                page_num      = page_num,
                img_index     = img_idx,
                width_px      = img_w,
                height_px     = img_h_img,
                area_px       = area,
                bbox_pt       = bbox,
                y_center_pt   = y_center,
                context_text  = context,
                piping_classes= ctx_classes,
                et_refs       = RE_ET_REF.findall(context),
                iso_score     = score,
                iso_keywords  = kw,
            )
            page_candidates.append(candidate)

            if score >= 0.3:  # threshold para "candidato relevante"
                result.n_iso_candidates += 1
                result.candidates.append(candidate)

        # Resumo da página
        result.page_summaries.append(PageSummary(
            page_num      = page_num,
            n_images      = len(img_list),
            n_large       = len(page_candidates),
            n_candidates  = sum(1 for c in page_candidates if c.iso_score >= 0.3),
            text_sample   = page_text[:300].replace('\n', ' '),
            piping_classes= relevant_classes,
            image_sizes   = [(c.width_px, c.height_px) for c in page_candidates],
        ))

    # Verifica isométricos multi-página (imagem grande em páginas consecutivas)
    candidate_pages = sorted(set(c.page_num for c in result.candidates))
    for i, p in enumerate(candidate_pages[:-1]):
        if candidate_pages[i+1] == p + 1:
            for c in result.candidates:
                if c.page_num == p:
                    c.is_multipage = True

    result.piping_class_map = {k: sorted(set(v)) for k, v in piping_class_map.items()}
    result.et_refs_found    = sorted(set(result.et_refs_found))
    doc.close()

    # Notas automáticas
    if result.n_iso_candidates == 0:
        result.inspection_notes.append(
            "ATENÇÃO: Nenhum candidato a isométrico encontrado. "
            "Verifique se os thresholds IMG_MIN_WIDTH/HEIGHT estão corretos "
            "para este PDF, ou se o isométrico está em PDF diferente."
        )
    if result.n_images_total > 0 and result.n_images_large == 0:
        result.inspection_notes.append(
            f"Todas as {result.n_images_total} imagens são menores que "
            f"{IMG_MIN_WIDTH}×{IMG_MIN_HEIGHT}px. "
            "Considere reduzir o threshold mínimo."
        )

    console.print(
        f"  → {result.n_pages} págs | {result.n_images_total} imgs total | "
        f"{result.n_images_large} grandes | {result.n_iso_candidates} candidatos ISO"
    )
    return result


# ─── PARTE 3: Consulta ao Qdrant (normas públicas) ───────────────────────────
def query_qdrant_for_isos(
    pipe_classes: list[str],
    et_refs: list[str],
    qdrant_url: str,
    api_key: Optional[str],
    collection: str,
) -> list[QdrantNormHit]:
    """
    Busca no Qdrant por chunks de normas públicas que referenciam
    piping classes + keywords de isométrico.

    Estratégia: scroll sem filtro (evita erro de índice ausente) +
    filtragem de keywords em Python sobre o payload retornado.
    """
    if not HAS_QDRANT:
        console.print("[AVISO] Qdrant client não disponível.")
        return []

    try:
        client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=10)
        cols = [c.name for c in client.get_collections().collections]
        if collection not in cols:
            console.print(f"[AVISO] Coleção '{collection}' não existe. Disponíveis: {cols}")
            return []
    except Exception as e:
        console.print(f"[ERRO] Qdrant inacessível em {qdrant_url}: {e}")
        return []

    # Descobre qual campo contém o texto nos payloads da coleção
    # (varia por ingestor: "text", "page_content", "content", "chunk")
    TEXT_FIELD_CANDIDATES = ["text", "page_content", "content", "chunk", "documento"]

    console.print(f"\nQdrant: scrolling coleção '{collection}' (sem filtro, busca em Python)...")

    hits   = []
    offset = None
    batch  = 100   # registros por página de scroll
    total_scanned = 0

    pipe_classes_upper = [p.upper() for p in pipe_classes]

    while True:
        try:
            results, next_offset = client.scroll(
                collection_name = collection,
                limit           = batch,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
                # SEM filtro — Qdrant aceita scroll sem índice
            )
        except Exception as e:
            console.print(f"  Scroll falhou: {e}")
            break

        if not results:
            break

        for r in results:
            total_scanned += 1
            payload = r.payload or {}

            # Detecta o campo de texto automaticamente
            text = ""
            for field in TEXT_FIELD_CANDIDATES:
                if field in payload and payload[field]:
                    text = str(payload[field])
                    break
            if not text:
                # Fallback: junta todos os valores string do payload
                text = " ".join(str(v) for v in payload.values() if isinstance(v, str))

            text_upper = text.upper()

            # Filtra em Python: precisa ter piping class OU keyword ISO
            iso_kw    = RE_ISO_KEYWORD.findall(text.lower())
            pc_found  = [p for p in pipe_classes_upper if p in text_upper]
            et_found  = [e for e in et_refs if e.upper() in text_upper]

            if not (iso_kw or pc_found or et_found):
                continue

            hits.append(QdrantNormHit(
                collection     = collection,
                score          = 0.0,
                text           = text[:300],
                source         = payload.get("source",
                                 payload.get("filename",
                                 payload.get("doc_id", "unknown"))),
                page           = payload.get("page",
                                 payload.get("page_num",
                                 payload.get("pagina", 0))),
                piping_classes = pc_found,
                iso_keywords   = iso_kw,
            ))

        offset = next_offset
        if offset is None:
            break   # scroll completo

        # Limite de segurança para coleções muito grandes
        if total_scanned >= 5000:
            console.print(f"  Limite de segurança atingido: {total_scanned} registros varridos.")
            break

    console.print(f"  → {total_scanned} chunks varridos | {len(hits)} hits relevantes")
    return hits


# ─── PARTE 4: Geração do Relatório ───────────────────────────────────────────
def _render_size_distribution(results: list[PDFInspectionResult]) -> str:
    """Monta histograma de tamanhos de imagens para calibrar thresholds."""
    sizes = []
    for r in results:
        for c in r.candidates:
            sizes.append(c.area_px)
    if not sizes:
        return "Nenhuma imagem acima do threshold encontrada."

    sizes.sort()
    p25  = sizes[len(sizes)//4]
    p50  = sizes[len(sizes)//2]
    p75  = sizes[3*len(sizes)//4]
    pmax = sizes[-1]
    pmin = sizes[0]

    return (f"min={pmin:,} | P25={p25:,} | P50={p50:,} | "
            f"P75={p75:,} | max={pmax:,} pixels²")


def generate_json_report(
    csv_extracted: dict,
    pdf_results: list[PDFInspectionResult],
    qdrant_hits: list[QdrantNormHit],
    output_path: Path,
) -> dict:
    """Gera inspection_report.json com todos os dados estruturados."""
    report = {
        "generated_at":    datetime.now().isoformat(),
        "config": {
            "img_min_width":  IMG_MIN_WIDTH,
            "img_min_height": IMG_MIN_HEIGHT,
            "img_min_area":   IMG_MIN_AREA,
            "text_window_pt": TEXT_WINDOW_PT,
        },
        "csv_extract":     csv_extracted,
        "pdf_inspections": [],
        "qdrant_hits":     [],
        "recommendations": [],
    }

    # PDFs
    for r in pdf_results:
        entry = {
            "pdf_name":          r.pdf_name,
            "pdf_path":          r.pdf_path,
            "n_pages":           r.n_pages,
            "n_images_total":    r.n_images_total,
            "n_images_large":    r.n_images_large,
            "n_iso_candidates":  r.n_iso_candidates,
            "et_refs_found":     r.et_refs_found,
            "piping_class_map":  r.piping_class_map,
            "inspection_notes":  r.inspection_notes,
            "top_candidates": [
                {
                    "uid":           c.uid,
                    "page_num":      c.page_num,
                    "size_px":       f"{c.width_px}×{c.height_px}",
                    "area_px":       c.area_px,
                    "iso_score":     round(c.iso_score, 3),
                    "iso_keywords":  c.iso_keywords,
                    "piping_classes":c.piping_classes,
                    "context_text":  c.context_text[:200],
                    "is_multipage":  c.is_multipage,
                    "bbox_pt":       [round(x, 1) for x in c.bbox_pt],
                }
                for c in sorted(r.candidates, key=lambda x: x.iso_score, reverse=True)[:20]
            ],
            "page_summaries": [
                {
                    "page":          s.page_num,
                    "n_images":      s.n_images,
                    "n_candidates":  s.n_candidates,
                    "piping_classes":s.piping_classes,
                    "image_sizes":   s.image_sizes,
                    "text_sample":   s.text_sample[:150],
                }
                for s in r.page_summaries if s.n_images > 0 or s.piping_classes
            ],
        }
        report["pdf_inspections"].append(entry)

    # Qdrant hits
    for h in qdrant_hits[:30]:
        report["qdrant_hits"].append({
            "source":       h.source,
            "page":         h.page,
            "piping_classes": h.piping_classes,
            "iso_keywords": h.iso_keywords,
            "text_sample":  h.text[:200],
        })

    # Recomendações automáticas
    recs = []

    # Threshold
    all_candidates = [c for r in pdf_results for c in r.candidates]
    if all_candidates:
        min_area = min(c.area_px for c in all_candidates)
        max_area = max(c.area_px for c in all_candidates)
        recs.append({
            "type": "THRESHOLD_CALIBRATION",
            "msg":  f"Área das imagens candidatas: {min_area:,} – {max_area:,} px². "
                    f"Sugestão de IMG_MIN_AREA para image_ingestor.py: {min_area:,}"
        })

    # Multi-página
    multipage = [c for c in all_candidates if c.is_multipage]
    if multipage:
        recs.append({
            "type": "MULTIPAGE_ISO",
            "msg":  f"{len(multipage)} isométricos multi-página detectados. "
                    "image_ingestor.py deve suportar junção de páginas consecutivas."
        })

    # Piping classes sem mapeamento
    all_mapped_classes = set()
    for r in pdf_results:
        all_mapped_classes.update(r.piping_class_map.keys())
    unmapped = set(csv_extracted.get("piping_classes", [])) - all_mapped_classes
    if unmapped:
        recs.append({
            "type": "UNMAPPED_CLASSES",
            "msg":  f"Classes {sorted(unmapped)} mencionadas no CSV mas não encontradas "
                    "nos PDFs privados. Verificar se estão em norma pública."
        })

    # Normas públicas relevantes
    pub_sources = sorted(set(h.source for h in qdrant_hits if h.iso_keywords))
    if pub_sources:
        recs.append({
            "type": "PUBLIC_NORMS_TO_INGEST",
            "msg":  f"Normas públicas com possível isométrico: {pub_sources}. "
                    "PDFs originais precisam estar acessíveis para image_ingestor.py."
        })

    report["recommendations"] = recs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]JSON salvo:[/green] {output_path}" if RICH
                  else f"\nJSON salvo: {output_path}")
    return report


def generate_markdown_report(report: dict, output_path: Path) -> None:
    """Gera inspection_report.md legível para revisão manual."""
    lines = []
    ts    = report["generated_at"]
    cfg   = report["config"]

    lines += [
        "# ETAPA 1 — Relatório de Inspeção de Normas",
        f"Gerado em: {ts}",
        "",
        "## Configuração Usada",
        f"- `IMG_MIN_WIDTH`: {cfg['img_min_width']} px",
        f"- `IMG_MIN_HEIGHT`: {cfg['img_min_height']} px",
        f"- `IMG_MIN_AREA`: {cfg['img_min_area']:,} px²",
        f"- `TEXT_WINDOW_PT`: {cfg['text_window_pt']} pt (janela de texto adjacente)",
        "",
        "## Dados Extraídos do CSV",
    ]

    csv_ext = report["csv_extract"]
    for k, v in csv_ext.items():
        lines.append(f"- **{k}**: {', '.join(v) if v else '(nenhum)'}")

    lines += ["", "---", "## Inspeção dos PDFs de Normas Privadas", ""]

    for insp in report["pdf_inspections"]:
        lines += [
            f"### {insp['pdf_name']}",
            f"- Páginas: {insp['n_pages']}",
            f"- Imagens totais: {insp['n_images_total']}",
            f"- Imagens grandes (acima threshold): {insp['n_images_large']}",
            f"- Candidatos a isométrico: {insp['n_iso_candidates']}",
            f"- ET refs encontradas: {', '.join(insp['et_refs_found']) or '(nenhuma)'}",
            "",
        ]

        if insp["piping_class_map"]:
            lines.append("#### Mapa Piping Class → Páginas")
            lines.append("| Piping Class | Páginas |")
            lines.append("|---|---|")
            for pc, pages in sorted(insp["piping_class_map"].items()):
                lines.append(f"| {pc} | {', '.join(str(p) for p in pages)} |")
            lines.append("")

        if insp["top_candidates"]:
            lines.append("#### Top Candidatos a Isométrico")
            lines.append("| Pág | Tamanho | Score | Piping Class | Keywords | Multi-pág |")
            lines.append("|---|---|---|---|---|---|")
            for c in insp["top_candidates"][:10]:
                lines.append(
                    f"| {c['page_num']} | {c['size_px']} | {c['iso_score']:.2f} | "
                    f"{', '.join(c['piping_classes']) or '—'} | "
                    f"{', '.join(c['iso_keywords'][:3])} | "
                    f"{'Sim' if c['is_multipage'] else 'Não'} |"
                )
            lines.append("")
            lines.append("#### Contexto Textual dos Melhores Candidatos")
            for c in insp["top_candidates"][:5]:
                lines += [
                    f"**Pág {c['page_num']} · score {c['iso_score']:.2f}:**",
                    f"> {c['context_text'][:300]}",
                    "",
                ]

        if insp["inspection_notes"]:
            lines.append("#### ⚠ Notas de Inspeção")
            for note in insp["inspection_notes"]:
                lines.append(f"- {note}")
            lines.append("")

    # Qdrant
    lines += ["---", "## Normas Públicas no Qdrant (Busca de Contexto)", ""]
    if report["qdrant_hits"]:
        lines.append("| Fonte | Pág | Piping Classes | Keywords ISO |")
        lines.append("|---|---|---|---|")
        for h in report["qdrant_hits"][:15]:
            lines.append(
                f"| {h['source']} | {h['page']} | "
                f"{', '.join(h['piping_classes'][:3])} | "
                f"{', '.join(h['iso_keywords'][:3])} |"
            )
    else:
        lines.append("Nenhum hit relevante encontrado (Qdrant inacessível ou sem resultados).")
    lines.append("")

    # Recomendações
    lines += ["---", "## Recomendações para image_ingestor.py", ""]
    for rec in report["recommendations"]:
        lines += [f"### {rec['type']}", rec['msg'], ""]

    # Responde as 5 perguntas da ETAPA 1
    lines += [
        "---",
        "## Respostas às 5 Perguntas da ETAPA 1",
        "",
        "### 1. Em que páginas os isométricos aparecem?",
    ]
    for insp in report["pdf_inspections"]:
        pages = sorted(set(c["page_num"] for c in insp["top_candidates"] if c["iso_score"] >= 0.5))
        lines.append(f"- **{insp['pdf_name']}**: páginas com score ≥ 0.5 → {pages or 'nenhuma encontrada'}")
    lines += [
        "",
        "### 2. Qual o tamanho típico das imagens de isométrico?",
    ]
    all_candidates = [c for insp in report["pdf_inspections"] for c in insp["top_candidates"]]
    if all_candidates:
        areas = sorted(c["area_px"] for c in all_candidates)
        lines.append(f"- Distribuição de área: {areas[0]:,} – {areas[-1]:,} px²")
        lines.append(f"- Tamanhos encontrados: {sorted(set(c['size_px'] for c in all_candidates))}")
        lines.append(f"- **Sugestão de threshold:** IMG_MIN_AREA = {areas[0]:,} px²")
    else:
        lines.append("- Nenhum candidato encontrado acima dos thresholds atuais.")
    lines += [
        "",
        "### 3. O isométrico de uma piping class ocupa uma ou múltiplas páginas?",
    ]
    multipage_count = sum(1 for insp in report["pdf_inspections"]
                          for c in insp["top_candidates"] if c["is_multipage"])
    lines.append(f"- Candidatos multi-página detectados: {multipage_count}")
    lines.append("- Verificar manualmente as páginas consecutivas com score alto.")
    lines += [
        "",
        "### 4. O texto adjacente contém a identificação da piping class?",
    ]
    with_class = sum(1 for insp in report["pdf_inspections"]
                     for c in insp["top_candidates"] if c["piping_classes"])
    total_c    = sum(len(insp["top_candidates"]) for insp in report["pdf_inspections"])
    lines.append(f"- {with_class} de {total_c} candidatos têm piping class no contexto adjacente.")
    lines.append("- Se < 50%, a associação piping_class→imagem precisará de heurística adicional.")
    lines += [
        "",
        "### 5. Existe tabela de correspondência piping_class → página?",
    ]
    for insp in report["pdf_inspections"]:
        if insp["piping_class_map"]:
            lines.append(f"- **{insp['pdf_name']}**: {insp['piping_class_map']}")
        else:
            lines.append(f"- **{insp['pdf_name']}**: nenhuma correspondência direta encontrada.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding='utf-8')
    console.print(f"[green]Markdown salvo:[/green] {output_path}" if RICH
                  else f"Markdown salvo: {output_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    global IMG_MIN_WIDTH, IMG_MIN_HEIGHT, IMG_MIN_AREA   # ← primeira linha

    parser = argparse.ArgumentParser(
        description="ETAPA 1 — Inspeção de normas e mapeamento de isométricos"
    )
    parser.add_argument("--csv",          default=str(CSV_PATH),
                        help="Path do DescricaodeDocumentos.csv")
    parser.add_argument("--normas-dir",   default=str(NORMAS_PRIVADAS),
                        help="Diretório das normas privadas (PDFs)")
    parser.add_argument("--output-dir",   default=str(OUTPUT_DIR),
                        help="Diretório de saída dos relatórios")
    parser.add_argument("--qdrant-url",   default=QDRANT_URL)
    parser.add_argument("--qdrant-key",   default=QDRANT_API_KEY)
    parser.add_argument("--qdrant-col",   default=COLLECTION_NAME)
    parser.add_argument("--skip-qdrant",  action="store_true",
                        help="Pula consulta ao Qdrant")
    parser.add_argument("--pdf-only",     default=None,
                        help="Inspeciona apenas este PDF (nome do arquivo)")
    parser.add_argument("--min-width",    type=int, default=IMG_MIN_WIDTH)
    parser.add_argument("--min-height",   type=int, default=IMG_MIN_HEIGHT)
    args = parser.parse_args()

    # Aplica overrides de threshold
    IMG_MIN_WIDTH  = args.min_width
    IMG_MIN_HEIGHT = args.min_height
    IMG_MIN_AREA   = IMG_MIN_WIDTH * IMG_MIN_HEIGHT

    output_dir   = Path(args.output_dir)
    normas_dir   = Path(args.normas_dir)
    csv_path     = Path(args.csv)

    console.rule("ETAPA 1 — Inspeção de Normas e Mapeamento de Isométricos")

    # ── 1. Lê o CSV ──────────────────────────────────────────────────────────
    csv_extracted = {"et_refs": [], "piping_classes": [], "ss_numbers": [], "line_tags": []}
    if csv_path.exists():
        df = load_csv(csv_path)
        csv_extracted = extract_from_csv(df)
    else:
        console.print(f"[yellow]CSV não encontrado: {csv_path}. "
                      "Usando piping classes padrão do POC.[/yellow]" if RICH else
                      f"[AVISO] CSV não encontrado: {csv_path}. Usando padrão POC.")
        # Fallback: classes conhecidas do POC
        csv_extracted["piping_classes"] = ["B10S", "B10", "B3S", "P4X"]
        csv_extracted["et_refs"] = [
            "I-ET-3010.68-1200-200-JUR-001",
            "I-ET-3000.00-1200-200-P4X-001",
            "DR-ENGP-1.1-R14",
        ]

    pipe_classes_set = set(csv_extracted["piping_classes"])

    # ── 2. Inspeciona PDFs privados ───────────────────────────────────────────
    pdf_results: list[PDFInspectionResult] = []

    if not normas_dir.exists():
        console.print(f"[red]Diretório não encontrado: {normas_dir}[/red]" if RICH
                      else f"[ERRO] Diretório não encontrado: {normas_dir}")
    else:
        pdf_files = sorted(normas_dir.glob("*.pdf"))
        if args.pdf_only:
            pdf_files = [f for f in pdf_files if args.pdf_only in f.name]

        if not pdf_files:
            console.print("[yellow]Nenhum PDF encontrado no diretório de normas.[/yellow]" if RICH
                          else "[AVISO] Nenhum PDF encontrado.")
        else:
            console.print(f"\nEncontrados {len(pdf_files)} PDFs: "
                          f"{[f.name for f in pdf_files]}")
            for pdf_path in pdf_files:
                result = inspect_pdf(pdf_path, pipe_classes_set)
                pdf_results.append(result)

    # ── 3. Consulta Qdrant ────────────────────────────────────────────────────
    qdrant_hits: list[QdrantNormHit] = []
    if not args.skip_qdrant and HAS_QDRANT:
        qdrant_hits = query_qdrant_for_isos(
            pipe_classes = csv_extracted["piping_classes"],
            et_refs      = csv_extracted["et_refs"],
            qdrant_url   = args.qdrant_url,
            api_key      = args.qdrant_key,
            collection   = args.qdrant_col,
        )

    # ── 4. Gera relatórios ────────────────────────────────────────────────────
    console.rule("Gerando Relatórios")

    json_path = output_dir / "inspection_report.json"
    md_path   = output_dir / "inspection_report.md"

    report = generate_json_report(csv_extracted, pdf_results, qdrant_hits, json_path)
    generate_markdown_report(report, md_path)

    # ── 5. Sumário final no terminal ──────────────────────────────────────────
    console.rule("Sumário Final")
    total_candidates = sum(r.n_iso_candidates for r in pdf_results)
    total_images     = sum(r.n_images_total   for r in pdf_results)
    total_large      = sum(r.n_images_large   for r in pdf_results)

    print(f"""
  PDFs inspecionados  : {len(pdf_results)}
  Imagens encontradas : {total_images}
  Imagens grandes     : {total_large}
  Candidatos ISO      : {total_candidates}
  Hits Qdrant         : {len(qdrant_hits)}

  Relatórios salvos em: {output_dir}
    ├── inspection_report.json  (para image_ingestor.py)
    └── inspection_report.md   (para revisão manual)
    """)

    if total_candidates == 0 and pdf_results:
        print("""
  ⚠  NENHUM CANDIDATO ENCONTRADO.
     Ações recomendadas:
       1. Abra os PDFs manualmente e identifique a página de um isométrico.
       2. Meça a imagem em pixels e ajuste IMG_MIN_WIDTH/HEIGHT:
            python etapa1_inspect_normas.py --min-width 150 --min-height 100
       3. Verifique se as imagens estão embutidas (vetoriais) ou escaneadas.
          PDFs vetoriais podem não ter imagens extraíveis pelo PyMuPDF.
        """)


if __name__ == "__main__":
    main()