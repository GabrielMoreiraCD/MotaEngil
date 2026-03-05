"""
Extrator de chunks para livros técnicos de referência.
Suporta PDFs com texto nativo e PDFs rasterizados (scan) via Pytesseract.
Detecção automática do tipo de PDF — sem configuração manual.
"""

import io
import os
import logging
import re
import unicodedata
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

log = logging.getLogger(__name__)

# ─── Configurações globais ────────────────────────────────────────────────────

# Configuração do executável Tesseract (Caminho absoluto)
TESSERACT_PATH = r'C:\Users\gabriel.moreira\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

OCR_QUALITY_THRESHOLD = 0.45   # chunks abaixo disso são descartados
PAGES_PER_OCR_CHUNK   = 2      # agrupa N páginas por chunk em PDFs scaneados
OCR_DPI               = 300    # resolução de renderização para Tesseract

# Limites de posição Y para filtrar cabeçalho/rodapé (pixels, 72dpi base)
Y_HEADER_LIMIT = 55
Y_FOOTER_LIMIT = 785

# ─── Dataclass ───────────────────────────────────────────────────────────────

@dataclass
class LivroChunk:
    fonte: str           # ex: "TELLES_TUBULACOES_INDUSTRIAIS"
    capitulo: str        # título do capítulo detectado
    subtitulo: str       # título da seção/subcapítulo
    texto: str           # texto bruto (pré-limpeza)
    texto_limpo: str     # texto pós-limpeza OCR
    pagina: int
    idioma: str          # "pt" | "en"
    qualidade_ocr: float # 0.0 (lixo) a 1.0 (perfeito)

# ─── Regex de estrutura ──────────────────────────────────────────────────────

_RE_CAPITULO_PT = re.compile(
    r'^(\d{1,2})\s+([A-ZÁÉÍÓÚÀÃÕÂÊÔÇ][A-ZÁÉÍÓÚÀÃÕÂÊÔÇ\s]{5,50})$',
    re.MULTILINE,
)
_RE_CAPITULO_EN = re.compile(
    r'^(Chapter\s+\d+|CHAPTER\s+\d+|SECTION\s+\d+)\s*[:\-]?\s*(.+)$',
    re.MULTILINE,
)
_RE_SUBCAPITULO = re.compile(
    r'^(\d{1,2}\.\d{1,3})\s+(.{5,60})$',
    re.MULTILINE,
)
_RE_PAGE_HEADER = re.compile(
    r'^\d+\s*/\s*TUBULAÇÕES|^Pressure Vessel Design Manual\s*\d*$',
    re.MULTILINE | re.IGNORECASE,
)
_RE_OCR_NOISE   = re.compile(
    r'[^\w\s\d\.\,\;\:\!\?\-\(\)\[\]\/\%\°\"\'\+\=\<\>\*\&\#\@'
    r'áéíóúàãõâêôçÁÉÍÓÚÀÃÕÂÊÔÇüÜ]+',
)
_RE_MULTI_SPACE  = re.compile(r'[ \t]{2,}')
_RE_BROKEN_WORD  = re.compile(r'(\w)-\s*\n\s*(\w)')
_RE_LONE_CHAR    = re.compile(r'(?<!\w)[a-zA-Z](?!\w)')

# ─── Utilitários de texto ─────────────────────────────────────────────────────

def score_ocr_quality(text: str) -> float:
    if not text or len(text) < 20:
        return 0.0
    total = len(text)
    valid = len(re.findall(r'[a-zA-ZáéíóúàãõâêôçÀ-ÿ0-9\s\.,;:\-()\[\]]', text))
    base = valid / total
    no_vowel = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}', text)
    penalty = min(0.4, len(no_vowel) * 0.05)
    return max(0.0, base - penalty)


def clean_ocr_text(text: str) -> str:
    text = _RE_PAGE_HEADER.sub('', text)
    text = _RE_BROKEN_WORD.sub(r'\1\2', text)
    text = _RE_OCR_NOISE.sub(' ', text)
    text = _RE_LONE_CHAR.sub(' ', text)
    text = _RE_MULTI_SPACE.sub(' ', text)
    text = unicodedata.normalize('NFC', text)
    return text.strip()


def detect_language(text: str) -> str:
    pt = len(re.findall(
        r'\b(tubulação|norma|pressão|espessura|fluido|soldagem|montagem|fabricação|'
        r'cálculo|diâmetro|temperatura|seção|válvula|parede)\b', text, re.I))
    en = len(re.findall(
        r'\b(vessel|pressure|design|manual|piping|thickness|welding|fabrication|'
        r'calculation|diameter|temperature|section|valve|wall)\b', text, re.I))
    return "pt" if pt >= en else "en"

# ─── Detecção de tipo de PDF ──────────────────────────────────────────────────

def _is_scanned_pdf(doc: fitz.Document, sample_pages: int = 5) -> bool:
    pages_to_check = min(sample_pages, len(doc))
    text_pages = 0
    for i in range(pages_to_check):
        if len(doc[i].get_text("text").strip()) > 50:
            text_pages += 1
    return text_pages < pages_to_check * 0.3

# ─── Pré-processamento de imagem para OCR ────────────────────────────────────

def _preprocess_for_ocr(pix: fitz.Pixmap) -> Image.Image:
    try:
        from scipy.ndimage import uniform_filter
        import numpy as np

        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        arr = np.array(img, dtype=np.float32)
        blurred = uniform_filter(arr, size=5)
        threshold = blurred - 10
        binary = (arr > threshold).astype(np.uint8) * 255
        return Image.fromarray(binary)

    except ImportError:
        log.warning("scipy não disponível — usando imagem grayscale sem threshold adaptativo")
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")


def _ocr_page(doc: fitz.Document, page_idx: int, lang: str = "por+eng") -> str:
    # Verificação defensiva do executável antes de chamar a lib
    if not os.path.exists(TESSERACT_PATH):
        raise FileNotFoundError(f"Executável do Tesseract não encontrado em: {TESSERACT_PATH}")

    page = doc[page_idx]
    mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = _preprocess_for_ocr(pix)
    config = "--oem 3 --psm 6 -c tessedit_char_blacklist=|{}~`"
    return pytesseract.image_to_string(img, lang=lang, config=config)

# ─── Extratores por tipo de PDF ───────────────────────────────────────────────

def _extract_native_pdf(
    doc: fitz.Document,
    fonte_id: str,
    min_chunk_chars: int,
) -> list[LivroChunk]:
    chunks: list[LivroChunk] = []
    current_capitulo = "Introdução"
    current_sub      = ""
    buffer: list[str] = []
    buffer_pagina    = 1

    def flush():
        nonlocal buffer, buffer_pagina
        if not buffer:
            return
        raw   = " ".join(buffer).strip()
        limpo = clean_ocr_text(raw)
        q     = score_ocr_quality(limpo)
        if q >= OCR_QUALITY_THRESHOLD and len(limpo) >= min_chunk_chars:
            chunks.append(LivroChunk(
                fonte=fonte_id,
                capitulo=current_capitulo,
                subtitulo=current_sub,
                texto=raw,
                texto_limpo=limpo,
                pagina=buffer_pagina,
                idioma=detect_language(limpo),
                qualidade_ocr=q,
            ))
        buffer.clear()

    for page in doc:
        for block in page.get_text("blocks", sort=True):
            x0, y0, x1, y1, text, _, btype = block
            if y0 < Y_HEADER_LIMIT or y1 > Y_FOOTER_LIMIT or btype != 0:
                continue
            text = text.strip()
            if not text:
                continue

            first_line = text.split('\n')[0].strip()

            m_cap = _RE_CAPITULO_PT.match(first_line) or _RE_CAPITULO_EN.match(first_line)
            if m_cap:
                flush()
                current_capitulo = (
                    m_cap.group(2).strip() if len(m_cap.groups()) >= 2 else first_line
                )
                current_sub = ""
                buffer.append(text)
                buffer_pagina = page.number + 1
                continue

            m_sub = _RE_SUBCAPITULO.match(first_line)
            if m_sub:
                flush()
                current_sub   = m_sub.group(2).strip()
                buffer.append(text)
                buffer_pagina = page.number + 1
                continue

            buffer.append(text)
            if len(buffer) == 1:
                buffer_pagina = page.number + 1

        flush()

    return chunks


def _extract_scanned_pdf(
    doc: fitz.Document,
    fonte_id: str,
    min_chunk_chars: int,
    ocr_lang: str,
) -> list[LivroChunk]:
    chunks: list[LivroChunk] = []
    current_capitulo = "Introdução"
    current_sub      = ""
    buffer_pages: list[str] = []
    buffer_start     = 1
    total            = len(doc)

    def flush_buffer():
        nonlocal buffer_pages, buffer_start
        if not buffer_pages:
            return
        texto = " ".join(buffer_pages).strip()
        if len(texto) < min_chunk_chars:
            buffer_pages.clear()
            return
        q = score_ocr_quality(texto)
        if q >= OCR_QUALITY_THRESHOLD:
            chunks.append(LivroChunk(
                fonte=fonte_id,
                capitulo=current_capitulo,
                subtitulo=current_sub,
                texto=texto,
                texto_limpo=texto,
                pagina=buffer_start,
                idioma=detect_language(texto),
                qualidade_ocr=q,
            ))
        buffer_pages.clear()

    for idx in range(total):
        if idx % 10 == 0:
            log.info(f"  OCR página {idx + 1}/{total}...")

        raw  = _ocr_page(doc, idx, lang=ocr_lang)
        limpo = clean_ocr_text(raw)

        if not limpo or len(limpo) < 30:
            continue

        for line in limpo.split('\n')[:5]:
            line = line.strip()
            m = _RE_CAPITULO_PT.match(line) or _RE_CAPITULO_EN.match(line)
            if m:
                flush_buffer()
                current_capitulo = (
                    m.group(2).strip() if len(m.groups()) >= 2 else line
                )
                current_sub  = ""
                buffer_start = idx + 1
                break
            m_sub = _RE_SUBCAPITULO.match(line)
            if m_sub:
                current_sub = m_sub.group(2).strip()

        if not buffer_pages:
            buffer_start = idx + 1
        buffer_pages.append(limpo)

        if len(buffer_pages) >= PAGES_PER_OCR_CHUNK:
            flush_buffer()

    flush_buffer()
    return chunks

# ─── Ponto de entrada público ─────────────────────────────────────────────────

def extract_livro_chunks_with_ocr_fallback(
    pdf_path: str,
    fonte_id: str,
    min_chunk_chars: int = 150,
    ocr_lang: str = "por+eng",
) -> list[LivroChunk]:
    doc = fitz.open(pdf_path)
    scanned = _is_scanned_pdf(doc)

    if scanned:
        log.info(f"  [{fonte_id}] PDF rasterizado → Pytesseract (lang={ocr_lang})")
        # Validação preventiva para evitar falha no meio do processo
        if not os.path.exists(TESSERACT_PATH):
            doc.close()
            raise FileNotFoundError(f"Erro crítico: Tesseract não encontrado em {TESSERACT_PATH}")
            
        result = _extract_scanned_pdf(doc, fonte_id, min_chunk_chars, ocr_lang)
    else:
        log.info(f"  [{fonte_id}] PDF nativo → extração direta fitz")
        result = _extract_native_pdf(doc, fonte_id, min_chunk_chars)

    doc.close()
    log.info(f"  [{fonte_id}] {len(result)} chunks extraídos")
    return result