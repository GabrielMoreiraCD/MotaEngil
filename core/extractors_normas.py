# core/extractors_normas.py
import fitz
import re
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class NormaChunk:
    norma_id: str          # "N-115"
    secao: str             # "4.2.3"
    titulo_secao: str      # "Espessuras Mínimas"
    tipo: str              # "texto" | "tabela" | "figura" | "nota"
    texto: str
    pagina: int
    bbox: Optional[list] = None
    tags: list = field(default_factory=list)

# Padrões estruturais de normas PETROBRAS/ABNT
_RE_SECAO      = re.compile(r'^(\d{1,2}(?:\.\d{1,2}){0,3})\s+(.+)$')
_RE_TABELA     = re.compile(r'^Tabela\s+[\dA-Z\.]+\s*[-–]', re.IGNORECASE)
_RE_FIGURA     = re.compile(r'^Figura\s+[\dA-Z\.]+\s*[-–]', re.IGNORECASE)
_RE_NOTA       = re.compile(r'^NOTA\s*\d*\s', re.IGNORECASE)
_RE_NPS        = re.compile(r'\b(\d+(?:[/\s]\d+)?)\s*(?:pol(?:egadas?)?|"|NPS|DN)\b', re.IGNORECASE)
_RE_NORMA_REF  = re.compile(r'\bN-\d{3,4}\b|\bNBR\s*\d+\b|\bASME\s+B\s*\d+\b|\bASTM\s+[A-Z]\s*\d+\b')

Y_HEADER_LIMIT = 60   # px — ignora faixa de cabeçalho
Y_FOOTER_LIMIT = 780  # px — ignora faixa de rodapé

def extract_norma_chunks(pdf_path: str, norma_id: str) -> list[NormaChunk]:
    """
    Extrai chunks hierárquicos de uma norma técnica PDF.
    Preserva contexto de seção para cada chunk.
    Vincula tabelas/figuras ao título imediatamente anterior.
    """
    doc = fitz.open(pdf_path)
    chunks: list[NormaChunk] = []

    current_secao = "0"
    current_titulo = "Geral"
    pending_table_title: Optional[str] = None
    buffer: list[str] = []
    buffer_pagina: int = 1

    def flush_buffer(tipo="texto"):
        nonlocal buffer, buffer_pagina
        if not buffer:
            return
        texto = " ".join(buffer).strip()
        if len(texto) < 30:  # descarta ruído
            buffer = []
            return
        tags = list(set(re.findall(_RE_NORMA_REF, texto)))
        nps_matches = re.findall(_RE_NPS, texto)
        if nps_matches:
            tags += [f"NPS_{m.replace(' ', '/')}" for m in nps_matches]
        chunks.append(NormaChunk(
            norma_id=norma_id,
            secao=current_secao,
            titulo_secao=current_titulo,
            tipo=tipo,
            texto=texto,
            pagina=buffer_pagina,
            tags=list(set(tags)),
        ))
        buffer = []

    for page in doc:
        blocks = page.get_text("blocks", sort=True)  # sorted by Y
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            # Filtra cabeçalho/rodapé por posição Y
            if y0 < Y_HEADER_LIMIT or y1 > Y_FOOTER_LIMIT:
                continue
            if block_type != 0:  # 0=texto, 1=imagem
                flush_buffer()
                continue

            text = text.strip()
            if not text:
                continue

            first_line = text.split('\n')[0].strip()

            # Detecta nova seção numerada
            m_sec = _RE_SECAO.match(first_line)
            if m_sec and len(m_sec.group(1)) <= 7:
                flush_buffer()
                current_secao = m_sec.group(1)
                current_titulo = m_sec.group(2).strip()
                # A própria linha da seção vai para o buffer
                buffer.append(text)
                buffer_pagina = page.number + 1
                continue

            # Detecta título de tabela — flush do texto anterior, inicia buffer de tabela
            if _RE_TABELA.match(first_line):
                flush_buffer()
                pending_table_title = first_line
                buffer.append(text)
                buffer_pagina = page.number + 1
                continue

            # Detecta fim de bloco tabela (próxima seção ou próxima tabela)
            if pending_table_title and (_RE_SECAO.match(first_line) or _RE_TABELA.match(first_line)):
                flush_buffer(tipo="tabela")
                pending_table_title = None

            # Detecta figura
            if _RE_FIGURA.match(first_line):
                flush_buffer()
                chunks.append(NormaChunk(
                    norma_id=norma_id,
                    secao=current_secao,
                    titulo_secao=current_titulo,
                    tipo="figura",
                    texto=first_line,  # apenas o caption
                    pagina=page.number + 1,
                    tags=[norma_id],
                ))
                continue

            # Nota: mantém vinculada à seção corrente mas com tipo dedicado
            if _RE_NOTA.match(first_line):
                flush_buffer()
                buffer.append(text)
                buffer_pagina = page.number + 1
                flush_buffer(tipo="nota")
                continue

            buffer.append(text)
            if not buffer_pagina:
                buffer_pagina = page.number + 1

        flush_buffer(tipo="tabela" if pending_table_title else "texto")
        pending_table_title = None

    doc.close()
    return chunks