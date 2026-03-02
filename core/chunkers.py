"""
chunkers.py — Chunker Espacial e Semântico para Normas Técnicas Petrobras
==========================================================================
Este módulo implementa o `SpatialChunker`, um chunker avançado projetado para processar normas técnicas da Petrobras. Ele combina técnicas de chunking baseadas em tamanho, detecção de cabeçalhos de seção e vinculação espacial de imagens para criar chunks ricos em contexto e semanticamente coerentes.
"""

import math
import re

_SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+){0,4}\s+[A-ZÁÉÍÓÚÂÊÎÔÛÃÕ][A-ZÁÉÍÓÚÂÊÎÔÛÃÕa-záéíóúâêîôûãõ]"
    r"|Anexo\s+[A-Z]\b"
    r"|Figura\s+[A-Z]\.\d+"
    r")"
)

_TYPE_MAP = {
    "texto_extraido": "text",
    "tabela_extraida": "table",
}


class SpatialChunker:
    def __init__(
        self,
        target_chunk_size: int = 1200,
        split_on_section_headers: bool = True,
    ):
        self.target_chunk_size = target_chunk_size
        self.split_on_section_headers = split_on_section_headers

    # --------------------------------------------------------------------------
    # Geometria
    # --------------------------------------------------------------------------

    def _calculate_center(self, bbox) -> tuple:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _calculate_distance(self, c1, c2) -> float:
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _merge_bboxes(self, bbox1, bbox2) -> list:
        return [
            min(bbox1[0], bbox2[0]),
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3]),
        ]

    # --------------------------------------------------------------------------
    # Split em seções
    # --------------------------------------------------------------------------

    def _is_section_header(self, text: str) -> bool:
        first_line = text.strip().split("\n")[0].strip()
        return bool(_SECTION_HEADER_RE.match(first_line)) and len(first_line) <= 80

    def _should_split_before(self, next_block: dict, current_size: int) -> bool:
        if current_size == 0:
            return False
        if self.split_on_section_headers:
            if self._is_section_header(next_block.get("content", "")):
                return True
        if current_size >= self.target_chunk_size:
            return True
        return False

    # --------------------------------------------------------------------------
    # Pipeline principal
    # --------------------------------------------------------------------------

    def link_and_chunk(self, extracted_pages: list, norma_id: str) -> list:
        chunks = []

        for page in extracted_pages:
            page_num = page["page_number"]

            # ------------------------------------------------------------------
            # Passo 1 — Block Merging com split em seções
            # ------------------------------------------------------------------
            merged_blocks = []
            current_text    = ""
            current_bbox    = None
            anchor_section  = ""

            def _flush():
                nonlocal current_text, current_bbox, anchor_section
                if current_text:
                    merged_blocks.append({
                        "type":          "texto_extraido",
                        "content":       current_text.strip(),
                        "bbox":          tuple(current_bbox),
                        "section_title": anchor_section,
                    })
                    current_text   = ""
                    current_bbox   = None
                    anchor_section = ""

            for block in page["blocks"]:
                content = block.get("content", "").strip()

                if block["type"] == "tabela_extraida":
                    _flush()
                    merged_blocks.append({
                        "type":          "tabela_extraida",
                        "content":       content,
                        "bbox":          tuple(block["bbox"]),
                        "section_title": block.get("section_title", ""),
                    })
                    continue

                if self._should_split_before(block, len(current_text)):
                    _flush()

                if not current_text:
                    current_text   = content
                    current_bbox   = list(block["bbox"])
                    anchor_section = block.get("section_title", "")
                else:
                    current_text += "\n\n" + content
                    current_bbox  = self._merge_bboxes(current_bbox, block["bbox"])

            _flush()

            # ------------------------------------------------------------------
            # Passo 2 — Vinculação Espacial de Imagens
            # ------------------------------------------------------------------
            for img_meta in page["images_metadata"]:
                img_center    = self._calculate_center(img_meta["bbox"])
                closest_block = None
                min_distance  = float("inf")

                for m_block in merged_blocks:
                    dist = self._calculate_distance(
                        img_center, self._calculate_center(m_block["bbox"])
                    )
                    if dist < min_distance:
                        min_distance  = dist
                        closest_block = m_block

                if closest_block is not None:
                    if "associated_images" not in closest_block:
                        closest_block["associated_images"] = []
                    closest_block["associated_images"].append(img_meta)

            # ------------------------------------------------------------------
            # Passo 3 — Formatação Final dos Chunks
            # ------------------------------------------------------------------
            for m_block in merged_blocks:
                chunk_type = _TYPE_MAP.get(m_block["type"], "text")

                chunks.append({
                    "content": m_block["content"],
                    "metadata": {
                        "chunk_type":    chunk_type,
                        "page_number":   page_num,
                        "norma_id":      norma_id,
                        "section_title": m_block.get("section_title", ""),
                        "bounding_box":  m_block.get("bbox"),
                    },
                })

                # Chunk independente para cada imagem associada
                for img in m_block.get("associated_images", []):
                    surrounding = img.get("surrounding_text", "")
                    img_type    = img.get("image_type", "raster")  # [FIX 1]

                    # Prefixo descritivo diferenciado para VLM
                    if img_type == "vector_render":
                        prefix = "[FIGURA TÉCNICA VETORIAL — RENDER DE PÁGINA]"
                    else:
                        prefix = "[IMAGEM TÉCNICA RASTER]"

                    img_content = (
                        f"{prefix} {norma_id} — Página {page_num}. "
                        f"Seção: {m_block.get('section_title', 'não identificada')}. "
                        f"Contexto: {surrounding[:400] if surrounding else 'não capturado'}."
                    )

                    chunks.append({
                        "content": img_content,
                        "metadata": {
                            "chunk_type":       "image_ref",
                            "page_number":      page_num,
                            "norma_id":         norma_id,
                            "section_title":    m_block.get("section_title", ""),
                            "bounding_box":     img.get("bbox"),
                            "image_path":       img.get("image_path"),
                            "image_width":      img.get("image_width"),
                            "image_height":     img.get("image_height"),
                            "surrounding_text": surrounding,
                            "image_type":       img_type,  # [FIX 1]
                        },
                    })

        return chunks
