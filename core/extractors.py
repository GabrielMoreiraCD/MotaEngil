"""
extractors.py — Extrator Espacial de PDFs de Normas Técnicas Petrobras
==========================================================================
Este módulo implementa o `StandardPDFExtractor`, um extrator avançado projetado para processar normas técnicas da Petrobras. Ele combina técnicas de extração de texto, detecção de tabelas
"""

import os
import re
import fitz

fitz.TOOLS.mupdf_display_errors(False)

MIN_IMAGE_PX: int = 80
TEXT_MARGIN: float = 65.0
RENDER_SCALE: float = 2.0

_SECTION_TITLE_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+){0,4}\s+[A-ZÁÉÍÓÚÂÊÎÔÛÃÕ][A-ZÁÉÍÓÚÂÊÎÔÛÃÕa-záéíóúâêîôûãõ]"
    r"|Anexo\s+[A-Z]\b"
    r"|Figura\s+[A-Z]\.\d+"
    r")"
)

_FIGURE_CAPTION_RE = re.compile(
    r"Figura\s+[A-Z]\.\d+\s*[-–]",
    re.IGNORECASE,
)


class StandardPDFExtractor:
    def __init__(self, file_path: str, image_output_dir: str):
        self.file_path = file_path
        self.doc_id = os.path.basename(file_path).replace(".pdf", "")
        self.image_output_dir = image_output_dir
        os.makedirs(self.image_output_dir, exist_ok=True)
        self.document = fitz.open(file_path)

        # Ajuste cirúrgico das margens baseado na geometria real (y_header_end=74, y_footer_start=809)
        self.y_min = 76.0
        self.y_max = 805.0

    def extract_all(self) -> list[dict]:
        extracted_pages = []

        for page_num, page in enumerate(self.document, start=1):
            page_data = {
                "page_number": page_num,
                "blocks": [],
                "images_metadata": [],
            }

            # 1. Extração de imagens raster
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]

                rects = page.get_image_rects(xref)
                if not rects:
                    continue

                bbox = tuple(rects[0])
                x0, y0, x1, y1 = bbox

                # Filtra apenas se a imagem estiver totalmente inserida na margem
                if y1 <= self.y_min or y0 >= self.y_max:
                    continue

                try:
                    base_image = self.document.extract_image(xref)
                except Exception:
                    continue

                img_w = base_image["width"]
                img_h = base_image["height"]
                img_ext = base_image["ext"]

                if img_w < MIN_IMAGE_PX or img_h < MIN_IMAGE_PX:
                    continue

                image_filename = f"{self.doc_id}_p{page_num}_img{img_index}.{img_ext}"
                image_filepath = os.path.join(self.image_output_dir, image_filename)
                with open(image_filepath, "wb") as f:
                    f.write(base_image["image"])

                surrounding_text = self._extract_surrounding_text(page, bbox)

                page_data["images_metadata"].append({
                    "image_path":       image_filepath,
                    "bbox":             bbox,
                    "image_width":      img_w,
                    "image_height":     img_h,
                    "surrounding_text": surrounding_text,
                    "image_type":       "raster",
                    "processed":        False,
                })

            # 2. Extração de blocos de texto (Filtragem em nível de linha)
            text_blocks = page.get_text("dict")["blocks"]
            current_section_title = ""

            for b in text_blocks:
                if b.get("type") != 0:
                    continue

                valid_lines_text = []
                valid_bboxes = []

                # Itera sobre linhas para não descartar um bloco massivo inteiro por causa do cabeçalho
                for line in b.get("lines", []):
                    lx0, ly0, lx1, ly1 = line["bbox"]
                    
                    if ly1 <= self.y_min or ly0 >= self.y_max:
                        continue
                        
                    span_text = " ".join(span["text"] for span in line.get("spans", [])).strip()
                    if span_text:
                        valid_lines_text.append(span_text)
                        valid_bboxes.append((lx0, ly0, lx1, ly1))

                if not valid_lines_text:
                    continue

                text_content = " ".join(valid_lines_text).strip()
                if len(text_content) <= 15:
                    continue

                nx0 = min(bx[0] for bx in valid_bboxes)
                ny0 = min(bx[1] for bx in valid_bboxes)
                nx1 = max(bx[2] for bx in valid_bboxes)
                ny1 = max(bx[3] for bx in valid_bboxes)

                first_line = text_content.split("\n")[0].strip()
                if _SECTION_TITLE_RE.match(text_content) and len(first_line) <= 80:
                    current_section_title = first_line

                page_data["blocks"].append({
                    "type":          "texto_extraido",
                    "content":       text_content,
                    "bbox":          (nx0, ny0, nx1, ny1),
                    "section_title": current_section_title,
                })

            # 3. Extração de tabelas
            try:
                tabs = page.find_tables(strategy="lines_strict")
            except Exception:
                tabs = []

            if tabs:
                for tab in tabs:
                    x0, y0, x1, y1 = tab.bbox
                    # Filtra apenas se estiver totalmente inserida na margem
                    if y1 <= self.y_min or y0 >= self.y_max:
                        continue
                    try:
                        df = tab.to_pandas()
                        markdown_table = df.to_markdown(index=False)
                    except Exception:
                        continue
                    if not markdown_table or len(markdown_table.strip()) < 10:
                        continue
                    page_data["blocks"].append({
                        "type":          "tabela_extraida",
                        "content":       markdown_table,
                        "bbox":          tuple(tab.bbox),
                        "section_title": current_section_title,
                    })

            page_data["blocks"] = sorted(
                page_data["blocks"], key=lambda b: b["bbox"][1]
            )

            # 4. Detecção e renderização de figuras vetoriais
            page_text_full = " ".join(b["content"] for b in page_data["blocks"])
            vector_figure_match = _FIGURE_CAPTION_RE.search(page_text_full)

            if vector_figure_match and not page_data["images_metadata"]:
                vector_meta = self._render_vector_figure_page(page, page_num)
                if vector_meta:
                    page_data["images_metadata"].append(vector_meta)

            extracted_pages.append(page_data)

        self.document.close()
        return extracted_pages

    def _extract_surrounding_text(self, page, img_bbox: tuple) -> str:
        x0, y0, x1, y1 = img_bbox

        rect_above = fitz.Rect(0, max(self.y_min, y0 - TEXT_MARGIN), page.rect.width, y0)
        rect_below = fitz.Rect(0, y1, page.rect.width, min(self.y_max, y1 + TEXT_MARGIN))

        text_above = page.get_text("text", clip=rect_above).strip()
        text_below = page.get_text("text", clip=rect_below).strip()

        parts = [p for p in [text_above, text_below] if p]
        return " ".join(" ".join(parts).split())

    def _render_vector_figure_page(self, page, page_num: int) -> dict | None:
        try:
            clip = fitz.Rect(0, self.y_min, page.rect.width, self.y_max)

            matrix = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
            pixmap = page.get_pixmap(matrix=matrix, clip=clip, colorspace=fitz.csRGB)

            img_filename = f"{self.doc_id}_p{page_num}_vec.png"
            img_filepath = os.path.join(self.image_output_dir, img_filename)
            pixmap.save(img_filepath)

            img_w = pixmap.width
            img_h = pixmap.height

            surrounding = page.get_text("text", clip=clip).strip()
            surrounding = " ".join(surrounding.split())

            return {
                "image_path":       img_filepath,
                "bbox":             tuple(clip),
                "image_width":      img_w,
                "image_height":     img_h,
                "surrounding_text": surrounding[:600],
                "image_type":       "vector_render",
                "processed":        False,
            }

        except Exception as e:
            return None