"""
vision_engine.py - Motor de Visão Computacional para Extração de Informações de Imagens em Documentos Técnicos
============================================================================================================
Este módulo implementa o CloudVisionEngine, que utiliza a API de inferência da Hugging Face para processar imagens extraídas de documentos técnicos (PDFs) 
e gerar descrições textuais detalhadas. O motor é projetado para lidar com artefatos gráficos complexos, como diagramas e tabelas, 
comuns em normas técnicas, e inclui pré-processamento específico para remover logos e filtrar imagens irrelevantes.
! este modulo esta em teste e deve ser substituido pelo modelo Qwen/Qwen2.5-VL-72B-Instruct com prompt de descrição de imagem, para evitar a necessidade de pré-processamento complexo e garantir uma extração mais robusta e precisa.
"""
import os
import fitz 
import base64
import io
import logging
import time

from pathlib import Path
from typing import List
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ---------------------------------------------------------
# CONFIGURAÇÕES GERAIS E AMBIENTE
# ---------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Variável de ambiente HF_TOKEN não encontrada. Verifique seu arquivo .env")

PDF_PATH = r"Data\normas_locais\N-2918.pdf"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT_PATH = "prompts/describe_image.MD"
HF_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct:hyperbolic"

MIN_IMAGE_SIZE = 350  # Filtra artefatos e logos pequenos na extração

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# VISION ENGINE (HUGGING FACE)
# ---------------------------------------------------------
class CloudVisionEngine:
    def __init__(
        self,
        api_key: str,
        model: str,
        prompt_path: str,
        max_retries: int = 3,
        upscale_min_size: int = 512,
    ):
        self.client = InferenceClient(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.upscale_min_size = upscale_min_size

        try:
            self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo de prompt não encontrado no caminho: {prompt_path}")

    # -----------------------------------------------------
    def _remove_petrobras_logo(self, image: Image.Image) -> Image.Image:
        """
        Remove região superior onde fica a logo da Petrobras.
        """
        w, h = image.size
        crop_top = int(h * 0.12)
        return image.crop((0, crop_top, w, h))

    # -----------------------------------------------------
    def _preprocess(self, image: Image.Image) -> Image.Image:
        """
        Aplica os filtros geométricos e redimensionamento necessário.
        """
        image = self._remove_petrobras_logo(image)
        w, h = image.size

        if w < self.upscale_min_size or h < self.upscale_min_size:
            scale = max(
                self.upscale_min_size / w,
                self.upscale_min_size / h
            )
            image = image.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS
            )
        return image

    # -----------------------------------------------------
    def _to_base64_data_url(self, image: Image.Image) -> str:
        """
        Converte a imagem diretamente para o formato esperado pelo modelo Qwen.
        """
        buffer = io.BytesIO()
        # O modelo Qwen processa bem o formato PNG, mantendo definição de linhas técnicas
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"

    # -----------------------------------------------------
    def _call(self, image_data_url: str) -> str:
        """
        Executa a inferência via Hugging Face InferenceClient com tratativas de erro.
        """
        for i in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.prompt_template
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url
                                    }
                                }
                            ]
                        }
                    ],
                )
                
                content = completion.choices[0].message.content
                
                if content and len(content) > 50:
                    return content
                else:
                    logger.warning(f"Resposta curta ou vazia recebida na tentativa {i+1}.")

            except Exception as e:
                logger.warning(f"Erro na requisição (Tentativa {i+1}/{self.max_retries}): {e}")
                time.sleep(2)

        return "Falha na extração após múltiplas tentativas."

    # -----------------------------------------------------
    def describe(self, image: Image.Image) -> str:
        image = self._preprocess(image)
        data_url = self._to_base64_data_url(image)
        return self._call(data_url)


# ---------------------------------------------------------
# PDF IMAGE EXTRACTION
# ---------------------------------------------------------
def extract_images(pdf_path: str) -> List[Image.Image]:
    """
    Varre o PDF extraindo blocos de imagem nativos utilizando PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        img_list = page.get_images(full=True)

        if img_list:
            logger.info(f"Página {page_num+1}: {len(img_list)} imagens encontradas.")

        for img in img_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Filtro de dimensão mínima
            if image.width < MIN_IMAGE_SIZE or image.height < MIN_IMAGE_SIZE:
                continue

            images.append(image)

    return images


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    logger.info("Iniciando pipeline de extração de imagens do PDF...")
    images = extract_images(PDF_PATH)
    logger.info(f"Total de imagens válidas extraídas: {len(images)}")

    engine = CloudVisionEngine(
        api_key=HF_TOKEN,
        model=HF_MODEL,
        prompt_path=PROMPT_PATH,
    )

    output_file = OUTPUT_DIR / "image_descriptions.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, image in enumerate(images):
            logger.info(f"Iniciando inferência da imagem {i+1}/{len(images)}...")
            
            desc = engine.describe(image)

            if desc:
                f.write(f"\n\n--- IMAGEM {i+1} ---\n\n")
                f.write(desc)
                f.write("\n\n")
                
            # Rate limit preventivo para evitar bloqueios na API pública da Hugging Face
            time.sleep(1)

    logger.info(f"Processamento finalizado. Arquivo salvo em: {output_file}")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()