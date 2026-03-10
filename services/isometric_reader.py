"""
isometric_reader.py — Pass 5: Leitura Visual de Isométricos via Qwen2.5-VL
===========================================================================
Usa o modelo Qwen/Qwen2.5-VL-72B-Instruct via HuggingFace Inference API
(InferenceClient) para extrair especificações de materiais de imagens de
isométricos de tubulação (JPG/PNG).

Entradas:
  - Arquivo de imagem (JPG/PNG) contendo o isométrico
  - Contexto do serviço (tag da linha, piping class)

Saídas:
  - Lista de IsometricExtractedSpec com tipo_material, descrição, quantidade, etc.

Uso:
    from services.isometric_reader import IsometricReader
    reader = IsometricReader(hf_token="hf_...")
    specs = reader.extract_specs("Figura 1 atualizada.jpg", context={
        "tag_linha": "2\"-F-B10S-200",
        "piping_class": "B10S",
    })
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Optional

from core.schemas import IsometricExtractedSpec

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt para extração de materiais de isométrico
# ─────────────────────────────────────────────────────────────────────────────

ISOMETRIC_SYSTEM_PROMPT = """\
Você é um engenheiro de tubulação sênior Petrobras especializado em leitura de isométricos.
Ao analisar a imagem, identifique APENAS materiais físicos que precisam ser comprados:
tubos, flanges, conexões (cotovelos, tees, reduções), juntas, parafusos, válvulas.
NÃO liste dimensões, cotas, setas de direção, títulos ou elementos de desenho não-material.
Retorne APENAS JSON válido — sem texto antes ou depois."""

ISOMETRIC_USER_PROMPT = """\
Analise este isométrico de tubulação Petrobras e extraia a lista de materiais.

Contexto:
- Tag da linha: {tag_linha}
- Piping Class: {piping_class}
- Plataforma: P-54

Retorne um array JSON com os materiais identificados visualmente:
[
  {{
    "tipo_material": "tubo_conducao",
    "descricao_tecnica": "Tubo aço carbono API 5L GrB, 2\\", SCH 40",
    "quantidade": 2.08,
    "unidade": "M",
    "diametro_nps": "2",
    "schedule": "SCH 40",
    "notas": "trecho entre flange e cotovelo"
  }},
  {{
    "tipo_material": "flange_pescoço",
    "descricao_tecnica": "Flange pescoço de solda ASTM A105, 2\\", 150#",
    "quantidade": 2,
    "unidade": "UN",
    "diametro_nps": "2",
    "schedule": null,
    "notas": "flanges de extremidade do spool"
  }}
]

Tipos válidos:
- tubo_conducao: tubos retos
- flange_pescoço: flange weld neck
- flange_encaixe: flange socket weld
- junta_espiralada: gaxetas espirometálicas
- parafuso_estojo: studs ASTM A193 B7
- cotovelo: cotovelos 90° ou 45°
- tee: tês retos ou redução
- reducao: redução concêntrica ou excêntrica
- valvula: qualquer tipo de válvula
- eletrodo_smaw: eletrodo revestido
- consumivel_end: kit líquido penetrante

Se não conseguir identificar um material claramente, não inclua no array.
Retorne [] se o isométrico não mostrar materiais identificáveis."""


def _extract_json_array(text: str) -> list:
    """Extrai array JSON de texto que pode conter markdown ou texto adicional."""
    text = text.strip()
    # Remove blocos de código markdown
    text = re.sub(r'```(?:json)?', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'```', '', text).strip()

    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        log.warning("Nenhum array JSON encontrado na resposta do modelo de visão.")
        return []

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate_fixed = candidate.replace('\\"', '"').replace('\\n', ' ')
        try:
            return json.loads(candidate_fixed)
        except json.JSONDecodeError as e:
            log.warning(f"JSON inválido na resposta do isométrico: {e}")
            return []


class IsometricReader:
    """
    Leitor de isométricos de tubulação via modelo de visão Qwen2.5-VL-72B.
    Usa HuggingFace Inference API (serverless) — requer HF_TOKEN.
    """

    MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

    def __init__(self, hf_token: str, model: Optional[str] = None):
        self.hf_token = hf_token
        self.model_id = model or self.MODEL
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(token=self.hf_token)
                log.info(f"IsometricReader: InferenceClient inicializado (modelo: {self.model_id})")
            except ImportError:
                raise ImportError(
                    "huggingface_hub não instalado. Execute: pip install huggingface-hub>=0.24.0"
                )
        return self._client

    def _image_to_base64(self, image_path: str) -> tuple[str, str]:
        """Converte imagem para base64 e retorna (b64_string, mime_type)."""
        p = Path(image_path)
        ext = p.suffix.lower().lstrip(".")
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "gif": "image/gif", "webp": "image/webp"}
        mime = mime_map.get(ext, "image/jpeg")
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64, mime

    def _parse_response(self, raw: str, image_path: str) -> list[IsometricExtractedSpec]:
        """Parseia resposta do modelo e retorna lista de IsometricExtractedSpec."""
        items = _extract_json_array(raw)
        specs = []
        fonte = Path(image_path).name

        for item in items:
            if not isinstance(item, dict):
                continue
            tipo = (item.get("tipo_material") or "").strip()
            desc = (item.get("descricao_tecnica") or "").strip()
            if not tipo or not desc:
                continue

            try:
                qtd = item.get("quantidade")
                qtd_float = float(qtd) if qtd is not None else None
            except (ValueError, TypeError):
                qtd_float = None

            specs.append(IsometricExtractedSpec(
                tipo_material=tipo,
                descricao_tecnica=desc,
                quantidade=qtd_float,
                unidade=(item.get("unidade") or "UN").upper(),
                diametro_nps=item.get("diametro_nps"),
                schedule=item.get("schedule"),
                fonte_imagem=fonte,
                confianca=0.82,
                notas=item.get("notas"),
            ))

        log.info(f"  [Isométrico] {len(specs)} specs extraídas de '{fonte}'")
        return specs

    def extract_specs(
        self,
        image_path: str,
        context: Optional[dict] = None,
    ) -> list[IsometricExtractedSpec]:
        """
        Envia imagem ao Qwen2.5-VL e extrai specs de material.

        Args:
            image_path: Caminho para o arquivo de imagem (JPG/PNG)
            context: Dict com 'tag_linha', 'piping_class', etc.

        Returns:
            Lista de IsometricExtractedSpec
        """
        p = Path(image_path)
        if not p.exists():
            log.warning(f"Isométrico não encontrado: {image_path}")
            return []

        ctx = context or {}
        tag = ctx.get("tag_linha", "não identificada")
        piping = ctx.get("piping_class", "não identificada")

        log.info(f"[IsometricReader] Analisando: {p.name} | Tag: {tag} | Piping: {piping}")

        try:
            b64, mime = self._image_to_base64(image_path)
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": ISOMETRIC_USER_PROMPT.format(
                        tag_linha=tag,
                        piping_class=piping,
                    ),
                },
            ]

            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": ISOMETRIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=2048,
                temperature=0.1,
            )
            raw_text = response.choices[0].message.content
            log.debug(f"  [Qwen2.5-VL] Resposta ({len(raw_text)} chars): {raw_text[:200]}...")
            return self._parse_response(raw_text, image_path)

        except Exception as e:
            log.error(f"Erro na chamada ao modelo de visão para '{p.name}': {e}", exc_info=True)
            return []

    def extract_specs_batch(
        self,
        image_paths: list[str],
        context: Optional[dict] = None,
    ) -> list[IsometricExtractedSpec]:
        """Processa múltiplos isométricos e combina as specs."""
        all_specs = []
        for path in image_paths:
            specs = self.extract_specs(path, context)
            all_specs.extend(specs)
        log.info(f"[IsometricReader] Total: {len(all_specs)} specs de {len(image_paths)} imagem(ns)")
        return all_specs
