"""
normas_agent.py — Etapa 3: Consulta de Normas Técnicas Petrobras
=================================================================================
Dado um EscopoTriagemUnificado (saída da Etapa 2), consulta cada norma aplicável
no Qdrant (coleção normas_tecnicas_publicas_v2) e extrai especificações de materiais
e consumíveis usando Claude (padrão) ou Llama3 (fallback offline).

Fluxo por norma:
  1. Gera 3 queries direcionadas com base no tipo de serviço e piping class
  2. Recupera chunks via query_with_norma_context() com filtro norma_id + tipo
  3. Envia chunks ao LLM com prompt de extração estruturada (skeleton JSON)
  4. Valida output com Pydantic → EspecificacaoMaterial

Saída: NormasConsultaResult (persistida em etapa3_normas_result.json)
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from core.config import config
from core.schemas import (
    EscopoTriagemUnificado,
    EspecificacaoMaterial,
    NormasConsultaResult,
)
from services.rag_engine import query_with_norma_context

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Regex para extrair piping class da tag da linha (ex: "2\"-F-B10S-200" → "B10S")
# Reusa lógica de etapa1.py
# ─────────────────────────────────────────────────────────────────────────────
RE_PIPING_CLASS = re.compile(
    r'\b([A-Z][0-9]{1,2}[A-Z]{0,2}(?:-[A-Z0-9]+)?)\b'
)
RE_NPS = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:"|\'\'|polegadas?|in\b)', re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# Skeleton JSON para o LLM (mesmo padrão de triage_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
SKELETON_SPECS = """[
  {
    "tipo_material": "???",
    "descricao_tecnica": "???",
    "norma_origem": "???",
    "secao_norma": "???",
    "especificacao_aws": "??? ou null",
    "especificacao_astm": "??? ou null",
    "especificacao_asme": "??? ou null",
    "diametro_nps": "??? ou null",
    "schedule": "??? ou null",
    "pressao_classe": "??? ou null",
    "unidade": "??? ou null",
    "observacoes": "??? ou null",
    "confianca": 0.9
  }
]"""

SYSTEM_PROMPT_CLAUDE = """\
Você é engenheiro de materiais especialista em normas técnicas Petrobras.
Sua tarefa é extrair EXCLUSIVAMENTE os materiais e consumíveis EXPLICITAMENTE \
especificados nos trechos de norma fornecidos, para o serviço descrito.

REGRAS RÍGIDAS:
1. Cite apenas materiais que aparecem EXPLICITAMENTE no texto das normas fornecidas.
2. Cada item DEVE ter norma_origem e secao_norma preenchidos.
3. Para soldagem: identifique classificação AWS, processo (SMAW/GTAW/FCAW/GMAW).
4. Para END: identifique método (LP=líquido penetrante, UT=ultrassom) e consumíveis.
5. NÃO invente especificações ausentes do texto. Use null para campos não encontrados.
6. Retorne APENAS JSON válido (lista de objetos). Nenhum texto adicional."""

PROMPT_TEMPLATE_CLAUDE = """\
ESCOPO DO SERVIÇO:
- Serviço: {titulo_servico}
- TAG da linha: {tag_linha_principal}
- Piping class: {piping_class}
- NPS (polegadas): {nps}
- Tarefas: {tarefas}

TRECHOS DA NORMA [{norma_id}]:
{chunks_text}

Extraia os materiais/consumíveis dos trechos acima e retorne no formato JSON abaixo.
Preencha "???" com os valores encontrados. Use null para campos não presentes no texto.

{skeleton}"""

# Prompt simplificado para Llama3 (menos campos, contexto menor)
PROMPT_TEMPLATE_LLAMA = """\
<s>[INST] Você é engenheiro especialista em normas Petrobras.
Extraia materiais dos trechos de norma abaixo para o serviço indicado.
Retorne APENAS JSON válido. Sem texto adicional.

Serviço: {titulo_servico} | Linha: {tag_linha_principal} | Norma: {norma_id}

TRECHOS:
{chunks_text}

JSON esperado (lista):
[{{"tipo_material":"...","descricao_tecnica":"...","norma_origem":"{norma_id}",
"secao_norma":"...","especificacao_aws":null,"especificacao_astm":null,
"unidade":null,"confianca":0.8}}]
[/INST]"""


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _extract_piping_class(tag: str) -> str:
    """Extrai piping class da tag da linha. Ex: '2\"-F-B10S-200' → 'B10S'."""
    matches = RE_PIPING_CLASS.findall(tag.upper())
    # Filtra matches que parecem ser piping class (letra + número + letras opcionais)
    candidates = [m for m in matches if re.match(r'^[A-Z]\d{1,2}[A-Z]{0,2}$', m)]
    return candidates[0] if candidates else ""


def _extract_nps(tag: str) -> str:
    """Extrai NPS (polegadas nominais) da tag. Ex: '2\"-F-B10S-200' → '2'."""
    m = RE_NPS.search(tag)
    return m.group(1) if m else ""


def _extract_json_robust(text: str) -> list:
    """
    Extrai lista JSON do texto do LLM com tratamento de erros.
    Reutiliza a estratégia de bracket balancing do triage_agent.py.
    """
    text = text.strip()
    # Encontra o primeiro '[' e fecha no último ']'
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        log.warning("Nenhum array JSON encontrado na resposta do LLM.")
        return []
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Tenta corrigir aspas escapadas e retry
        candidate_fixed = candidate.replace('\\"', '"').replace('\\n', ' ')
        try:
            return json.loads(candidate_fixed)
        except json.JSONDecodeError as e:
            log.warning(f"JSON inválido após tentativa de correção: {e}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Mapa de normas Petrobras por tipo de serviço (fallback quando escopo sem N-XXX)
# ─────────────────────────────────────────────────────────────────────────────
_NORMAS_POR_TIPO: list[tuple[list[str], list[str]]] = [
    # (keywords no título/tarefas, normas inferidas)
    (["solda", "soldagem", "weld", "reparo com solda"],           ["N-1852", "N-0133"]),
    (["pintura", "revestimento", "paint"],                         ["N-2879", "N-1272"]),
    (["end", "ensaio", "penetrante", "ultrassom", "radiografia"],  ["N-2008"]),
    (["tubulação", "tubo", "piping", "spool", "fabricar", "fabricação"], ["N-0116", "N-2595"]),
    (["válvula", "valvula"],                                       ["N-0116"]),
    (["flanges", "flange"],                                        ["N-0116"]),
    (["inspecao", "inspeção", "inspection"],                       ["N-2008", "N-1424"]),
    (["montagem", "instalacao", "instalar"],                       ["N-0116"]),
]


def _infer_normas_from_scope(escopo: "EscopoTriagemUnificado") -> list[str]:
    """
    Infere normas Petrobras aplicáveis com base em palavras-chave do escopo.
    Usado como fallback quando `normas_petrobras_aplicaveis` está vazio.
    """
    text = " ".join([
        escopo.titulo_servico or "",
        " ".join(escopo.tarefas_execucao or []),
        " ".join(escopo.notas_e_restricoes or []),
    ]).lower()

    inferred: list[str] = []
    for keywords, normas in _NORMAS_POR_TIPO:
        if any(kw in text for kw in keywords):
            for n in normas:
                if n not in inferred:
                    inferred.append(n)

    return inferred


def _chunks_to_text(chunks: list[dict], max_chars: int = 12000) -> str:
    """Converte lista de chunks para texto concatenado com delimitadores."""
    parts = []
    total = 0
    for c in chunks:
        secao = c.get("secao") or "?"
        texto = c.get("texto", "").strip()
        if not texto:
            continue
        block = f"[Seção {secao}]\n{texto}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# NormasConsultationAgent
# ─────────────────────────────────────────────────────────────────────────────

class NormasConsultationAgent:
    """
    Consulta normas técnicas Petrobras no Qdrant e extrai especificações
    de materiais usando LLM (Claude ou Llama3 fallback).
    """

    def __init__(self, use_claude: bool = True):
        self.use_claude = use_claude and config.USE_CLAUDE and bool(config.ANTHROPIC_API_KEY)

        # Embedder para queries no Qdrant
        log.info("NormasConsultationAgent: carregando embedder...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        # Qdrant client
        self.qdrant = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

        # LLM client
        if self.use_claude:
            from core.models import get_claude_client
            self.claude = get_claude_client()
            self.llm = None
            log.info(f"NormasConsultationAgent: usando Claude ({config.CLAUDE_MODEL})")
        else:
            from core.models import get_llm
            self.llm = get_llm()
            self.claude = None
            log.info("NormasConsultationAgent: usando Llama3 (fallback offline)")

    # ── Query generation ──────────────────────────────────────────────────────

    def _build_queries(self, norma_id: str, escopo: EscopoTriagemUnificado) -> list[str]:
        """
        Gera 3 queries especializadas por norma, considerando o tipo de serviço
        e a piping class da linha.
        """
        piping_class = _extract_piping_class(escopo.tag_linha_principal)
        nps = _extract_nps(escopo.tag_linha_principal)
        tarefas_str = " ".join(escopo.tarefas_execucao[:3])

        # Detecta tipo de serviço pelas tarefas para especializar as queries
        is_welding = any(
            kw in (tarefas_str + escopo.titulo_servico).lower()
            for kw in ["solda", "soldagem", "weld", "preenchimento"]
        )
        is_ndt = any(
            kw in (tarefas_str + escopo.titulo_servico).lower()
            for kw in ["ensaio", "ndt", "end", "penetrante", "ultrassom", "radiografia", "inspecao", "inspeção"]
        )

        queries = []

        if is_welding:
            queries.append(
                f"materiais consumíveis soldagem eletrodo arame {norma_id} "
                f"piping class {piping_class} aço carbono"
            )
            queries.append(
                f"especificação eletrodo revestido AWS ASTM processo soldagem {norma_id}"
            )
        if is_ndt:
            queries.append(
                f"ensaio não destrutivo END líquido penetrante consumíveis reagentes {norma_id}"
            )
            queries.append(
                f"inspeção visual critérios aceitação soldagem {norma_id} {piping_class}"
            )
        if not queries:
            # Fallback genérico
            queries.append(
                f"materiais especificação técnica requisitos {norma_id} {tarefas_str}"
            )

        # Sempre adiciona query de tubo/linha para capturar specs do material base
        if nps:
            queries.append(
                f"tubulação tubo aço carbono {nps} polegadas {piping_class} "
                f"especificação ASTM {norma_id}"
            )

        return queries[:4]  # máximo 4 queries por norma

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve_chunks(
        self, norma_id: str, queries: list[str], top_k: int = 12
    ) -> list[dict]:
        """
        Recupera chunks relevantes da norma no Qdrant.
        Filtra por tipo=tabela e tipo=texto (exclui figuras).
        Deduplica por chunk_id para evitar repetição.
        """
        seen_ids: set = set()
        all_chunks: list[dict] = []

        for query in queries:
            try:
                results = query_with_norma_context(
                    qdrant_client=self.qdrant,
                    embedder=self.embedder,
                    collection=config.COLLECTION_NORMAS,
                    query=query,
                    norma_ids=[norma_id],
                    tipos=["tabela", "texto"],
                    top_k=top_k,
                )
                for chunk in results:
                    cid = chunk.get("chunk_id")
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(chunk)
            except Exception as e:
                log.warning(f"Erro ao recuperar chunks para norma={norma_id}, query='{query[:60]}...': {e}")

        # Ordena por score decrescente e limita para não estourar contexto
        all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        return all_chunks[:20]

    # ── LLM extraction ────────────────────────────────────────────────────────

    def _extract_with_claude(
        self,
        norma_id: str,
        chunks: list[dict],
        escopo: EscopoTriagemUnificado,
    ) -> list[dict]:
        """Envia chunks ao Claude e retorna lista de dicts de especificações."""
        piping_class = _extract_piping_class(escopo.tag_linha_principal)
        nps = _extract_nps(escopo.tag_linha_principal)
        chunks_text = _chunks_to_text(chunks, max_chars=15000)

        if not chunks_text.strip():
            return []

        user_message = PROMPT_TEMPLATE_CLAUDE.format(
            titulo_servico=escopo.titulo_servico,
            tag_linha_principal=escopo.tag_linha_principal,
            piping_class=piping_class or "não identificada",
            nps=nps or "não identificado",
            tarefas="; ".join(escopo.tarefas_execucao[:5]),
            norma_id=norma_id,
            chunks_text=chunks_text,
            skeleton=SKELETON_SPECS,
        )

        try:
            response = self.claude.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT_CLAUDE,
                messages=[{"role": "user", "content": user_message}],
            )
            raw_text = response.content[0].text
            return _extract_json_robust(raw_text)
        except Exception as e:
            log.error(f"Erro na chamada Claude para norma {norma_id}: {e}")
            return []

    def _extract_with_llama(
        self,
        norma_id: str,
        chunks: list[dict],
        escopo: EscopoTriagemUnificado,
    ) -> list[dict]:
        """Envia chunks ao Llama3 via Ollama e retorna lista de dicts."""
        chunks_text = _chunks_to_text(chunks, max_chars=6000)  # contexto menor
        if not chunks_text.strip():
            return []

        prompt = PROMPT_TEMPLATE_LLAMA.format(
            titulo_servico=escopo.titulo_servico,
            tag_linha_principal=escopo.tag_linha_principal,
            norma_id=norma_id,
            chunks_text=chunks_text,
        )

        try:
            raw_text = self.llm.invoke(prompt)
            return _extract_json_robust(raw_text)
        except Exception as e:
            log.error(f"Erro no Llama3 para norma {norma_id}: {e}")
            return []

    def _parse_specs(self, raw_items: list[dict], norma_id: str) -> list[EspecificacaoMaterial]:
        """Valida e converte dicts brutos para modelos Pydantic EspecificacaoMaterial."""
        specs = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            # Remove sentinels "???"
            cleaned = {
                k: (None if str(v).strip() in ("???", "null", "None", "") else v)
                for k, v in item.items()
            }
            # Garante campos obrigatórios
            if not cleaned.get("descricao_tecnica") or not cleaned.get("tipo_material"):
                continue
            if not cleaned.get("norma_origem"):
                cleaned["norma_origem"] = norma_id
            if not cleaned.get("secao_norma"):
                cleaned["secao_norma"] = "não identificada"
            try:
                spec = EspecificacaoMaterial(**{
                    k: v for k, v in cleaned.items()
                    if k in EspecificacaoMaterial.model_fields
                })
                specs.append(spec)
            except Exception as e:
                log.debug(f"Item inválido ignorado: {e} | item={item}")
        return specs

    # ── Main ──────────────────────────────────────────────────────────────────

    def process(
        self,
        escopo: EscopoTriagemUnificado,
        output_path: Optional[Path] = None,
    ) -> NormasConsultaResult:
        """
        Processa todas as normas aplicáveis do escopo e retorna NormasConsultaResult.

        Args:
            escopo: Dados estruturados do serviço (saída da Etapa 2)
            output_path: Se fornecido, persiste resultado como JSON neste caminho
        """
        log.info(f"[Etapa 3] Iniciando consulta de normas para serviço: {escopo.id_servico}")
        normas = list(escopo.normas_petrobras_aplicaveis or [])

        if not normas:
            inferred = _infer_normas_from_scope(escopo)
            if inferred:
                log.warning(
                    f"Nenhuma norma explícita no escopo. "
                    f"Inferindo por tipo de serviço: {inferred}"
                )
                normas = inferred
            else:
                log.warning("Nenhuma norma Petrobras identificada no escopo. Retornando resultado vazio.")
                return NormasConsultaResult(
                    id_servico=escopo.id_servico,
                    normas_consultadas=[],
                    especificacoes_extraidas=[],
                    normas_sem_resultado=[],
                )

        all_specs: list[EspecificacaoMaterial] = []
        normas_sem_resultado: list[str] = []
        total_chunks = 0

        for norma_id in normas:
            log.info(f"  Consultando {norma_id}...")
            queries = self._build_queries(norma_id, escopo)
            chunks = self._retrieve_chunks(norma_id, queries)

            if not chunks:
                log.warning(f"  Nenhum chunk encontrado para {norma_id} no Qdrant.")
                normas_sem_resultado.append(norma_id)
                continue

            log.info(f"  {len(chunks)} chunks recuperados para {norma_id}")
            total_chunks += len(chunks)

            if self.use_claude:
                raw_items = self._extract_with_claude(norma_id, chunks, escopo)
            else:
                raw_items = self._extract_with_llama(norma_id, chunks, escopo)

            specs = self._parse_specs(raw_items, norma_id)
            log.info(f"  {len(specs)} especificações extraídas de {norma_id}")
            all_specs.extend(specs)

        result = NormasConsultaResult(
            id_servico=escopo.id_servico,
            normas_consultadas=normas,
            especificacoes_extraidas=all_specs,
            chunks_utilizados=total_chunks,
            normas_sem_resultado=normas_sem_resultado,
        )

        log.info(
            f"[Etapa 3] Concluído: {len(all_specs)} specs extraídas de "
            f"{len(normas) - len(normas_sem_resultado)}/{len(normas)} normas"
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
            log.info(f"  Resultado salvo em: {output_path}")

        return result
