"""
normas_agent.py — Etapa 3: Consulta de Normas Técnicas Petrobras
=================================================================================
Dado um EscopoTriagemUnificado (saída da Etapa 2), consulta cada norma aplicável
no Qdrant (coleção normas_tecnicas_publicas_v2) e extrai especificações de materiais
e consumíveis usando Ollama local (qwen2.5:7b-instruct — 100% offline).

Fluxo:
  0. Injeta specs diretas do IEIS (eletrodo, NDT) e EBP (tubo/flange do spool) — bypass Qdrant
  0b. Injeta specs visuais de isométricos (Qwen2.5-VL) quando disponíveis
  1. Para cada norma: gera queries direcionadas (piping class, tipo serviço)
  2. Recupera chunks via query_with_norma_context() com filtro norma_id + tipo
  3. Envia chunks ao LLM (Ollama local) com prompt de extração estruturada (skeleton JSON)
  4. Filtra lixo (figuras, revisões de norma, atividades de planejamento)
  5. Valida output com Pydantic → EspecificacaoMaterial

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
from core.models import ollama_chat
from core.schemas import (
    EscopoTriagemUnificado,
    EspecificacaoMaterial,
    IsometricExtractedSpec,
    NormasConsultaResult,
)
from services.rag_engine import query_with_norma_context

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Regex para extrair piping class da tag da linha (ex: "2"-F-B10S-200" → "B10S")
# ─────────────────────────────────────────────────────────────────────────────
RE_PIPING_CLASS = re.compile(
    r'\b([A-Z][0-9]{1,2}[A-Z]{0,2}(?:-[A-Z0-9]+)?)\b'
)
RE_NPS = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:"|\'\'|polegadas?|in\b)', re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# Skeleton JSON para o LLM
# ─────────────────────────────────────────────────────────────────────────────
SKELETON_SPECS = """[{"tipo_material":"...","descricao_tecnica":"...","norma_origem":"...","secao_norma":"...","confianca":0.9}]"""

SYSTEM_PROMPT = """\
Você é engenheiro de materiais especialista em normas técnicas Petrobras.
Sua tarefa é identificar materiais, consumíveis e insumos NECESSÁRIOS para o serviço descrito, \
com base nos trechos de norma fornecidos.

REGRAS:
1. Identifique materiais mencionados ou REQUERIDOS pelos procedimentos descritos nos trechos.
2. Inclua: eletrodos, arames de solda, tubos, flanges, juntas, parafusos, tintas, abrasivos, \
   reagentes END, EPIs de soldagem, gases de proteção.
3. Para soldagem: identifique classificação AWS, processo (SMAW/GTAW/FCAW/GMAW).
4. Para END: identifique consumíveis (penetrante, revelador, removedor).
5. Para pintura: identifique tipo de tinta, abrasivo para jateamento, primer.
6. Se o texto menciona um procedimento que REQUER um material específico, inclua esse material.
7. Use null para campos não encontrados. NÃO invente códigos ASTM/AWS que não aparecem no texto.
8. Retorne APENAS JSON válido (lista de objetos). Nenhum texto adicional.
9. Se não houver materiais identificáveis nos trechos, retorne []."""

PROMPT_TEMPLATE = """\
SERVIÇO: {titulo_servico} | TAG: {tag_linha_principal} | Piping: {piping_class} | NPS: {nps}"
TAREFAS: {tarefas}

TRECHOS DA NORMA [{norma_id}]:
{chunks_text}

Identifique materiais e consumíveis. Retorne JSON:
{skeleton}"""


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _extract_piping_class(tag: str) -> str:
    """Extrai piping class da tag da linha. Ex: '2"-F-B10S-200' → 'B10S'."""
    matches = RE_PIPING_CLASS.findall(tag.upper())
    candidates = [m for m in matches if re.match(r'^[A-Z]\d{1,2}[A-Z]{0,2}$', m)]
    return candidates[0] if candidates else ""


def _extract_nps(tag: str) -> str:
    """Extrai NPS (polegadas nominais) da tag. Ex: '2"-F-B10S-200' → '2'."""
    m = RE_NPS.search(tag)
    return m.group(1) if m else ""


def _extract_json_robust(text: str) -> list:
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        log.warning("Nenhum array JSON encontrado na resposta do LLM.")
        return []
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate_fixed = candidate.replace('\\"', '"').replace('\\n', ' ')
        try:
            return json.loads(candidate_fixed)
        except json.JSONDecodeError as e:
            log.warning(f"JSON inválido após tentativa de correção: {e}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Mapa de normas Petrobras por tipo de serviço
# ─────────────────────────────────────────────────────────────────────────────
_NORMAS_POR_TIPO: list[tuple[list[str], list[str]]] = [
    (["solda", "soldagem", "weld", "reparo com solda", "preenchimento por solda",
      "substituir", "substituição", "trecho", "spool", "fabricação spool"],  ["N-1852", "N-0133"]),
    (["pintura", "revestimento", "paint"],                         ["N-2879", "N-1272"]),
    (["end", "ensaio", "penetrante", "ultrassom", "radiografia",
      "inspeção pós-solda", "líquido penetrante"],                 ["N-2008", "N-1424"]),
    (["flange", "flanges", "junta espiralada", "parafuso estojo",
      "flangeamento", "gasket"],                                   ["N-0116"]),
    (["válvula", "valvula"],                                       ["N-0116"]),
    (["inspecao", "inspeção", "inspection"],                       ["N-2008", "N-1424"]),
    (["montagem", "instalacao", "instalar"],                       ["N-0133"]),
]


def _infer_normas_from_scope(escopo: "EscopoTriagemUnificado") -> list[str]:
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


def _chunks_to_text(chunks: list[dict], max_chars: int = 6000) -> str:
    """Converte chunks para texto. Limite menor para modelo 7B local."""
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

_NORMAS_CRITICAS = {"N-0115", "N-0116", "N-1852"}


class NormasConsultationAgent:
    """
    Consulta normas técnicas Petrobras no Qdrant e extrai especificações
    de materiais usando Ollama local (qwen2.5:7b-instruct).
    """

    def __init__(self):
        log.info("NormasConsultationAgent: carregando embedder...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.qdrant = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
        log.info(f"NormasConsultationAgent: usando Ollama ({config.OLLAMA_MODEL})")

    # ── Query generation ──────────────────────────────────────────────────────

    def _build_queries(self, norma_id: str, escopo: EscopoTriagemUnificado) -> list[str]:
        piping_class = _extract_piping_class(escopo.tag_linha_principal)
        nps = _extract_nps(escopo.tag_linha_principal)
        tarefas_str = " ".join(escopo.tarefas_execucao[:3])

        _context_str = (tarefas_str + " " + (escopo.titulo_servico or "")).lower()
        is_welding = any(
            kw in _context_str
            for kw in ["solda", "soldagem", "weld", "preenchimento"]
        )
        is_ndt = any(
            kw in _context_str
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

        is_pipe_replacement = any(
            kw in _context_str
            for kw in ["substituir", "substituição", "trecho", "spool", "fabricar spool",
                       "fabricação spool", "trocando trecho"]
        )

        if is_pipe_replacement:
            if nps:
                queries.append(
                    f"tubo aço carbono API 5L Gr B {nps} polegadas SCH 40 {piping_class}"
                )
                queries.append(
                    f"flange pescoço aço carbono ASTM A105 classe 150 {nps} polegadas"
                )
                queries.append(
                    f"junta espiralada AISI 316 ASME B16.20 classe 150 {nps} polegadas"
                )
                queries.append(
                    f"parafuso estojo ASTM A193 B7 Zn-Ni {nps} flange classe 150"
                )
            else:
                queries.append(
                    f"tubo aço carbono API 5L Gr B SCH 40 {piping_class} especificação ASTM"
                )
                queries.append(
                    f"flange pescoço aço carbono ASTM A105 ASME B16.5 classe 150"
                )

        if not queries:
            queries.append(
                f"materiais especificação técnica requisitos {norma_id} {tarefas_str}"
            )

        if nps and not is_pipe_replacement:
            queries.append(
                f"tubulação tubo aço carbono {nps} polegadas {piping_class} "
                f"especificação ASTM {norma_id}"
            )

        return queries[:4]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve_chunks(
        self, norma_id: str, queries: list[str], top_k: int = 8
    ) -> list[dict]:
        """
        Recupera chunks relevantes da norma no Qdrant.
        Limita a 8 chunks por query (modelo 7B não aguenta 20 chunks).
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

        all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        return all_chunks[:12]  # máximo 12 chunks (modelo 7B)

    # ── LLM extraction ────────────────────────────────────────────────────────

    def _extract_with_llm(
        self,
        norma_id: str,
        chunks: list[dict],
        escopo: EscopoTriagemUnificado,
    ) -> list[dict]:
        """Envia chunks ao Ollama local e retorna lista de dicts de especificações."""
        piping_class = _extract_piping_class(escopo.tag_linha_principal or "")
        nps = _extract_nps(escopo.tag_linha_principal or "")
        chunks_text = _chunks_to_text(chunks, max_chars=6000)

        if not chunks_text.strip():
            return []

        user_message = PROMPT_TEMPLATE.format(
            titulo_servico=escopo.titulo_servico or "",
            tag_linha_principal=escopo.tag_linha_principal or "",
            piping_class=piping_class or "não identificada",
            nps=nps or "não identificado",
            tarefas="; ".join((escopo.tarefas_execucao or [])[:5]),
            norma_id=norma_id,
            chunks_text=chunks_text,
            skeleton=SKELETON_SPECS,
        )

        try:
            # DEBUG: salva prompt para inspeção
            with open(f"_debug_prompt_{norma_id}.txt", "w", encoding="utf-8") as _df:
                _df.write(f"=== SYSTEM ===\n{SYSTEM_PROMPT}\n\n=== USER ===\n{user_message}\n")
            raw_text = ollama_chat(SYSTEM_PROMPT, user_message)
            log.info(f"  [Ollama] {norma_id}: {len(raw_text)} chars retornados")
            log.info(f"  [Ollama] Primeiros 300 chars: {raw_text[:300]}")
            parsed = _extract_json_robust(raw_text)
            log.info(f"  [Ollama] {norma_id}: {len(parsed)} items parseados")
            return parsed
        except Exception as e:
            log.error(f"Erro no Ollama para norma {norma_id}: {e}")
            return []

    @staticmethod
    def _is_valid_material_spec(item: dict) -> bool:
        INVALID_TIPO_KEYWORDS = [
            "figura", "figuras", "figura a.", "figura a1", "figura a2",
            "figura a3", "figura a4", "figura a5", "figura a6", "figura a7",
            "figura a8", "figura a9",
            "rev.", "revisão", "revisao", "partes atingidas",
            "índice", "indice", "índice de revisões", "descrição da alteração",
            "revalidação", "revalidacao", "incluída", "incluida", "revisada",
            "apêndice", "apendice", "anexo",
        ]
        INVALID_ACTIVITY_KEYWORDS = [
            "plano", "programa", "procedimento", "cronograma",
            "elaborar", "deve ser elaborado", "fase de projeto",
            "condicionamento", "rearme", "sif", "sis",
            "relatório", "relatorio", "documento", "instrução",
            "manutenção preventiva", "periodicidade", "testes comprovatórios",
            "não se aplica", "nao se aplica",
        ]
        GENERIC_DESCRIPTIONS = {
            "figura relacionada à norma",
            "figuras relacionadas à norma",
            "revisão da norma",
            "revisao da norma",
            "partes atingidas",
            "descrição da alteração",
        }

        tipo = (item.get("tipo_material") or "").lower().strip()
        desc = (item.get("descricao_tecnica") or "").lower().strip()

        if any(tipo.startswith(kw) for kw in INVALID_TIPO_KEYWORDS):
            log.debug(f"  [filtro] Tipo inválido descartado: '{item.get('tipo_material')}'")
            return False

        if desc in GENERIC_DESCRIPTIONS:
            log.debug(f"  [filtro] Descrição genérica descartada: '{desc}'")
            return False

        combined = tipo + " " + desc
        if any(kw in combined for kw in INVALID_ACTIVITY_KEYWORDS):
            log.debug(f"  [filtro] Atividade/procedimento descartado: tipo='{tipo}'")
            return False

        return True

    def _parse_specs(self, raw_items: list[dict], norma_id: str) -> list[EspecificacaoMaterial]:
        specs = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            if not self._is_valid_material_spec(item):
                log.debug(f"  Spec descartada (não é material físico): {item.get('tipo_material')}")
                continue
            cleaned = {
                k: (None if str(v).strip() in ("???", "null", "None", "") else v)
                for k, v in item.items()
            }
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

        # ── INJEÇÃO 0a: Specs diretas do IEIS ──────────────────────────────────
        ieis = getattr(escopo, "especificacoes_soldagem", None)
        if ieis:
            n_ieis_before = len(all_specs)
            if ieis.metal_adicao:
                all_specs.append(EspecificacaoMaterial(
                    tipo_material="eletrodo_smaw",
                    descricao_tecnica=(
                        f"Eletrodo revestido {ieis.metal_adicao} para soldagem "
                        f"{ieis.processo_soldagem or 'SMAW'} aço carbono "
                        f"Classe {ieis.classe_tubo or 'II'}"
                    ),
                    norma_origem="IEIS",
                    secao_norma="metal_adicao",
                    especificacao_aws=ieis.classificacao_aws or ieis.metal_adicao,
                    confianca=0.95,
                    observacoes="Extraído diretamente do IEIS — alta confiança",
                ))
                log.info(f"  [IEIS direto] Eletrodo injetado: {ieis.metal_adicao}")

            if "LP" in (ieis.ndt_requerido or []):
                all_specs.append(EspecificacaoMaterial(
                    tipo_material="consumivel_end",
                    descricao_tecnica=(
                        "Kit líquido penetrante END — penetrante fluorescente Tipo II, "
                        "revelador não-aquoso, removedor/limpador"
                    ),
                    norma_origem="IEIS_NDT",
                    secao_norma="ndt_requerido",
                    confianca=0.85,
                    observacoes="NDT LP requerido no IEIS",
                ))
                log.info("  [IEIS direto] Kit LP (líquido penetrante) injetado.")

            if ieis.material_base_tubo and ieis.material_base_tubo not in ("EPS", "N/A", "???"):
                log.info(f"  [IEIS] Material base do tubo identificado: {ieis.material_base_tubo}")

            added = len(all_specs) - n_ieis_before
            log.info(f"  [IEIS direto] {added} specs injetadas (bypass Qdrant)")

        # ── INJEÇÃO 0b: Specs do spool list do EBP ──────────────────────────────
        spool_list = getattr(escopo, "spool_list", None) or []
        if spool_list:
            n_spool_before = len(all_specs)
            for spool in spool_list:
                dn = getattr(spool, "dn", None) or ""
                sch = getattr(spool, "schedule", None) or "SCH 40"
                mat = getattr(spool, "material_tubo", None) or "API 5L GrB"
                spool_id = getattr(spool, "spool_id", None)
                comprimento = getattr(spool, "comprimento_m", None)

                if sch and re.match(r'^[A-Z]\d+[A-Z]+$', sch):
                    log.debug(f"  [EBP] Schedule '{sch}' parece piping class — usando SCH 40")
                    sch = "SCH 40"

                if dn:
                    all_specs.append(EspecificacaoMaterial(
                        tipo_material="tubo_conducao",
                        descricao_tecnica=(
                            f"Tubo aço carbono {mat}, {dn}\", {sch}, ASME B36.10"
                            + (f" — comprimento {comprimento} m" if comprimento else "")
                        ),
                        norma_origem="EBP_SPOOL",
                        secao_norma="spool_list",
                        diametro_nps=dn.replace('"', '').replace("\"", ""),
                        schedule=sch,
                        unidade="M" if comprimento else "UN",
                        confianca=0.90,
                        observacoes=f"Spool {spool_id}" if spool_id else None,
                    ))

                    fl_qty = getattr(spool, "flange_quantidade", None)
                    fl_classe = getattr(spool, "flange_classe", None) or "150#"
                    if fl_qty and fl_qty > 0:
                        all_specs.append(EspecificacaoMaterial(
                            tipo_material="flange_pescoço",
                            descricao_tecnica=(
                                f"Flange pescoço de solda ASTM A105, {dn}\", "
                                f"{fl_classe}, ASME B16.5"
                            ),
                            norma_origem="EBP_SPOOL",
                            secao_norma="spool_list",
                            diametro_nps=dn.replace('"', ''),
                            pressao_classe=fl_classe,
                            confianca=0.88,
                        ))

            added = len(all_specs) - n_spool_before
            log.info(f"  [EBP spool] {added} specs injetadas de {len(spool_list)} spool(s)")
        else:
            tag = escopo.tag_linha_principal or ""
            piping_cls = _extract_piping_class(tag)
            nps_val = _extract_nps(tag)
            if nps_val and piping_cls:
                log.info(f"  [EBP fallback] Sem spool_list, injetando tubo/flange genérico para {nps_val}\" {piping_cls}")
                all_specs.append(EspecificacaoMaterial(
                    tipo_material="tubo_conducao",
                    descricao_tecnica=f"Tubo aço carbono API 5L GrB, {nps_val}\", SCH 40, ASME B36.10 — piping class {piping_cls}",
                    norma_origem="ESCOPO_TAG",
                    secao_norma="tag_linha_principal",
                    diametro_nps=nps_val,
                    schedule="SCH 40",
                    confianca=0.70,
                    observacoes="Estimado pela tag da linha — sem spool_list no EBP",
                ))
                all_specs.append(EspecificacaoMaterial(
                    tipo_material="flange_pescoço",
                    descricao_tecnica=f"Flange pescoço de solda ASTM A105, {nps_val}\", 150#, ASME B16.5",
                    norma_origem="ESCOPO_TAG",
                    secao_norma="tag_linha_principal",
                    diametro_nps=nps_val,
                    pressao_classe="150#",
                    confianca=0.65,
                    observacoes="Estimado pela tag da linha — piping class B10S padrão 150#",
                ))
                all_specs.append(EspecificacaoMaterial(
                    tipo_material="junta_espiralada",
                    descricao_tecnica=f"Junta espiralada AISI 316/grafite, {nps_val}\", 150#, ASME B16.20",
                    norma_origem="ESCOPO_TAG",
                    secao_norma="tag_linha_principal",
                    diametro_nps=nps_val,
                    pressao_classe="150#",
                    confianca=0.65,
                ))
                all_specs.append(EspecificacaoMaterial(
                    tipo_material="parafuso_estojo",
                    descricao_tecnica=f"Parafuso estojo ASTM A193 GrB7 Zn-Ni, {nps_val}\" flange 150#",
                    norma_origem="ESCOPO_TAG",
                    secao_norma="tag_linha_principal",
                    diametro_nps=nps_val,
                    pressao_classe="150#",
                    confianca=0.65,
                ))

        # ── INJEÇÃO 0c: Specs visuais de isométricos (Qwen2.5-VL) ───────────────
        iso_specs = getattr(escopo, "isometric_specs", None) or []
        if iso_specs:
            n_iso_before = len(all_specs)
            for iso in iso_specs:
                all_specs.append(EspecificacaoMaterial(
                    tipo_material=iso.tipo_material,
                    descricao_tecnica=iso.descricao_tecnica,
                    norma_origem=f"ISOMETRICO:{iso.fonte_imagem}",
                    secao_norma="isometrico_visual",
                    diametro_nps=iso.diametro_nps,
                    schedule=iso.schedule,
                    unidade=iso.unidade,
                    confianca=iso.confianca,
                    observacoes=iso.notas,
                ))
            log.info(f"  [Isométrico visual] {len(all_specs) - n_iso_before} specs injetadas")

        log.info(
            f"  [Injeção direta] Total antes do Qdrant: {len(all_specs)} specs "
            f"(IEIS + EBP + isométrico)"
        )

        for norma_id in normas:
            log.info(f"  Consultando {norma_id}...")
            queries = self._build_queries(norma_id, escopo)
            log.debug(f"    Queries geradas: {queries}")
            chunks = self._retrieve_chunks(norma_id, queries)

            if not chunks:
                log.warning(f"  Nenhum chunk encontrado para {norma_id} no Qdrant.")
                normas_sem_resultado.append(norma_id)
                continue

            log.info(f"  {len(chunks)} chunks recuperados para {norma_id}")
            total_chunks += len(chunks)

            raw_items = self._extract_with_llm(norma_id, chunks, escopo)

            log.debug(f"    LLM retornou {len(raw_items)} items brutos para {norma_id}")
            specs = self._parse_specs(raw_items, norma_id)
            log.info(f"  {len(specs)} especificações válidas extraídas de {norma_id}")
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
