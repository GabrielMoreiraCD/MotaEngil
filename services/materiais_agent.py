"""
materiais_agent.py — Etapa 4: Mapeamento de Especificações → Catálogo de Materiais
===================================================================================
Para cada EspecificacaoMaterial extraída na Etapa 3, busca no Qdrant
(coleção catalogo_materiais_v1) os itens mais similares e usa Llama3 para
disambiguation quando há múltiplos candidatos próximos.

Fluxo por especificação:
  1. Constrói query espelhando o formato de embedding usado na ingestão (build_embed_text)
  2. Busca catalogo_materiais_v1 com filtros de payload (diametro, categoria)
  3. Se score do melhor >= 0.65: mapeado direto
  4. Se score 0.40–0.65 e múltiplos candidatos: disambiguation via Llama3
  5. Se score < 0.40: unmapped com descrição técnica preservada

Saída: MateriaisRequisitados (persistida em etapa4_materiais_result.json)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

from core.config import config
from core.schemas import (
    CatalogItemMatch,
    EscopoTriagemUnificado,
    EspecificacaoMaterial,
    MateriaisRequisitados,
    NormasConsultaResult,
)

log = logging.getLogger(__name__)

# Thresholds de score coseno para decisão de mapeamento
# Elevados para reduzir falsos positivos (anteriormente 0.65 / 0.40)
SCORE_DIRECT_MATCH = 0.72   # Mapeamento direto sem disambiguation
SCORE_PARTIAL_MATCH = 0.55  # Tenta disambiguation; abaixo → unmapped

# Máximo de candidatos enviados ao LLM para disambiguation
MAX_CANDIDATES_LLM = 5

# ─────────────────────────────────────────────────────────────────────────────
# Mapa de tipo_material → prefixos de categoria no catálogo
# (nome do stem do arquivo XLSX = campo "categoria" no payload Qdrant)
# ─────────────────────────────────────────────────────────────────────────────
TIPO_MATERIAL_TO_CATEGORIA_HINTS: dict[str, list[str]] = {
    # ── Tubos ──────────────────────────────────────────────────────────────────
    "tubo_reparo":        ["TUBO CONDUÇÃO", "TUBO DE CONDUÇÃO", "TUBO CARBONO", "TUBO INOX"],
    "tubo_conducao":      ["TUBO CONDUÇÃO", "TUBO DE CONDUÇÃO", "TUBO CARBONO", "TUBO INOX"],
    # ── Eletrodos e consumíveis de solda ──────────────────────────────────────
    "eletrodo_revestido": ["ELETRODO", "CONSUMIVEL", "CONSUMÍVEL", "SOLDA"],
    "eletrodo_smaw":      ["ELETRODO", "CONSUMIVEL", "CONSUMÍVEL", "SOLDA"],
    "arame_solda":        ["ARAME", "CONSUMIVEL", "CONSUMÍVEL", "SOLDA"],
    # ── END ───────────────────────────────────────────────────────────────────
    "consumivel_end":     ["ENSAIO", "INSPECAO", "INSPEÇÃO", "NDT", "LIQUIDO", "LÍQUIDO",
                           "PENETRANTE", "REVELADOR", "REMOVEDOR"],
    # ── Flanges ───────────────────────────────────────────────────────────────
    "flanges":            ["FLANGE"],
    "flange_pescoço":     ["FLANGE", "FLANGE PESCOÇO", "FLANGE PESCOÇO DE SOLDA",
                           "FLANGE SOLDA DE TOPO"],
    "flange_encaixe":     ["FLANGE", "FLANGE ENCAIXE", "FLANGE DE ENCAIXE",
                           "FLANGE SOCKET WELD"],
    # ── Juntas / gaxetas ──────────────────────────────────────────────────────
    "junta_vedacao":      ["JUNTA", "GAXETA", "VEDAÇÃO"],
    "junta_espiralada":   ["JUNTA", "JUNTA ESPIRALADA", "GAXETA ESPIRALADA"],
    # ── Parafusos / fixações ──────────────────────────────────────────────────
    "parafuso":           ["PARAFUSO", "PORCA", "ARRUELA"],
    "parafuso_estojo":    ["PARAFUSO ESTOJO", "PARAFUSO", "PORCA"],
    # ── Válvulas ──────────────────────────────────────────────────────────────
    "valvula":            ["VÁLVULA", "VALVULA"],
    "valvula_esfera":     ["VÁLVULA DE ESFERA", "VÁLVULA ESFERA", "VÁLVULA"],
    # ── Conectores / fittings ─────────────────────────────────────────────────
    "conector":           ["CONECTOR", "FITTING", "ENCAIXE", "COTOVELO", "CURVA", "TE ", "REDUÇÃO"],
    "meia_luva":          ["MEIA LUVA", "MEIA-LUVA", "CONECTOR", "FITTING"],
    "tee_reto":           ["TEE", "TE ", "ACESSÓRIO DE TUBULAÇÃO", "FITTING"],
    # ── Tampões ───────────────────────────────────────────────────────────────
    "tampao":             ["TAMPÃO", "PLUGUE", "ENCAIXE"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

RE_NPS = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:"|\'\'|polegadas?|in\b)', re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# Mapa de canonicalização de tipo_material
# LLMs retornam texto livre; este mapa normaliza para as chaves do dicionário
# TIPO_MATERIAL_TO_CATEGORIA_HINTS acima.
# None = sem filtro de categoria (busca em todo o catálogo)
# ─────────────────────────────────────────────────────────────────────────────
_TIPO_CANONICAL: dict[str, str | None] = {
    # tubos
    "tubo": "tubo_conducao", "tubulação": "tubo_conducao",
    "tubo de condução": "tubo_conducao", "tubo condução": "tubo_conducao",
    "tubo carbono": "tubo_conducao", "tubo inox": "tubo_conducao",
    # flanges
    "flange pescoço": "flange_pescoço", "flange pescoço de solda": "flange_pescoço",
    "flange solda de topo": "flange_pescoço",
    "flange encaixe": "flange_encaixe", "flange de encaixe": "flange_encaixe",
    "flange socket weld": "flange_encaixe",
    "flange": "flanges",
    # juntas
    "junta espiralada": "junta_espiralada", "junta vedação": "junta_vedacao",
    "junta": "junta_espiralada", "gaxeta": "junta_espiralada",
    # parafusos
    "parafuso estojo": "parafuso_estojo", "parafuso": "parafuso_estojo",
    "porca": "parafuso_estojo", "arruela": "parafuso_estojo",
    # válvulas
    "válvula esfera": "valvula_esfera", "valvula esfera": "valvula_esfera",
    "válvula de esfera": "valvula_esfera",
    "válvula": "valvula", "valvula": "valvula",
    # fittings
    "meia-luva": "meia_luva", "meia luva": "meia_luva",
    "tee": "tee_reto", "te reto": "tee_reto", "tee reto": "tee_reto",
    "tampão": "tampao", "tampao": "tampao", "plugue": "tampao",
    # eletrodos
    "eletrodo": "eletrodo_smaw", "eletrodo revestido": "eletrodo_revestido",
    "consumível de soldagem": "eletrodo_smaw",
    # END
    "consumível end": "consumivel_end", "liquido penetrante": "consumivel_end",
    "líquido penetrante": "consumivel_end",
    # tipos sem categoria definida (busca sem filtro)
    "conexão universal": None, "conexao universal": None,
    "distribuidor": None, "distribuidor de fluxo": None,
}


def _normalize_tipo_material(tipo: str) -> str | None:
    """
    Normaliza tipo_material livre do LLM para a chave canônica do catálogo.
    Retorna None se tipo não tiver hint de categoria (busca sem filtro).
    Retorna o próprio tipo se não houver mapeamento (usa como fallback).
    """
    tipo_lower = tipo.lower().strip()
    # Busca match exato primeiro
    if tipo_lower in _TIPO_CANONICAL:
        return _TIPO_CANONICAL[tipo_lower]
    # Busca match parcial (início da string)
    for key, val in _TIPO_CANONICAL.items():
        if tipo_lower.startswith(key) or key in tipo_lower:
            return val
    return tipo  # mantém original se não encontrar


def _extract_nps(text: str) -> str:
    m = RE_NPS.search(text)
    return m.group(1).replace(",", ".") if m else ""


def _build_catalog_query(spec: EspecificacaoMaterial) -> str:
    """
    Constrói query espelhando o formato build_embed_text() do ingest_materials.py.
    Segue a mesma ordem de EMBED_PRIORITY para maximizar alinhamento semântico.

    Formato: "CATEGORIA: X | DESCRICAO: Y | ESPECIFICACAO: Z | MATERIAL BASE: W | ..."
    """
    parts: list[str] = []

    # Normaliza tipo_material (LLM pode retornar texto livre)
    tipo_canonical = _normalize_tipo_material(spec.tipo_material or "")
    categoria_hints = TIPO_MATERIAL_TO_CATEGORIA_HINTS.get(tipo_canonical or "", [])
    if categoria_hints:
        parts.append(f"CATEGORIA: {categoria_hints[0]}")

    # Descrição técnica
    parts.append(f"DESCRICAO: {spec.descricao_tecnica}")

    # Especificação (AWS, ASTM, ASME — o que estiver disponível)
    specs_disponíveis = [
        s for s in [spec.especificacao_aws, spec.especificacao_astm, spec.especificacao_asme]
        if s
    ]
    if specs_disponíveis:
        parts.append(f"ESPECIFICACAO: {' | '.join(specs_disponíveis)}")

    # Dimensões
    if spec.diametro_nps:
        parts.append(f"DIAMETRO: {spec.diametro_nps} polegadas")
    if spec.schedule:
        parts.append(f"ESPESSURA: {spec.schedule}")
    if spec.pressao_classe:
        parts.append(f"PRESSAO: {spec.pressao_classe}")

    return " | ".join(parts)


def _build_qdrant_filter(spec: EspecificacaoMaterial) -> Optional[Filter]:
    """
    Constrói filtro de payload para restringir busca a categorias relevantes.
    Usa os hints de categoria do tipo_material para pre-filtrar o espaço de busca.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    tipo_canonical = _normalize_tipo_material(spec.tipo_material or "")
    if tipo_canonical is None:
        return None  # tipo explicitamente sem categoria → busca sem filtro
    categoria_hints = TIPO_MATERIAL_TO_CATEGORIA_HINTS.get(tipo_canonical, [])
    if not categoria_hints:
        return None  # sem filtro → busca em todo o catálogo

    # Filtra por categoria usando match parcial (a categoria = stem do XLSX)
    # Qdrant MatchAny faz match exato, então adicionamos os prefixos mais prováveis
    return Filter(
        must=[
            FieldCondition(
                key="categoria",
                match=MatchAny(any=categoria_hints),
            )
        ]
    )


def _candidates_to_json(candidates: list[dict]) -> str:
    """Formata candidatos do Qdrant como JSON legível para o LLM."""
    items = []
    for c in candidates:
        p = c.get("payload", {})
        items.append({
            "codigo": p.get("codigo", ""),
            "descricao": p.get("descricao", ""),
            "especificacao": p.get("especificacao", ""),
            "diametro": p.get("diametro", ""),
            "material_base": p.get("material_base", ""),
            "pressao": p.get("pressao", ""),
            "norma": p.get("norma", ""),
            "score": round(c.get("score", 0), 3),
        })
    return json.dumps(items, ensure_ascii=False, indent=2)


DISAMBIG_PROMPT = """\
Você é almoxarife técnico Petrobras. Dado um requisito e {n} candidatos do catálogo,
escolha o que MELHOR atende ao requisito.

REQUISITO:
- Descrição: {descricao_tecnica}
- Especificação: {especificacao}
- Diâmetro NPS: {diametro_nps}
- Piping class: {piping_class}

CANDIDATOS:
{candidatos_json}

Responda APENAS com o valor do campo "codigo" do candidato escolhido.
Se nenhum candidato for adequado, responda: NENHUM"""


# ─────────────────────────────────────────────────────────────────────────────
# MateriaisMappingAgent
# ─────────────────────────────────────────────────────────────────────────────

class MateriaisMappingAgent:
    """
    Mapeia especificações técnicas de materiais (saída da Etapa 3)
    para itens concretos do catálogo Qdrant (catalogo_materiais_v1).
    """

    def __init__(self):
        log.info("MateriaisMappingAgent: carregando embedder...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

        from core.models import get_llm
        self.llm = get_llm()

        # Verifica se a coleção de materiais existe no Qdrant
        self.catalog_available = self._check_catalog_available()
        if not self.catalog_available:
            log.warning(
                f"Coleção '{config.COLLECTION_MATERIAIS}' NÃO encontrada no Qdrant. "
                "Etapa 4 retornará todos os itens como 'unmapped'. "
                "Para habilitar mapeamento, ingira o catálogo com ingest_materials.py."
            )
        log.info("MateriaisMappingAgent: pronto.")

    def _check_catalog_available(self) -> bool:
        """Verifica se a coleção de materiais existe no Qdrant."""
        try:
            return self.qdrant.collection_exists(config.COLLECTION_MATERIAIS)
        except Exception as e:
            log.warning(f"Não foi possível verificar coleção '{config.COLLECTION_MATERIAIS}': {e}")
            return False

    # ── Search ────────────────────────────────────────────────────────────────

    def _search_catalog(
        self,
        query: str,
        spec: EspecificacaoMaterial,
        top_k: int = MAX_CANDIDATES_LLM + 3,
    ) -> list[dict]:
        """
        Busca catálogo de materiais com embedding + filtro de categoria opcional.
        Retorna lista de hits com score e payload.
        Retorna [] se a coleção não existir (catalog_available=False).
        """
        if not self.catalog_available:
            return []

        query_vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        qfilter = _build_qdrant_filter(spec)

        try:
            results = self.qdrant.search(
                collection_name=config.COLLECTION_MATERIAIS,
                query_vector=query_vec,
                query_filter=qfilter,
                limit=top_k,
                with_payload=True,
            )
            return [
                {
                    "score": r.score,
                    "id": r.id,
                    "payload": r.payload,
                }
                for r in results
            ]
        except Exception as e:
            log.warning(f"Erro ao buscar catálogo para '{query[:60]}...': {e}")
            # Tenta sem filtro como fallback
            if qfilter:
                log.info("  Tentando busca sem filtro de categoria...")
                try:
                    results = self.qdrant.search(
                        collection_name=config.COLLECTION_MATERIAIS,
                        query_vector=query_vec,
                        limit=top_k,
                        with_payload=True,
                    )
                    return [{"score": r.score, "id": r.id, "payload": r.payload} for r in results]
                except Exception as e2:
                    log.error(f"  Fallback também falhou: {e2}")
            return []

    # ── Disambiguation ────────────────────────────────────────────────────────

    def _disambiguate(
        self,
        spec: EspecificacaoMaterial,
        candidates: list[dict],
        piping_class: str,
    ) -> Optional[dict]:
        """
        Usa Llama3 para escolher o melhor candidato entre os finalistas.
        Retorna o candidato escolhido ou None se LLM responder NENHUM.
        """
        specs_str = " / ".join(filter(None, [
            spec.especificacao_aws,
            spec.especificacao_astm,
            spec.especificacao_asme,
        ])) or "não especificada"

        prompt = DISAMBIG_PROMPT.format(
            n=len(candidates),
            descricao_tecnica=spec.descricao_tecnica,
            especificacao=specs_str,
            diametro_nps=spec.diametro_nps or "não especificado",
            piping_class=piping_class or "não identificada",
            candidatos_json=_candidates_to_json(candidates[:MAX_CANDIDATES_LLM]),
        )

        try:
            resposta = self.llm.invoke(prompt).strip()
        except Exception as e:
            log.warning(f"Erro na disambiguation LLM: {e}")
            return candidates[0] if candidates else None

        if resposta.upper() in ("NENHUM", "NONE", ""):
            return None

        # Busca o candidato com o código retornado
        codigo_escolhido = resposta.strip().strip('"').strip("'")
        for c in candidates:
            p = c.get("payload", {})
            if p.get("codigo", "").strip() == codigo_escolhido:
                return c

        # Código não encontrado nos candidatos → usa o de maior score
        log.debug(f"Código '{codigo_escolhido}' não encontrado nos candidatos. Usando top-1.")
        return candidates[0]

    # ── Map single spec ───────────────────────────────────────────────────────

    def _map_spec(
        self,
        spec: EspecificacaoMaterial,
        piping_class: str,
    ) -> CatalogItemMatch:
        """
        Mapeia uma única EspecificacaoMaterial para o melhor item do catálogo.
        """
        query = _build_catalog_query(spec)
        candidates = self._search_catalog(query, spec)

        tipo_canonical = _normalize_tipo_material(spec.tipo_material or "")
        categoria_base = ""
        if tipo_canonical:
            hints = TIPO_MATERIAL_TO_CATEGORIA_HINTS.get(tipo_canonical, [])
            categoria_base = hints[0] if hints else spec.tipo_material
        else:
            categoria_base = spec.tipo_material or ""

        base_item = CatalogItemMatch(
            requisito_origem=spec.descricao_tecnica,
            descricao_catalogo=spec.descricao_tecnica,
            categoria_catalogo=categoria_base,
            source_file="",
            score_similaridade=0.0,
            mapeamento_status="unmapped",
            quantidade=1.0,
            quantidade_estimada=True,
        )

        if not candidates:
            if not self.catalog_available:
                base_item.observacoes = (
                    f"Catálogo '{config.COLLECTION_MATERIAIS}' não disponível no Qdrant. "
                    f"Especificação técnica: {spec.especificacao_aws or spec.especificacao_astm or spec.descricao_tecnica}. "
                    "Execute ingest_materials.py para indexar o catálogo."
                )
            else:
                base_item.observacoes = (
                    f"Nenhum resultado no catálogo para: {spec.descricao_tecnica}. "
                    f"Especificação: {spec.especificacao_aws or spec.especificacao_astm or 'N/A'}. "
                    "Verificar catálogo ou adicionar planilha de consumíveis."
                )
            return base_item

        best = candidates[0]
        best_score = best["score"]
        best_payload = best["payload"]

        if best_score >= SCORE_DIRECT_MATCH:
            # Mapeamento direto
            status = "matched"
            chosen = best
        elif best_score >= SCORE_PARTIAL_MATCH and len(candidates) > 1:
            # Tenta disambiguation
            chosen = self._disambiguate(spec, candidates, piping_class)
            status = "partial" if chosen else "unmapped"
            if not chosen:
                chosen = best  # fallback
                status = "partial"
        elif best_score >= SCORE_PARTIAL_MATCH:
            chosen = best
            status = "partial"
        else:
            # Score muito baixo → unmapped
            base_item.score_similaridade = best_score
            base_item.observacoes = (
                f"Score de similaridade muito baixo ({best_score:.2f}) para: "
                f"{spec.descricao_tecnica}. Verificar catálogo manualmente."
            )
            return base_item

        p = chosen["payload"]
        return CatalogItemMatch(
            requisito_origem=spec.descricao_tecnica,
            codigo=p.get("codigo") or None,
            descricao_catalogo=p.get("descricao", spec.descricao_tecnica),
            categoria_catalogo=p.get("categoria", ""),
            especificacao_catalogo=p.get("especificacao") or None,
            diametro=p.get("diametro") or spec.diametro_nps,
            material_base=p.get("material_base") or None,
            norma_catalogo=p.get("norma") or None,
            unidade_fornecimento=p.get("unidade", "UN"),
            source_file=p.get("source_file", ""),
            score_similaridade=chosen["score"],
            mapeamento_status=status,
            quantidade=1.0,
            quantidade_estimada=True,
        )

    # ── Main ──────────────────────────────────────────────────────────────────

    def process(
        self,
        normas_result: NormasConsultaResult,
        escopo: EscopoTriagemUnificado,
        output_path: Optional[Path] = None,
    ) -> MateriaisRequisitados:
        """
        Processa todas as especificações da Etapa 3 e mapeia para o catálogo.

        Args:
            normas_result: Saída da Etapa 3
            escopo: Dados do serviço (para piping class e contexto)
            output_path: Se fornecido, persiste resultado como JSON
        """
        log.info(f"[Etapa 4] Mapeando {len(normas_result.especificacoes_extraidas)} specs para o catálogo...")

        # Extrai piping class uma vez
        from services.normas_agent import _extract_piping_class
        piping_class = _extract_piping_class(escopo.tag_linha_principal)

        itens: list[CatalogItemMatch] = []

        for i, spec in enumerate(normas_result.especificacoes_extraidas, 1):
            log.info(f"  [{i}/{len(normas_result.especificacoes_extraidas)}] {spec.tipo_material}: {spec.descricao_tecnica[:60]}...")
            item = self._map_spec(spec, piping_class)
            itens.append(item)
            log.info(f"    → {item.mapeamento_status} (score={item.score_similaridade:.2f})")

        total_mapeados = sum(1 for i in itens if i.mapeamento_status in ("matched", "partial"))
        total_nao_mapeados = sum(1 for i in itens if i.mapeamento_status == "unmapped")

        result = MateriaisRequisitados(
            id_servico=normas_result.id_servico,
            itens=itens,
            total_requisitos=len(itens),
            total_mapeados=total_mapeados,
            total_nao_mapeados=total_nao_mapeados,
        )

        log.info(
            f"[Etapa 4] Concluído: {total_mapeados} mapeados, "
            f"{total_nao_mapeados} não mapeados de {len(itens)} specs"
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
            log.info(f"  Resultado salvo em: {output_path}")

        return result
