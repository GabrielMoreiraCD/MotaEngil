"""
triage_agent.py — Agente de Triagem Unificado (Four-Pass) com Ollama local
====================================================================================================================
Extrai dados estruturados de formulário CSV, Memorial Descritivo, IEIS e EBP
usando qwen2.5:7b-instruct via Ollama local (100% offline, sem APIs cloud).
"""
import logging
import os
import sys
import re
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import config
from core.models import ollama_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TriageAgent")

# Regex para extrair normas N-XXXX do texto bruto (fallback quando LLM falha)
RE_NORMA_PETROBRAS = re.compile(r'\bN[-‐](\d{3,4})\b')

# Regex de fallback para campos críticos (pós-LLM)
RE_PLATAFORMA = re.compile(r'\b(P-\d{2,3}|FPSO-[A-Z0-9-]+|PCH-\d|FPZ-\d{2})\b')
RE_TAG_LINHA  = re.compile(r'\b(\d{1,4}["″\u201d][-\w]+-[A-Z]\d+[A-Z]+-\d+)\b')
RE_LP         = re.compile(r'\bLP[-\s]?(\d{3,4})\b', re.IGNORECASE)
RE_PIPING_CLASS = re.compile(r'\b([A-Z]\d{1,2}[A-Z]{1,2})\b')

# Regex para detectar placeholders de data/plataforma errônea
RE_DATA_PLACEHOLDER = re.compile(r'\|\s*\d{4}')

# ─────────────────────────────────────────────────────────────────────────────
# Skeletons JSON para cada pass
# ─────────────────────────────────────────────────────────────────────────────
SKELETON_CSV = """{
  "id_servico": "???",
  "titulo_servico": "???",
  "plataforma": "???",
  "ativo": "???",
  "ordem_manutencao_om": "???",
  "servico_critico": "???",
  "sistema": "???",
  "local_aplat": "???",
  "tag_equipamento_principal": "???",
  "tag_linha_principal": "???",
  "documentos_referencia": ["???"],
  "tarefas_execucao": ["???"],
  "notas_e_restricoes": ["???"],
  "numero_ze": "???",
  "materiais_criticos": ["???"],
  "servicos_simultaneos": ["???"],
  "estimativa_mao_de_obra": "???",
  "estimativa_prazo_horas": "???"
}"""

SKELETON_MD = """{
  "normas_petrobras_aplicaveis": ["???"],
  "detalhamento_por_disciplina": [
    {
      "disciplina_ou_sistema": "???",
      "tarefas": ["???"],
      "documentos_referencia": ["???"],
      "tags_relacionados": ["???"]
    }
  ]
}"""

SKELETON_IEIS = """{
  "material_base_tubo": "???",
  "material_base_acessorios": "???",
  "processo_soldagem": "???",
  "metal_adicao": "???",
  "classificacao_aws": "???",
  "ndt_requerido": ["???"],
  "normas_aplicaveis": ["???"],
  "classe_tubo": "???",
  "pre_aquecimento_min_C": "???",
  "pwht_requerido": "???"
}"""

SKELETON_EBP = """{
  "isometricos_referenciados": ["???"],
  "spool_list": [
    {
      "spool_id": "???",
      "material_tubo": "???",
      "dn": "???",
      "schedule": "???",
      "comprimento_m": "???",
      "flange_quantidade": "???",
      "flange_tipo": "???",
      "flange_classe": "???"
    }
  ],
  "piping_class_referencia": "???",
  "normas_projeto": ["???"]
}"""


class UnifiedTriageAgent:
    def __init__(self):
        logger.info(f"Inicializando Agente de Triagem (Four-Pass) — Ollama ({config.OLLAMA_MODEL})")

    # ── Limpeza ────────────────────────────────────────────────────────────────

    def _clean_csv_text(self, raw_text: str) -> str:
        raw_text = re.sub(r'(?i)\bSim\b\s*;+\s*X\s*;+\s*N[ãa]o\b', '[RESPOSTA: NÃO]', raw_text)
        raw_text = re.sub(r'(?i)X\s*;+\s*Sim\b(?:[\s;]*N[ãa]o\b)?', '[RESPOSTA: SIM]', raw_text)
        cleaned = []
        for line in raw_text.split('\n'):
            segs = [s.strip() for s in line.split(';') if s.strip()]
            if segs:
                cleaned.append(" | ".join(segs))
        return "\n".join(cleaned)

    def _clean_md_text(self, raw_text: str) -> str:
        return re.sub(r'\n{3,}', '\n\n', raw_text).strip()

    def _read_file(self, file_path: Path) -> str:
        for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
            try:
                return file_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Não foi possível decodificar: {file_path}")

    def _extract_text_from_file(self, file_path: Path) -> str:
        if file_path.suffix.lower() == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(file_path))
                pages_text = [page.get_text() for page in doc]
                doc.close()
                text = "\n".join(pages_text)
                logger.info(f"  PDF extraído via PyMuPDF: {len(text)} chars de {len(pages_text)} páginas.")
                return text
            except ImportError:
                logger.error("PyMuPDF (fitz) não instalado. Execute: pip install pymupdf>=1.24.0")
                raise
            except Exception as e:
                logger.error(f"Falha ao extrair PDF '{file_path.name}': {e}")
                raise
        return self._read_file(file_path)

    # ── LLM call unificado ──────────────────────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Chama Ollama local com system + user prompts."""
        return ollama_chat(system_prompt, user_prompt)

    # ── Extrator JSON robusto ──────────────────────────────────────────────────

    def _extract_json_robust(self, text: str, label: str = "") -> dict:
        text = re.sub(r'```(?:json)?', '', text, flags=re.IGNORECASE).strip()
        start = text.find('{')
        if start == -1:
            logger.error(f"[{label}] JSON não encontrado na resposta.")
            return {}

        depth = 0
        in_string = False
        escape_next = False
        end = -1

        for i, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False; continue
            if ch == '\\' and in_string:
                escape_next = True; continue
            if ch == '"':
                in_string = not in_string; continue
            if in_string:
                continue
            if   ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1; break

        if end == -1:
            logger.error(f"[{label}] Chaves JSON não balanceadas.")
            return {}

        json_str = text[start:end]
        json_str = re.sub(r'"(\?\?\?)"', 'null', json_str)
        json_str = re.sub(r',\s*"?\.\.\."?', '', json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[{label}] JSON inválido: {e}")
            return {}

    # ── Pós-processamento ──────────────────────────────────────────────────────

    def _sanitize(self, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        out = {}
        for k, v in data.items():
            if v in ("???", None, ""):
                out[k] = None
            elif isinstance(v, list):
                items = [i for i in v if i not in ("???", None, "")]
                out[k] = [self._sanitize(i) if isinstance(i, dict) else i for i in items]
            elif isinstance(v, dict):
                out[k] = self._sanitize(v)
            elif isinstance(v, bool):
                out[k] = "Sim" if v else "Não"
            else:
                out[k] = v
        return out

    # ── Pass 3: IEIS ───────────────────────────────────────────────────────────

    def _process_ieis(self, ieis_path: str) -> dict:
        ieis_file = Path(ieis_path)
        if not ieis_file.exists():
            logger.warning(f"IEIS não encontrado: {ieis_path}")
            return {}

        texto = self._clean_md_text(self._extract_text_from_file(ieis_file))
        # Limita contexto para o modelo local (qwen2.5 7B → ~6k chars seguros)
        texto = texto[:8000]
        logger.info(f"Pass 3/4 — Extraindo especificações do IEIS via Ollama ({config.OLLAMA_MODEL})...")

        try:
            system_p3 = (
                "Você é engenheiro de soldagem Petrobras. Extraia dados técnicos do IEIS.\n"
                "REGRAS:\n"
                "- Substitua '???' pelo valor real. Se não existir, use null ou [].\n"
                "- 'material_base_tubo': material do tubo (ex: API 5L GrB, ASTM A106 GrB). NÃO use siglas internas como 'EPS'.\n"
                "- 'metal_adicao': código do eletrodo/arame (ex: E7018, ER70S-6).\n"
                "- 'classificacao_aws': código AWS completo (ex: AWS A5.1 E7018).\n"
                "- 'ndt_requerido': lista com siglas (LP, VT, UT, RT, MT).\n"
                "- 'normas_aplicaveis': APENAS N-XXXX citadas no documento.\n"
                "- Retorne APENAS JSON válido."
            )
            user_p3 = f"<IEIS>\n{texto}\n</IEIS>\n\nJSON preenchido:\n{SKELETON_IEIS}"
            raw = self._call_llm(system_p3, user_p3)
            data = self._sanitize(self._extract_json_robust(raw, "IEIS"))
            logger.info(f"Pass 3 concluído. Eletrodo: {data.get('metal_adicao')} | "
                        f"NDT: {data.get('ndt_requerido')}")
            return data
        except Exception as e:
            logger.error(f"Falha no Pass 3 (IEIS): {e}", exc_info=True)
            return {}

    # ── Pass 4: EBP ────────────────────────────────────────────────────────────

    def _process_ebp(self, ebp_path: str) -> dict:
        ebp_file = Path(ebp_path)
        if not ebp_file.exists():
            logger.warning(f"EBP não encontrado: {ebp_path}")
            return {}

        texto = self._clean_md_text(self._extract_text_from_file(ebp_file))
        # Limita contexto para modelo local
        texto = texto[:10000]
        logger.info(f"Pass 4/4 — Extraindo spool list do EBP via Ollama ({config.OLLAMA_MODEL})...")

        try:
            system_p4 = (
                "Você é engenheiro de tubulação Petrobras. Extraia dados do Planejamento Executivo (EBP).\n"
                "REGRAS:\n"
                "- Substitua '???' pelo valor real. Se não existir, use null ou [].\n"
                "- 'spool_list': cada spool com DN (polegadas, ex: '2\"'), schedule (ex: SCH 40) e comprimento em metros.\n"
                "- 'dn': diâmetro nominal em polegadas (ex: '2\"'). NÃO use código de piping class como schedule.\n"
                "- 'schedule': espessura da parede (ex: SCH 40, STD). Se só tiver piping class, use SCH 40.\n"
                "- 'comprimento_m': número decimal (ex: 2.08). null se não encontrado.\n"
                "- 'flange_quantidade': número inteiro de flanges no spool. null se não encontrado.\n"
                "- 'piping_class_referencia': código da spec de tubulação (ex: I-ET-3010.68-...).\n"
                "- 'normas_projeto': refs de engenharia no formato I-ET-XXXX ou DR-ENGP-XXXX.\n"
                "- Retorne APENAS JSON válido."
            )
            user_p4 = f"<EBP>\n{texto}\n</EBP>\n\nJSON preenchido:\n{SKELETON_EBP}"
            raw = self._call_llm(system_p4, user_p4)
            data = self._sanitize(self._extract_json_robust(raw, "EBP"))
            n_spools = len(data.get("spool_list") or [])
            logger.info(f"Pass 4 concluído. Spools encontrados: {n_spools} | "
                        f"Piping spec: {data.get('piping_class_referencia')}")
            return data
        except Exception as e:
            logger.error(f"Falha no Pass 4 (EBP): {e}", exc_info=True)
            return {}

    # ── Orquestrador: Four-Pass ────────────────────────────────────────────────

    def process_project_files(self, csv_path: str, md_path: str,
                               ieis_path: str | None = None,
                               ebp_path: str | None = None) -> dict:
        csv_file, md_file = Path(csv_path), Path(md_path)

        if not csv_file.exists():
            logger.error(f"CSV não encontrado: {csv_file}"); return {}
        if not md_file.exists():
            logger.error(f"MD não encontrado: {md_file}"); return {}

        cleaned_csv = self._clean_csv_text(self._extract_text_from_file(csv_file))
        cleaned_md  = self._clean_md_text(self._extract_text_from_file(md_file))

        llm_label = f"Ollama ({config.OLLAMA_MODEL})"

        # ── PASS 1: extração do formulário ─────────────────────────────────────
        logger.info(f"Pass 1/4 — Extraindo dados do Formulário CSV via {llm_label}...")
        try:
            system_p1 = (
                "Você é um extrator de dados. Leia o formulário e preencha o JSON.\n"
                "REGRAS:\n"
                "- Substitua '???' pelo valor real. Se não existir, use null ou [].\n"
                "- 'id_servico': código no formato SS-XX, LC-XXX ou similar.\n"
                "- 'plataforma': apenas o código da plataforma (ex: P-54). NÃO inclua nome de campo ou data.\n"
                "- 'ativo': nome do ativo (ex: 'Libra'). NÃO inclua datas ou códigos de plataforma.\n"
                "- 'tag_equipamento_principal': TAG exato do equipamento (ex: P-2001A). NÃO inclua data.\n"
                "- 'servico_critico': 'Sim' se [RESPOSTA: SIM], senão 'Não'.\n"
                "- 'tarefas_execucao': ações listadas no item 4.1.\n"
                "- 'notas_e_restricoes': apenas as linhas sob 'NOTAS:' no item 4.1.\n"
                "- Retorne APENAS JSON válido. Sem texto antes ou depois."
            )
            user_p1 = f"<FORMULARIO>\n{cleaned_csv[:8000]}\n</FORMULARIO>\n\nJSON preenchido:\n{SKELETON_CSV}"
            raw_csv = self._call_llm(system_p1, user_p1)
            data_csv = self._sanitize(self._extract_json_robust(raw_csv, "CSV"))
            logger.info(f"Pass 1 concluído. Campos extraídos: {list(data_csv.keys())}")
        except Exception as e:
            logger.error(f"Falha no Pass 1: {e}", exc_info=True)
            data_csv = {}

        # ── PASS 2: extração do Memorial Descritivo ────────────────────────────
        logger.info(f"Pass 2/4 — Extraindo dados do Memorial Descritivo via {llm_label}...")
        try:
            system_p2 = (
                "Você é um extrator de dados de engenharia offshore Petrobras.\n"
                "REGRAS:\n"
                "- Substitua '???' pelo valor real. Se não existir, use null ou [].\n"
                "- 'normas_petrobras_aplicaveis': APENAS códigos N-XXX citados no texto (ex: N-279, N-115).\n"
                "  Se não houver normas explícitas no texto, retorne [].\n"
                "- 'detalhamento_por_disciplina': um objeto por sistema/disciplina no ESCOPO.\n"
                "- Retorne APENAS JSON válido. Sem texto antes ou depois."
            )
            user_p2 = f"<MEMORIAL_DESCRITIVO>\n{cleaned_md[:8000]}\n</MEMORIAL_DESCRITIVO>\n\nJSON preenchido:\n{SKELETON_MD}"
            raw_md = self._call_llm(system_p2, user_p2)
            data_md = self._sanitize(self._extract_json_robust(raw_md, "MD"))
            logger.info(f"Pass 2 concluído. Normas: {data_md.get('normas_petrobras_aplicaveis', [])}")
        except Exception as e:
            logger.error(f"Falha no Pass 2: {e}", exc_info=True)
            data_md = {}

        # ── Regex: sempre extrai N-XXXX do texto e merge com resultado LLM ──
        RE_PETROBRAS_STRICT = re.compile(r'\bN-(\d{3,4})\b')
        normas_llm_raw = data_md.get("normas_petrobras_aplicaveis") or []

        def _normalize_norma(n: str) -> str:
            m = RE_PETROBRAS_STRICT.match(n)
            return f"N-{m.group(1).zfill(4)}" if m else n

        normas_llm_filtradas = [_normalize_norma(n) for n in normas_llm_raw if RE_PETROBRAS_STRICT.match(n)]
        if len(normas_llm_filtradas) < len(normas_llm_raw):
            descartadas = set(normas_llm_raw) - set(normas_llm_filtradas)
            logger.info(f"  Normas descartadas (não-Petrobras): {sorted(descartadas)}")

        normas_regex = [f"N-{m.zfill(4)}" for m in RE_NORMA_PETROBRAS.findall(cleaned_md)]
        normas_regex_dedup = list(dict.fromkeys(normas_regex))
        normas_merged = list(dict.fromkeys(normas_llm_filtradas + normas_regex_dedup))
        if normas_merged != normas_llm_raw:
            logger.info(f"  Normas finais (regex+LLM): {normas_merged}")
        data_md["normas_petrobras_aplicaveis"] = normas_merged

        # Regex N-XXXX também no CSV
        normas_regex_csv = [f"N-{m.zfill(4)}" for m in RE_NORMA_PETROBRAS.findall(cleaned_csv)]
        normas_merged_final = list(dict.fromkeys(normas_merged + normas_regex_csv))
        if normas_merged_final != normas_merged:
            logger.info(f"  Normas adicionais do CSV: {list(set(normas_merged_final) - set(normas_merged))}")
        data_md["normas_petrobras_aplicaveis"] = normas_merged_final

        # ── PASS 3: IEIS (opcional) ────────────────────────────────────────────
        data_ieis = {}
        if ieis_path:
            data_ieis = self._process_ieis(ieis_path)
            normas_ieis = [f"N-{m.zfill(4)}" for m in RE_NORMA_PETROBRAS.findall(
                self._clean_md_text(self._extract_text_from_file(Path(ieis_path)))
            )]
            normas_prev = data_md.get("normas_petrobras_aplicaveis") or []
            data_md["normas_petrobras_aplicaveis"] = list(dict.fromkeys(normas_prev + normas_ieis))

        # ── PASS 4: EBP (opcional) ─────────────────────────────────────────────
        data_ebp = {}
        if ebp_path:
            data_ebp = self._process_ebp(ebp_path)
            normas_n = [f"N-{m.zfill(4)}" for m in RE_NORMA_PETROBRAS.findall(
                self._clean_md_text(self._extract_text_from_file(Path(ebp_path)))
            )]
            normas_prev = data_md.get("normas_petrobras_aplicaveis") or []
            data_md["normas_petrobras_aplicaveis"] = list(dict.fromkeys(normas_prev + normas_n))

        # ── MERGE ──────────────────────────────────────────────────────────────
        resultado = {**data_csv, **data_md}
        if data_ieis:
            resultado["especificacoes_soldagem"] = data_ieis
        if data_ebp:
            resultado["spool_list"] = data_ebp.get("spool_list") or []
            resultado["isometricos_referenciados"] = data_ebp.get("isometricos_referenciados") or []
            resultado["piping_class_referencia"] = data_ebp.get("piping_class_referencia")
            docs_ref = resultado.get("documentos_referencia") or []
            for np in (data_ebp.get("normas_projeto") or []):
                if np and np not in docs_ref:
                    docs_ref.append(np)
            resultado["documentos_referencia"] = docs_ref

        # ── Fallbacks regex ────────────────────────────────────────────────────
        tag = resultado.get("tag_linha_principal") or ""
        id_svc = resultado.get("id_servico") or ""
        if not id_svc or id_svc == tag or id_svc in ("???", "null", "LP-XXX"):
            RE_SERVICE_ID = re.compile(r'\b([A-Z]{1,3}-\d{1,5})\b')
            titulo = resultado.get("titulo_servico") or ""
            m_svc = RE_SERVICE_ID.search(titulo)
            if m_svc and m_svc.group(1) != tag:
                resultado["id_servico"] = m_svc.group(1)
                logger.info(f"  id_servico corrigido via titulo_servico: {resultado['id_servico']}")
            else:
                resultado["id_servico"] = csv_file.stem
                logger.info(f"  id_servico fallback para nome do arquivo: {resultado['id_servico']}")

        # Ativo/tag_equipamento — detecta placeholders "X | AAAA"
        ativo_val = resultado.get("ativo") or ""
        if RE_DATA_PLACEHOLDER.search(ativo_val):
            logger.info(f"  ativo corrigido (placeholder detectado): '{ativo_val}' → None")
            resultado["ativo"] = None

        tag_eq_val = resultado.get("tag_equipamento_principal") or ""
        if RE_DATA_PLACEHOLDER.search(tag_eq_val):
            logger.info(f"  tag_equipamento_principal corrigido: '{tag_eq_val}' → None")
            resultado["tag_equipamento_principal"] = None

        # Plataforma — regex fallback
        plataforma_llm = resultado.get("plataforma") or ""
        if not RE_PLATAFORMA.match(plataforma_llm.strip()):
            for texto_fonte in [cleaned_csv, cleaned_md]:
                m = RE_PLATAFORMA.search(texto_fonte)
                if m:
                    resultado["plataforma"] = m.group(1)
                    logger.info(f"  plataforma corrigida via regex: {resultado['plataforma']}")
                    break

        # Tag linha — regex fallback
        tag_llm = resultado.get("tag_linha_principal") or ""
        if not tag_llm or tag_llm in ("LP-XXX", "???", "null") or not RE_TAG_LINHA.match(tag_llm.strip()):
            for texto_fonte in [cleaned_csv, cleaned_md]:
                m = RE_TAG_LINHA.search(texto_fonte)
                if m:
                    resultado["tag_linha_principal"] = m.group(1)
                    logger.info(f"  tag_linha_principal corrigida via regex: {resultado['tag_linha_principal']}")
                    break

        # Numero ZE / LP
        ze_llm = resultado.get("numero_ze") or ""
        if not ze_llm or ze_llm in ("LP-XXX", "???", "null"):
            for texto_fonte in [cleaned_csv, cleaned_md]:
                m = RE_LP.search(texto_fonte)
                if m:
                    resultado["numero_ze"] = f"LP-{m.group(1)}"
                    logger.info(f"  numero_ze corrigido via regex: {resultado['numero_ze']}")
                    break

        logger.info("Merge concluído. Triagem finalizada.")
        return resultado


if __name__ == "__main__":
    TEST_CSV = r"C:\luza_datasets\Rag-pipeline-main\Data\raw\DE-LC-029.csv"
    TEST_MD  = r"C:\luza_datasets\Rag-pipeline-main\Data\raw\MD-3010.68-1200-941-M9C-003.txt"

    agent = UnifiedTriageAgent()
    if os.path.exists(TEST_CSV) and os.path.exists(TEST_MD):
        result = agent.process_project_files(TEST_CSV, TEST_MD)
        print("\n=== RESULTADO DA TRIAGEM UNIFICADA ===")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print("Verifique os caminhos TEST_CSV e TEST_MD.")
