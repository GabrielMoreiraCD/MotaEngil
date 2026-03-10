''''
triage_agent.py - Implementação do Agente de Triagem Unificado utilizando abordagem Two-Pass para extração de dados do Formulário CSV e do Memorial Descritivo.
====================================================================================================================
'''
import logging
import os
import sys
import re
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import PromptTemplate
from core.models import get_llm
from core.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TriageAgent")

# Regex para extrair normas N-XXXX do texto bruto (fallback quando LLM falha)
RE_NORMA_PETROBRAS = re.compile(r'\bN[-‐](\d{3,4})\b')

# Regex de fallback para campos críticos (pós-LLM)
RE_PLATAFORMA = re.compile(r'\b(P-\d{2,3}|FPSO-[A-Z0-9-]+|PCH-\d|FPZ-\d{2})\b')
RE_TAG_LINHA  = re.compile(r'\b(\d{1,4}["″\u201d][-\w]+-[A-Z]\d+[A-Z]+-\d+)\b')
RE_LP         = re.compile(r'\bLP[-\s]?(\d{3,4})\b', re.IGNORECASE)
RE_PIPING_CLASS = re.compile(r'\b([A-Z]\d{1,2}[A-Z]{1,2})\b')

# ─────────────────────────────────────────────────────────────────────────────
# PASS 1: Skeleton focado APENAS no Formulário CSV
# Campos menores = modelo não se perde entre as duas fontes
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

# ─────────────────────────────────────────────────────────────────────────────
# PASS 2: Skeleton focado APENAS no Memorial Descritivo
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# PASS 3: Skeleton focado no IEIS (Instrução para Execução e Inspeção de Solda)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# PASS 4: Skeleton focado no EBP (Planejamento Executivo / Book of Planning)
# ─────────────────────────────────────────────────────────────────────────────
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


# Regex para detectar placeholders de data/plataforma errônea em campos do formulário
# Ex: "P-37 | 2027" ou "P-35 | 2026" — alucina plataforma/ativo de outro projeto
RE_DATA_PLACEHOLDER = re.compile(r'\|\s*\d{4}')


class UnifiedTriageAgent:
    def __init__(self, use_claude: bool = False, use_gemini: bool = True):
        logger.info("Inicializando Agente de Triagem Unificado (Four-Pass)...")

        # Gemini: padrão quando GEMINI_KEY disponível
        self.use_gemini = config.USE_GEMINI and bool(config.GEMINI_KEY)
        # Claude: legado, apenas se Gemini indisponível
        self.use_claude = (not self.use_gemini) and use_claude and config.USE_CLAUDE and bool(config.ANTHROPIC_API_KEY)

        self.gemini = None
        self.claude = None

        if self.use_gemini:
            try:
                from google import genai as _genai
                self.gemini = _genai.Client(api_key=config.GEMINI_KEY)
                logger.info(f"Triage usando Gemini ({config.GEMINI_MODEL}) para extração estruturada.")
            except Exception as e:
                logger.warning(f"Gemini não disponível ({e}). Fallback para Llama3.")
                self.use_gemini = False
                self.gemini = None
        elif self.use_claude:
            try:
                from core.models import get_claude_client
                self.claude = get_claude_client()
                logger.info(f"Triage usando Claude ({config.CLAUDE_MODEL}) para extração estruturada.")
            except Exception as e:
                logger.warning(f"Claude não disponível ({e}). Fallback para Llama3.")
                self.use_claude = False
                self.claude = None

        self.llm = get_llm()
        self.prompt_csv = self._build_prompt_csv()
        self.prompt_md = self._build_prompt_md()

    # ── Prompts ────────────────────────────────────────────────────────────────

    def _build_prompt_csv(self) -> PromptTemplate:
        template = (
            "Você é um extrator de dados. Preencha o JSON abaixo com valores do FORMULÁRIO.\n"
            "REGRAS:\n"
            "- Substitua '???' pelo valor real. Se não existir, use null (string) ou [] (lista).\n"
            "- 'id_servico': código no formato LC-XXX encontrado no título do formulário.\n"
            "- 'servico_critico': 'Sim' se [RESPOSTA: SIM], senão 'Não'.\n"
            "- 'numero_ze': número após 'Nº ZE' ou 'ZE:' no formulário.\n"
            "- 'tarefas_execucao': ações listadas no item 4.1 (fabricar, instalar, realizar...).\n"
            "- 'notas_e_restricoes': apenas as linhas sob 'NOTAS:' no item 4.1.\n"
            "- 'servicos_simultaneos': serviços listados após 'Quais?' no item 4.10.\n"
            "- NÃO use dados do Memorial Descritivo. Apenas do FORMULÁRIO abaixo.\n"
            "- Retorne APENAS o JSON. Sem texto antes ou depois.\n\n"
            "<FORMULARIO>\n{texto_formulario}\n</FORMULARIO>\n\n"
            "JSON preenchido:\n{skeleton}"
        )
        return PromptTemplate(
            template=template,
            input_variables=["texto_formulario"],
            partial_variables={"skeleton": SKELETON_CSV},
        )

    def _build_prompt_md(self) -> PromptTemplate:
        template = (
            "Você é um extrator de dados de engenharia. Preencha o JSON abaixo com dados do MEMORIAL DESCRITIVO.\n"
            "REGRAS:\n"
            "- Substitua '???' pelo valor real. Se não existir, use null (string) ou [] (lista).\n"
            "- 'normas_petrobras_aplicaveis': APENAS códigos N-XXX citados no texto (ex: N-279, N-115, N-858).\n"
            "- 'detalhamento_por_disciplina': crie UM objeto por sistema/disciplina listado no ESCOPO DE TUBULAÇÃO.\n"
            "  Os sistemas são: GÁS DE EXPORTAÇÃO, GÁS DE IMPORTAÇÃO, GÁS LIFT, GÁS PARA PILOTO DO FLARE,\n"
            "  GÁS COMBUSTÍVEL DE ALTA PRESSÃO, GÁS COMBUSTÍVEL DE BAIXA PRESSÃO, GÁS DO SEPARADOR DE TESTES.\n"
            "  Para cada sistema extraia:\n"
            "    - 'tarefas': frases de fabricação/montagem daquele sistema.\n"
            "    - 'documentos_referencia': apenas isométricos IS-... citados naquele sistema.\n"
            "    - 'tags_relacionados': TIE-INs e códigos de linha (ex: TIE-IN-003, 3/4\"-P-G2-782).\n"
            "- NÃO use dados do formulário CSV. Apenas do MEMORIAL abaixo.\n"
            "- Retorne APENAS o JSON. Sem texto antes ou depois.\n\n"
            "<MEMORIAL_DESCRITIVO>\n{texto_md}\n</MEMORIAL_DESCRITIVO>\n\n"
            "JSON preenchido:\n{skeleton}"
        )
        return PromptTemplate(
            template=template,
            input_variables=["texto_md"],
            partial_variables={"skeleton": SKELETON_MD},
        )

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
        """
        Extrai texto de qualquer arquivo suportado.
        PDFs → usa PyMuPDF (fitz) para extração precisa do texto embutido.
        Demais extensões (.csv, .txt, .md) → usa _read_file() com detecção de encoding.
        """
        if file_path.suffix.lower() == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(file_path))
                pages_text = []
                for page in doc:
                    pages_text.append(page.get_text())
                doc.close()
                text = "\n".join(pages_text)
                logger.info(f"  PDF extraído via PyMuPDF: {len(text)} caracteres de {len(pages_text)} páginas.")
                return text
            except ImportError:
                logger.error("PyMuPDF (fitz) não instalado. Execute: pip install pymupdf>=1.24.0")
                raise
            except Exception as e:
                logger.error(f"Falha ao extrair PDF '{file_path.name}': {e}")
                raise
        return self._read_file(file_path)

    def _call_llm_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Chama Gemini API com system + user prompt. Retorna texto da resposta."""
        from google.genai import types as _gtypes
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.gemini.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=full_prompt,
            config=_gtypes.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=0.1,
            ),
        )
        return response.text

    def _call_llm_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Chama Claude API com system + user prompt. Retorna texto da resposta."""
        response = self.claude.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Chama o LLM disponível: Gemini > Claude > Llama3."""
        if self.use_gemini and self.gemini:
            return self._call_llm_gemini(system_prompt, user_prompt)
        elif self.use_claude and self.claude:
            return self._call_llm_claude(system_prompt, user_prompt)
        else:
            # Llama3 via LangChain
            from langchain_core.prompts import PromptTemplate
            tmpl = "{system}\n\n{user}"
            pt = PromptTemplate(template=tmpl, input_variables=["system", "user"])
            return (pt | self.llm).invoke({"system": system_prompt, "user": user_prompt})

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
        json_str = re.sub(r'"(\?\?\?)"', 'null', json_str)   # sentinelas não substituídos
        json_str = re.sub(r',\s*"?\.\.\."?', '', json_str)   # anti-lazy-generation

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[{label}] JSON inválido: {e}")
            return {}

    # ── Pós-processamento ──────────────────────────────────────────────────────

    def _sanitize(self, data: dict) -> dict:
        """Remove entradas sentinela e normaliza tipos."""
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
                # modelo às vezes retorna true/false em vez de string
                out[k] = "Sim" if v else "Não"
            else:
                out[k] = v
        return out

    # ── Pass 3: IEIS ───────────────────────────────────────────────────────────

    def _process_ieis(self, ieis_path: str) -> dict:
        """
        Pass 3: Extrai especificações de soldagem e materiais base do IEIS.
        Retorna dict com material_base_tubo, eletrodo, processo, NDT, etc.
        """
        ieis_file = Path(ieis_path)
        if not ieis_file.exists():
            logger.warning(f"IEIS não encontrado: {ieis_path}")
            return {}

        texto = self._clean_md_text(self._extract_text_from_file(ieis_file))
        llm_label = "Gemini" if self.use_gemini else ("Claude" if self.use_claude else "Llama3")
        logger.info(f"Pass 3/4 — Extraindo especificações do IEIS via {llm_label}...")

        try:
            if self.use_gemini or self.use_claude:
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
            else:
                from langchain_core.prompts import PromptTemplate
                tmpl = (
                    "Você é engenheiro de soldagem. Extraia dados do IEIS abaixo.\n"
                    "Retorne APENAS o JSON preenchido. Sem texto adicional.\n\n"
                    "<IEIS>\n{texto}\n</IEIS>\n\nJSON preenchido:\n{skeleton}"
                )
                prompt = PromptTemplate(
                    template=tmpl, input_variables=["texto"],
                    partial_variables={"skeleton": SKELETON_IEIS}
                )
                raw = (prompt | self.llm).invoke({"texto": texto[:6000]})

            data = self._sanitize(self._extract_json_robust(raw, "IEIS"))
            logger.info(f"Pass 3 concluído. Eletrodo: {data.get('metal_adicao')} | "
                        f"NDT: {data.get('ndt_requerido')}")
            return data
        except Exception as e:
            logger.error(f"Falha no Pass 3 (IEIS): {e}", exc_info=True)
            return {}

    # ── Pass 4: EBP ────────────────────────────────────────────────────────────

    def _process_ebp(self, ebp_path: str) -> dict:
        """
        Pass 4: Extrai spool list, isométricos e referências de piping class do EBP.
        Retorna dict com spool_list (com comprimentos reais), isométricos, normas_projeto.
        """
        ebp_file = Path(ebp_path)
        if not ebp_file.exists():
            logger.warning(f"EBP não encontrado: {ebp_path}")
            return {}

        texto = self._clean_md_text(self._extract_text_from_file(ebp_file))
        llm_label = "Gemini" if self.use_gemini else ("Claude" if self.use_claude else "Llama3")
        logger.info(f"Pass 4/4 — Extraindo spool list do EBP via {llm_label}...")

        try:
            if self.use_gemini or self.use_claude:
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
                user_p4 = f"<EBP>\n{texto[:20000]}\n</EBP>\n\nJSON preenchido:\n{SKELETON_EBP}"
                raw = self._call_llm(system_p4, user_p4)
            else:
                from langchain_core.prompts import PromptTemplate
                tmpl = (
                    "Você é engenheiro de tubulação. Extraia dados do EBP abaixo.\n"
                    "Retorne APENAS o JSON preenchido. Sem texto adicional.\n\n"
                    "<EBP>\n{texto}\n</EBP>\n\nJSON preenchido:\n{skeleton}"
                )
                prompt = PromptTemplate(
                    template=tmpl, input_variables=["texto"],
                    partial_variables={"skeleton": SKELETON_EBP}
                )
                raw = (prompt | self.llm).invoke({"texto": texto[:6000]})

            data = self._sanitize(self._extract_json_robust(raw, "EBP"))
            n_spools = len(data.get("spool_list") or [])
            logger.info(f"Pass 4 concluído. Spools encontrados: {n_spools} | "
                        f"Piping spec: {data.get('piping_class_referencia')}")
            return data
        except Exception as e:
            logger.error(f"Falha no Pass 4 (EBP): {e}", exc_info=True)
            return {}

    # ── Orquestrador: Two-Pass (+ IEIS + EBP opcionais) ───────────────────────

    def process_project_files(self, csv_path: str, md_path: str,
                               ieis_path: str | None = None,
                               ebp_path: str | None = None) -> dict:
        csv_file, md_file = Path(csv_path), Path(md_path)

        if not csv_file.exists():
            logger.error(f"CSV não encontrado: {csv_file}"); return {}
        if not md_file.exists():
            logger.error(f"MD não encontrado: {md_file}"); return {}

        # Extração de texto — PDF via fitz, demais via read_file
        cleaned_csv = self._clean_csv_text(self._extract_text_from_file(csv_file))
        cleaned_md  = self._clean_md_text(self._extract_text_from_file(md_file))

        llm_label = "Gemini" if self.use_gemini else ("Claude" if self.use_claude else "Llama3")

        # ── PASS 1: extração do formulário ─────────────────────────────────────
        logger.info(f"Pass 1/4 — Extraindo dados do Formulário CSV via {llm_label}...")
        try:
            if self.use_gemini or self.use_claude:
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
                user_p1 = f"<FORMULARIO>\n{cleaned_csv}\n</FORMULARIO>\n\nJSON preenchido:\n{SKELETON_CSV}"
                raw_csv = self._call_llm(system_p1, user_p1)
            else:
                raw_csv = (self.prompt_csv | self.llm).invoke({"texto_formulario": cleaned_csv})
            data_csv = self._sanitize(self._extract_json_robust(raw_csv, "CSV"))
            logger.info(f"Pass 1 concluído. Campos extraídos: {list(data_csv.keys())}")
        except Exception as e:
            logger.error(f"Falha no Pass 1: {e}", exc_info=True)
            data_csv = {}

        # ── PASS 2: extração do Memorial Descritivo ────────────────────────────
        logger.info(f"Pass 2/4 — Extraindo dados do Memorial Descritivo via {llm_label}...")
        try:
            if self.use_gemini or self.use_claude:
                system_p2 = (
                    "Você é um extrator de dados de engenharia offshore Petrobras.\n"
                    "REGRAS:\n"
                    "- Substitua '???' pelo valor real. Se não existir, use null ou [].\n"
                    "- 'normas_petrobras_aplicaveis': APENAS códigos N-XXX citados no texto (ex: N-279, N-115).\n"
                    "  Se não houver normas explícitas no texto, retorne [].\n"
                    "- 'detalhamento_por_disciplina': um objeto por sistema/disciplina no ESCOPO.\n"
                    "- Retorne APENAS JSON válido. Sem texto antes ou depois."
                )
                user_p2 = f"<MEMORIAL_DESCRITIVO>\n{cleaned_md}\n</MEMORIAL_DESCRITIVO>\n\nJSON preenchido:\n{SKELETON_MD}"
                raw_md = self._call_llm(system_p2, user_p2)
            else:
                raw_md = (self.prompt_md | self.llm).invoke({"texto_md": cleaned_md})
            data_md = self._sanitize(self._extract_json_robust(raw_md, "MD"))
            logger.info(f"Pass 2 concluído. Normas: {data_md.get('normas_petrobras_aplicaveis', [])}")
        except Exception as e:
            logger.error(f"Falha no Pass 2: {e}", exc_info=True)
            data_md = {}

        # ── Regex: sempre extrai N-XXXX do texto e merge com resultado LLM ──
        # O LLM pode extrair normas ABNT/NBR que não existem no Qdrant — filtramos para manter
        # apenas normas Petrobras no formato N-XXXX (ex: N-0115, N-133) encontradas no texto.
        RE_PETROBRAS_STRICT = re.compile(r'\bN-(\d{3,4})\b')
        normas_llm_raw = data_md.get("normas_petrobras_aplicaveis") or []
        # Filtra LLM: mantém apenas strings que combinam com padrão N-XXXX Petrobras, normaliza
        def _normalize_norma(n: str) -> str:
            m = RE_PETROBRAS_STRICT.match(n)
            return f"N-{m.group(1).zfill(4)}" if m else n
        normas_llm_filtradas = [_normalize_norma(n) for n in normas_llm_raw if RE_PETROBRAS_STRICT.match(n)]
        if len(normas_llm_filtradas) < len(normas_llm_raw):
            descartadas = set(normas_llm_raw) - set(normas_llm_filtradas)
            logger.info(f"  Normas descartadas (não-Petrobras): {sorted(descartadas)}")
        # Regex direto no texto limpo para capturar N-XXXX (normaliza 3 dígitos para 4)
        normas_regex = [f"N-{m.zfill(4)}" for m in RE_NORMA_PETROBRAS.findall(cleaned_md)]
        normas_regex_dedup = list(dict.fromkeys(normas_regex))
        # Merge: LLM filtradas + regex, sem duplicatas
        normas_merged = list(dict.fromkeys(normas_llm_filtradas + normas_regex_dedup))
        if normas_merged != normas_llm_raw:
            logger.info(f"  Normas finais (regex+LLM): {normas_merged}")
        data_md["normas_petrobras_aplicaveis"] = normas_merged

        # ── Regex N-XXXX também no texto do CSV (documentos_referencia pode citar normas) ──
        normas_regex_csv = [f"N-{m.zfill(4)}" for m in RE_NORMA_PETROBRAS.findall(cleaned_csv)]
        normas_merged_final = list(dict.fromkeys(normas_merged + normas_regex_csv))
        if normas_merged_final != normas_merged:
            logger.info(f"  Normas adicionais do CSV: {list(set(normas_merged_final) - set(normas_merged))}")
        data_md["normas_petrobras_aplicaveis"] = normas_merged_final

        # ── PASS 3: IEIS (opcional) ────────────────────────────────────────────
        data_ieis = {}
        if ieis_path:
            data_ieis = self._process_ieis(ieis_path)
            # Merge normas do IEIS
            normas_ieis = [f"N-{m.zfill(4)}" for m in RE_NORMA_PETROBRAS.findall(
                self._clean_md_text(self._extract_text_from_file(Path(ieis_path)))
            )]
            normas_merged_final_prev = data_md.get("normas_petrobras_aplicaveis") or []
            normas_merged_final_prev = list(dict.fromkeys(normas_merged_final_prev + normas_ieis))
            data_md["normas_petrobras_aplicaveis"] = normas_merged_final_prev

        # ── PASS 4: EBP (opcional) ─────────────────────────────────────────────
        data_ebp = {}
        if ebp_path:
            data_ebp = self._process_ebp(ebp_path)
            # Merge normas do EBP
            normas_projeto = data_ebp.get("normas_projeto") or []
            normas_n = [f"N-{m.zfill(4)}" for m in RE_NORMA_PETROBRAS.findall(
                self._clean_md_text(self._extract_text_from_file(Path(ebp_path)))
            )]
            normas_prev = data_md.get("normas_petrobras_aplicaveis") or []
            data_md["normas_petrobras_aplicaveis"] = list(dict.fromkeys(normas_prev + normas_n))

        # ── MERGE: une os dicionários ───────────────────────────────────────────
        resultado = {**data_csv, **data_md}
        if data_ieis:
            resultado["especificacoes_soldagem"] = data_ieis
        if data_ebp:
            resultado["spool_list"] = data_ebp.get("spool_list") or []
            resultado["isometricos_referenciados"] = data_ebp.get("isometricos_referenciados") or []
            resultado["piping_class_referencia"] = data_ebp.get("piping_class_referencia")
            # Merge normas_projeto do EBP como documentos de referência
            docs_ref = resultado.get("documentos_referencia") or []
            for np in (data_ebp.get("normas_projeto") or []):
                if np and np not in docs_ref:
                    docs_ref.append(np)
            resultado["documentos_referencia"] = docs_ref

        # ── Fallback: id_servico inválido quando LLM confunde com tag da linha ──
        tag = resultado.get("tag_linha_principal") or ""
        id_svc = resultado.get("id_servico") or ""
        if not id_svc or id_svc == tag or id_svc in ("???", "null", "LP-XXX"):
            # Tenta extrair de titulo_servico (ex: "SS-93", "LC-029")
            RE_SERVICE_ID = re.compile(r'\b([A-Z]{1,3}-\d{1,5})\b')
            titulo = resultado.get("titulo_servico") or ""
            m_svc = RE_SERVICE_ID.search(titulo)
            if m_svc and m_svc.group(1) != tag:
                resultado["id_servico"] = m_svc.group(1)
                logger.info(f"  id_servico corrigido via titulo_servico: {resultado['id_servico']}")
            else:
                # Usa o stem do arquivo CSV como fallback final
                resultado["id_servico"] = csv_file.stem
                logger.info(f"  id_servico fallback para nome do arquivo: {resultado['id_servico']}")

        # ── Fallback: ativo/tag_equipamento — LLM às vezes alucina "P-37 | 2027" ──
        # Padrão "X | AAAA" indica que o LLM confundiu data ou campo de outra linha
        ativo_val = resultado.get("ativo") or ""
        if RE_DATA_PLACEHOLDER.search(ativo_val):
            logger.info(f"  ativo corrigido (placeholder detectado): '{ativo_val}' → None")
            resultado["ativo"] = None

        tag_eq_val = resultado.get("tag_equipamento_principal") or ""
        if RE_DATA_PLACEHOLDER.search(tag_eq_val):
            logger.info(f"  tag_equipamento_principal corrigido (placeholder detectado): '{tag_eq_val}' → None")
            resultado["tag_equipamento_principal"] = None

        # ── Fallback: plataforma — Llama3 frequentemente alucina plataformas diferentes ──
        plataforma_llm = resultado.get("plataforma") or ""
        if not RE_PLATAFORMA.match(plataforma_llm.strip()):
            # Busca plataforma diretamente no texto dos dois documentos
            for texto_fonte in [cleaned_csv, cleaned_md]:
                m = RE_PLATAFORMA.search(texto_fonte)
                if m:
                    resultado["plataforma"] = m.group(1)
                    logger.info(f"  plataforma corrigida via regex: {resultado['plataforma']}")
                    break

        # ── Fallback: tag_linha_principal — Llama3 usa placeholder "LP-XXX" ──
        tag_llm = resultado.get("tag_linha_principal") or ""
        if not tag_llm or tag_llm in ("LP-XXX", "???", "null") or not RE_TAG_LINHA.match(tag_llm.strip()):
            for texto_fonte in [cleaned_csv, cleaned_md]:
                m = RE_TAG_LINHA.search(texto_fonte)
                if m:
                    resultado["tag_linha_principal"] = m.group(1)
                    logger.info(f"  tag_linha_principal corrigida via regex: {resultado['tag_linha_principal']}")
                    break

        # ── Fallback: numero_ze / LP — captura LP-NNN do texto ──
        ze_llm = resultado.get("numero_ze") or ""
        if not ze_llm or ze_llm in ("LP-XXX", "???", "null"):
            for texto_fonte in [cleaned_csv, cleaned_md]:
                m = RE_LP.search(texto_fonte)
                if m:
                    resultado["numero_ze"] = f"LP-{m.group(1)}"
                    logger.info(f"  numero_ze corrigido via regex: {resultado['numero_ze']}")
                    break

        logger.info("✓ Merge concluído. Triagem finalizada.")
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