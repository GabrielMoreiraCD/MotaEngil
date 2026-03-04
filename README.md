# Delineador Técnico Autônomo (DTA)
> Pipeline de IA para estimativa de quantitativos, BOM e homem-hora em projetos de manutenção offshore Petrobras.

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Problema de Negócio](#2-problema-de-negócio)
3. [Arquitetura do Sistema](#3-arquitetura-do-sistema)
4. [Stack Tecnológica](#4-stack-tecnológica)
5. [Estrutura do Repositório](#5-estrutura-do-repositório)
6. [Componentes Detalhados](#6-componentes-detalhados)
7. [Fluxo de Dados End-to-End](#7-fluxo-de-dados-end-to-end)
8. [Esquemas de Dados](#8-esquemas-de-dados)
9. [Configuração e Instalação](#9-configuração-e-instalação)
10. [Estado Atual do Desenvolvimento](#10-estado-atual-do-desenvolvimento)
11. [Lacunas Conhecidas e Roadmap](#11-lacunas-conhecidas-e-roadmap)
12. [Decisões Arquiteturais](#12-decisões-arquiteturais)
13. [Diretrizes de Engenharia](#13-diretrizes-de-engenharia)

---

## 1. Visão Geral

O **Delineador Técnico Autônomo (DTA)** é um pipeline de IA multimodal que automatiza a fase de delineamento técnico em projetos de manutenção offshore da Petrobras — especificamente adequação de tubulações, TIE-INs e suportes estruturais.

O sistema recebe como entrada documentos não padronizados (escopo em XLSX e Memorial Descritivo em PDF), cruza essas informações com um banco de dados vetorial de normas técnicas (N-115, N-279, etc.) e catálogos teóricos, e produz como saída:

- **BOM (Bill of Materials):** lista de materiais com quantitativos por isométrico
- **HH Estimado:** estimativa de homem-hora por tarefa e disciplina
- **Documentação estruturada:** JSON/XLSX padronizado para uso nos sistemas de planejamento

---

## 2. Problema de Negócio

### Contexto Operacional

Projetos de manutenção offshore envolvem a análise cruzada de:

| Entrada | Formato | Problema Típico |
|---|---|---|
| Escopo de Serviço | XLSX exportado | Formatação suja, checkboxes, codificação ambígua |
| Memorial Descritivo | PDF técnico | Layout variável, tabelas, isométricos embarcados |
| Normas Técnicas | PDF (N-115, N-279...) | Volume > 500 páginas, cabeçalhos/rodapés ruidosos |
| Isométricos | PDF / imagem | Cotas, tags, listas de materiais visuais |

### Processo Manual Atual

```
Engenheiro recebe XLSX + PDF
        ↓
Leitura manual do escopo (2–4h)
        ↓
Consulta manual às normas (4–8h)
        ↓
Leitura dos isométricos (3–6h por isométrico)
        ↓
Preenchimento manual do BOM (2–4h)
        ↓
Estimativa de HH por experiência tácita (1–3h)
```

**Tempo total estimado por projeto:** 12–25 horas/engenheiro.

### Objetivo do Sistema

Reduzir o tempo de delineamento técnico para **< 30 minutos** por projeto, mantendo rastreabilidade normativa completa e auditabilidade das fontes utilizadas.

---

## 3. Arquitetura do Sistema

### Visão de Alto Nível

```
┌────────────────────────────────────────────────────────────────┐
│                        ENTRADAS                                │
│   [XLSX - Escopo de Serviço]    [PDF - Memorial Descritivo]    │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│              TRIAGE AGENT  (LangGraph Node 1)                  │
│  Modelo: Llama 3 8B via Ollama                                 │
│  Saída:  JSON estruturado — Sistemas, Tarefas, Isométricos,    │
│          Normas citadas                                         │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│              RAG ENGINE  (LangGraph Node 2)                    │
│  Vector Store: Qdrant Cloud                                    │
│  Embeddings:   BAAI/bge-m3 (1024 dim, HNSW)                   │
│  Filtro:       Metadata filter por norma/disciplina            │
│  Saída:        Chunks normativos rankeados por relevância      │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│              VISION ENGINE  (LangGraph Node 3)                 │
│  Modelo: Qwen2.5-VL-72B via HuggingFace Inference             │
│  Input:  Imagens de isométricos extraídas via PyMuPDF          │
│  Saída:  BOM parcial por isométrico (JSON)                     │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│              SYNTHESIS AGENT  (LangGraph Node 4)               │
│  Consolida BOM + contexto normativo + HH estimado              │
│  Saída:  XLSX / JSON final                                     │
└────────────────────────────────────────────────────────────────┘
```

### Grafo LangGraph

```
START
  │
  ▼
triage_node ──→ rag_node ──→ vision_node ──→ synthesis_node ──→ END
                    ▲               │
                    └───────────────┘
                  (loop: isométricos pendentes)
```

O estado compartilhado (`TypedDict`) trafega pelo grafo carregando:
- JSON da triagem validado por Pydantic
- Chunks RAG recuperados com scores
- BOM parcial acumulado
- Lista de isométricos pendentes de análise visual

---

## 4. Stack Tecnológica

| Camada | Tecnologia | Versão | Função |
|---|---|---|---|
| Linguagem | Python | 3.10+ | Runtime principal |
| Orquestração | LangGraph / LangChain | latest | Controle de fluxo agentivo |
| Vector Store | Qdrant Cloud | — | Armazenamento e busca de embeddings |
| Embeddings | BAAI/bge-m3 | — | Vetorização de textos PT/EN técnicos |
| LLM Local | Llama 3 8B (Ollama) | — | Triagem, roteamento, extração estruturada |
| VLM | Qwen2.5-VL-72B (HF Inference) | — | Leitura de isométricos e tabelas visuais |
| Extração PDF | PyMuPDF (fitz) | — | Extração com metadados espaciais |
| OCR | Pytesseract | — | Fallback para PDFs escaneados |
| Validação | Pydantic | v2 | Schemas dinâmicos agnósticos de disciplina |

---

## 5. Estrutura do Repositório

```
dta/
├── core/
│   ├── extractors.py          # Extração de texto/imagem de PDFs com filtro espacial
│   └── chunkers.py            # Agregação semântica de blocos para indexação
│
├── services/
│   └── triage_agent.py        # Agente de triagem (XLSX + PDF → JSON estruturado)
│
├── rag_engine.py              # Recuperação vetorial com metadata filters (Qdrant)
├── vision_engine.py           # Análise de isométricos via VLM (Qwen2.5-VL-72B)
├── synthesis_agent.py         # Consolidação de BOM + HH [EM DESENVOLVIMENTO]
│
├── ingest_standards.py        # Pipeline de ingestão de normas para Qdrant
│
├── graph.py                   # Definição do grafo LangGraph
├── state.py                   # TypedDict do estado compartilhado
│
├── schemas/
│   ├── triage_schema.py       # Modelos Pydantic: Projeto, Sistema, Tarefa, BOM
│   └── bom_schema.py          # Schema do Bill of Materials final
│
├── utils/
│   ├── pdf_utils.py           # Helpers PyMuPDF (filtro Y, extração de imagens)
│   ├── csv_utils.py           # Higienização de XLSX/CSV com regex
│   └── json_utils.py          # Extrator regex destrutivo para JSON de LLMs
│
├── config.py                  # Variáveis de ambiente e parâmetros globais
├── requirements.txt
└── README.md
```

---

## 6. Componentes Detalhados

### 6.1 Ingestão de Normas (`ingest_standards.py` + `core/`)

Popula o Qdrant com o conhecimento das normas técnicas Petrobras.

**Filtro espacial (cabeçalho/rodapé):**
```python
# Descarta blocos fora da área útil da página
# Valores calibrados para layout padrão de normas Petrobras
HEADER_THRESHOLD_PX = 70    # y0 < 70 → cabeçalho → descartado
FOOTER_THRESHOLD_PX = 60    # y1 > (page_height - 60) → rodapé → descartado
```

**Associação texto-imagem:**
A vinculação de imagens vetorizadas ao seu contexto textual é feita por distância euclidiana mínima entre o centróide da imagem e o centróide dos blocos de texto da mesma página:

```
d = sqrt((cx_img - cx_txt)² + (cy_img - cy_txt)²)
```

O `bounding_box` da imagem é salvo no payload do Qdrant junto ao chunk de texto mais próximo, permitindo recuperação multimodal.

**Chunking semântico (`core/chunkers.py`):**
- Blocos consecutivos são fundidos enquanto `tokens_acumulados < 400`
- Quebra forçada em mudança de seção: regex `^\d+\.\d+` (numeração de norma)
- Cada chunk carrega: `{norma, secao, pagina, tem_imagem: bool, bbox_imagem}`

---

### 6.2 Agente de Triagem (`services/triage_agent.py`)

Transforma entradas caóticas em JSON estruturado e validado.

**Tratamento de encoding:**
```python
# Alternância automática de encoding
try:
    df = pd.read_csv(path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(path, encoding='cp1252')
```

**Higienização de checkboxes:**
```python
# Regex para checkboxes exportados do Excel
# ☑, ☒, X, Sim/Não → tokens normalizados
CHECKBOX_PATTERN = re.compile(r'[☑✓X]\s*;?\s*(?:Sim|Não)?', re.IGNORECASE)
```

**Estratégia de prompt para Llama 3 8B:**

Dois princípios críticos aplicados:
1. **Delimitadores XML** para isolar o contexto da instrução
2. **Template JSON no final absoluto** para explorar o viés de recência do modelo

```
<ESCOPO>{conteúdo do XLSX higienizado}</ESCOPO>
<MEMORIAL>{conteúdo do PDF}</MEMORIAL>
<INSTRUCAO>Extraia a topologia do projeto...</INSTRUCAO>

{  ← template JSON posicionado no final
  "sistemas": [],
  "tarefas": [],
  ...
}
```

**Extração robusta de JSON:**
```python
# Nunca confiar em JsonOutputParser para modelos < 13B
# Pipeline regex destrutivo:
# 1. Strip de blocos markdown (```json ... ```)
# 2. Captura do primeiro {...} com re.DOTALL
# 3. Remoção de reticências de "lazy generation" (LLM preguiçoso)
# 4. json.loads() com fallback para reparo estrutural
```

---

### 6.3 RAG Engine (`rag_engine.py`)

Recupera trechos normativos relevantes para cada tarefa identificada.

**Roteamento condicional por metadata:**
```python
# Evita cross-contamination entre normas de disciplinas distintas
# Ex: N-115 (tubulação) não contamina busca de N-279 (suportes)
filters = Filter(
    must=[FieldCondition(key="norma", match=MatchValue(value=norma_alvo))]
)
results = client.search(collection_name, query_vector, query_filter=filters, limit=5)
```

**Enfileiramento para visão:**
Chunks recuperados com `tem_imagem: True` no payload são automaticamente adicionados à fila do Vision Engine.

---

### 6.4 Vision Engine (`vision_engine.py`)

Extrai BOM de isométricos usando VLM de 72B parâmetros.

**Por que Qwen2.5-VL-72B:**
Isométricos técnicos contêm cotas sobrepostas, tags alfanuméricas (`6"-LI-1234-A1A`), tabelas de materiais e símbolos normalizados. Modelos < 30B apresentam taxa de falha > 40% na extração de tags e quantitativos nesses layouts.

**Saída esperada por isométrico:**
```json
{
  "isometrico": "ISO-001",
  "linha": "6\"-LI-1234-A1A",
  "materiais": [
    {
      "descricao": "Joelho 90° LR",
      "norma_dimensional": "ASME B16.9",
      "diametro_nominal": "6\"",
      "quantidade": 2,
      "unidade": "UN"
    }
  ]
}
```

---

### 6.5 Synthesis Agent (`synthesis_agent.py`) — EM DESENVOLVIMENTO

Consolida todos os BOMs parciais, aplica as regras das normas recuperadas e calcula HH estimado.

A estimativa de HH requer o mapeamento:
```
Atividade (ex: "soldagem topo DN150") → Tabela de Produtividade (N-115 Anexo X) → HH/junta
```

Este módulo depende da implementação do parser de tabelas de produtividade das normas.

---

## 7. Fluxo de Dados End-to-End

```
1. INPUT
   ├── escopo.xlsx  →  csv_utils.higienizar()  →  DataFrame limpo
   └── memorial.pdf →  pdf_utils.extrair()     →  texto + imagens com bbox

2. TRIAGE NODE
   ├── Prompt XML + template JSON → Llama 3 8B (Ollama)
   ├── json_utils.extrair_json()  → JSON bruto do LLM
   └── Pydantic.validate()        → ProjetoSchema validado

3. RAG NODE
   ├── Para cada (tarefa, norma_citada):
   │   ├── bge-m3.encode(tarefa.descricao) → vetor 1024-dim
   │   ├── qdrant.search(filtro=norma)     → top-5 chunks
   │   └── chunks com tem_imagem=True → fila_visao
   └── contexto_normativo acumulado

4. VISION NODE
   ├── Para cada isométrico em fila_visao:
   │   ├── pdf_utils.extrair_imagem(bbox)  → bytes da imagem
   │   ├── hf_client.chat(qwen_vl, imagem) → JSON de materiais
   │   └── Pydantic.validate()             → BOMParcial validado
   └── bom_acumulado = merge(todos BOMParcial)

5. SYNTHESIS NODE  [EM DESENVOLVIMENTO]
   ├── bom_final = deduplicar(bom_acumulado)
   ├── hh_estimado = calcular_hh(bom_final, contexto_normativo)
   └── exportar(bom_final, hh_estimado) → output.xlsx / output.json

6. OUTPUT
   └── {projeto, bom[], hh_estimado{total, breakdown{}}}
```

---

## 8. Esquemas de Dados

### Schema de Triagem (Pydantic)

```python
class Tarefa(BaseModel):
    id: str
    descricao: str
    sistema: str
    isometricos: List[str]
    normas_aplicaveis: List[str]
    disciplina: Optional[str] = None

class ProjetoSchema(BaseModel):
    nome_projeto: str
    sistemas: List[str]
    tarefas: List[Tarefa]
    normas_citadas: List[str]
```

### Schema do BOM Final (JSON)

```json
{
  "projeto": "TIE-IN L-1234",
  "bom": [
    {
      "isometrico": "ISO-001",
      "material": "Joelho 90° ASME B16.9",
      "diametro_nominal": "4\"",
      "quantidade": 3,
      "unidade": "UN",
      "norma": "N-279"
    }
  ],
  "hh_estimado": {
    "total": 240,
    "breakdown": {
      "soldagem": 120,
      "montagem": 80,
      "inspecao": 40
    }
  }
}
```

### Payload de Chunk no Qdrant

```json
{
  "id": "uuid-v4",
  "vector": [1024 floats],
  "payload": {
    "norma": "N-115",
    "secao": "5.3.2",
    "pagina": 47,
    "texto": "...",
    "tem_imagem": true,
    "bbox_imagem": [x0, y0, x1, y1]
  }
}
```

---

## 9. Configuração e Instalação

### Pré-requisitos

```bash
python --version   # 3.10+
ollama --version   # com modelo llama3:8b baixado
```

### Instalação

```bash
git clone <repo>
cd dta
pip install -r requirements.txt
```

### Variáveis de Ambiente

```bash
# .env
QDRANT_URL=https://<seu-cluster>.qdrant.io
QDRANT_API_KEY=<sua-chave>
HF_TOKEN=<token-huggingface>       # Para Qwen2.5-VL-72B Inference
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b
EMBED_MODEL=BAAI/bge-m3
QDRANT_COLLECTION=normas_petrobras
```

### Ingestão Inicial de Normas

```bash
# Processar e indexar normas técnicas no Qdrant
python ingest_standards.py --input_dir ./normas/ --collection normas_petrobras
```

### Execução do Pipeline

```bash
python main.py --escopo escopo.xlsx --memorial memorial.pdf
```

---

## 10. Estado Atual do Desenvolvimento

| Componente | Status | Observações |
|---|---|---|
| `ingest_standards.py` | ✅ Funcional | Calibrado para layout padrão de normas Petrobras |
| `core/extractors.py` | ✅ Funcional | Filtro espacial Y validado |
| `core/chunkers.py` | ✅ Funcional | Threshold de 400 tokens empírico |
| `services/triage_agent.py` | ✅ Funcional | Sensível a variações de layout de XLSX |
| `rag_engine.py` | ✅ Funcional | Precision@5 não mensurada formalmente |
| `vision_engine.py` | ✅ Integrado | Qualidade dependente da resolução dos isométricos |
| `synthesis_agent.py` | 🔴 Não implementado | Bloqueado por: parser de tabelas de produtividade |
| `graph.py` (LangGraph) | 🟡 Parcial | Nós 1–3 conectados; nó 4 stub |
| Testes unitários | 🔴 Ausentes | Sem cobertura de testes automatizados |
| Métricas RAG | 🔴 Ausentes | Precision@K, MRR não implementados |

---

## 11. Lacunas Conhecidas e Roadmap

### Lacunas Críticas (Bloqueantes)

**1. Synthesis Agent (`synthesis_agent.py`)**
Requer implementação do parser de tabelas de produtividade das normas (ex: N-115 Anexo A). O mapeamento `atividade → HH/unidade` ainda é feito manualmente.

**2. Avaliação do RAG**
Sem ground truth para medir Precision@K e MRR. Necessário construir dataset de avaliação com pares (consulta, trecho_normativo_correto).

**3. Testes Automatizados**
Nenhuma cobertura de testes unitários ou de integração. Regressões são detectadas manualmente.

### Roadmap

```
FASE 1 — Completar pipeline base
├── [ ] Implementar synthesis_agent.py
├── [ ] Parser de tabelas de produtividade (PyMuPDF + heurística estrutural)
└── [ ] Conectar nó 4 no grafo LangGraph

FASE 2 — Qualidade e Avaliação
├── [ ] Dataset de avaliação RAG (50 pares mínimos)
├── [ ] Métricas: Precision@5, MRR, latência por nó
└── [ ] Testes unitários (pytest) para extractors, chunkers, json_utils

FASE 3 — Robustez e Escala
├── [ ] Suporte a isométricos escaneados (OCR via Pytesseract como fallback)
├── [ ] Interface de revisão humana para BOM gerado
└── [ ] Exportação direta para formato XLSX Petrobras
```

---

## 12. Decisões Arquiteturais

| Decisão | Alternativa Considerada | Motivo da Escolha |
|---|---|---|
| Llama 3 8B apenas para triagem | GPT-4 / Claude para tudo | Tarefa estruturada de baixa complexidade; latência e custo zero com Ollama |
| Qwen2.5-VL-72B para visão | LLaVA 13B local | Isométricos exigem modelo capaz de ler cotas e tags técnicas densas; modelos < 30B falham |
| Qdrant com metadata filters | Chroma / FAISS | Filtro por norma antes do similarity search aumenta precision sem elevar K |
| BAAI/bge-m3 para embeddings | text-embedding-ada-002 | Melhor desempenho em textos técnicos bilíngues PT/EN; sem custo de API |
| LangGraph vs LCEL simples | LangChain LCEL | O loop condicional de visão (isométricos pendentes) exige grafo stateful com memória entre nós |
| PyMuPDF para extração | pdfplumber / pdfminer | Metadados espaciais (bounding box por bloco) são essenciais para filtro de cabeçalho/rodapé |

---

## 13. Diretrizes de Engenharia

### LLMs Locais (≤ 8B parâmetros)

Modelos desta escala sofrem dois problemas recorrentes neste domínio:

- **Lazy Generation:** O modelo insere `"..."` ou `"etc."` no meio de listas JSON ao invés de completar os itens
- **Language Drift:** Troca PT→EN no meio da geração, quebrando chaves do JSON

**Mitigações obrigatórias:**

```python
# 1. Delimitadores XML para isolar contexto
# 2. Template JSON posicionado no FINAL do prompt (viés de recência)
# 3. NUNCA usar JsonOutputParser nativo — implementar extrator regex:

def extrair_json(texto: str) -> dict:
    # Remove blocos markdown
    texto = re.sub(r'```(?:json)?', '', texto)
    # Captura primeiro objeto JSON completo
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if not match:
        raise ValueError("Nenhum JSON encontrado na resposta do LLM")
    json_str = match.group(0)
    # Remove lazy generation artifacts
    json_str = re.sub(r'\.\.\.|…', '', json_str)
    return json.loads(json_str)
```

### Robustez de Ingestão de Dados

```python
# Alternância de encoding para XLSX exportados
try:
    df = pd.read_csv(path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(path, encoding='cp1252')

# Normalização de checkboxes do Excel
CHECKBOX_REGEX = re.compile(r'[☑☒✓✗X]\s*;?\s*(Sim|Não|Yes|No)?', re.IGNORECASE)
df = df.applymap(lambda x: normalizar_checkbox(x) if isinstance(x, str) else x)
```

### Padrão de Tratamento de Erros

Todo módulo deve implementar tratamento de erro explícito sem suprimir exceções silenciosamente:

```python
try:
    resultado = processar(entrada)
except ValueError as e:
    logger.error(f"[{modulo}] Falha na validação: {e}")
    raise
except Exception as e:
    logger.critical(f"[{modulo}] Erro inesperado: {e}", exc_info=True)
    raise
```

---

## Contribuição

Ao modificar qualquer script, forneça **sempre** o bloco completo da função alterada — nunca use `# ... resto do código igual`. Lógicas de tratamento de erro não devem ser abreviadas.

---

*Documento gerado em: 2026-03 | Versão: 0.3.0-alpha*
