"""
rag_metrics.py
==============
Avaliação RAG multi-coleção — Delineador Técnico Autônomo (Petrobras/Mota-Engil).
Implementa: Precision@K, MRR, NDCG@K, Hit Rate@K, ComparisonReport.

Variáveis consumidas do .env:
  QDRANT_URL         → URL do cluster Qdrant Cloud
  QDRANT_API_KEY     → API key do Qdrant Cloud
  COLLECTION_NAMES   → Coleções separadas por vírgula (multi-avaliação)
                       Ex: petrobras_rag_teoria,normas_tecnicas_publicas
  COLLECTION_NAME    → Fallback para uma única coleção (retrocompatível)
  EMBEDDING_MODEL    → sentence-transformers/all-MiniLM-L6-v2
  HF_TOKEN           → Token HuggingFace
  LLM_MODEL          → llama3:latest

Lógica matemática:
  Precision@K = |{rel docs em top-K}| / K
  MRR         = mean(1 / rank_primeiro_relevante) ; 0.0 se nenhum relevante
  NDCG@K      = DCG@K / IDCG@K   onde  DCG@K = Σ rel_i / log2(i+1)
  Hit Rate@K  = 1 se ∃ relevante em top-K, else 0  (média sobre queries)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _resolve_env_file(env_path: str | Path | None) -> Path | None:
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates += [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path.cwd().parent.parent / ".env",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@dataclass(frozen=True)
class Settings:
    """
    Configurações imutáveis lidas do .env.

    Resolução de coleções (em ordem de prioridade):
      1. COLLECTION_NAMES=col_a,col_b   → lista com múltiplas coleções
      2. COLLECTION_NAME=col_a          → lista com uma coleção (retrocompat)
      Se ambas ausentes → EnvironmentError.

    Uso:
        cfg = Settings.from_env()
        for col in cfg.collection_names:   # itera sobre as coleções alvo
            ...
    """
    qdrant_url: str
    qdrant_api_key: str
    collection_names: tuple[str, ...]      # sempre uma tupla, nunca vazia
    embedding_model: str
    hf_token: str
    llm_model: str

    # Propriedade de conveniência para código que usa uma única coleção
    @property
    def collection_name(self) -> str:
        return self.collection_names[0]

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "Settings":
        resolved = _resolve_env_file(env_path)
        if resolved:
            load_dotenv(dotenv_path=resolved, override=False)
            print(f"[Settings] .env carregado: {resolved.resolve()}")
        else:
            print("[Settings] AVISO: .env não encontrado — usando variáveis do SO.")

        qdrant_url     = os.environ.get("QDRANT_URL", "").strip()
        qdrant_api_key = os.environ.get("QDRANT_API_KEY", "").strip()
        embedding_model = os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
        hf_token  = os.environ.get("HF_TOKEN", "").strip()
        llm_model = os.environ.get("LLM_MODEL", "llama3:latest").strip()

        # Resolução de coleções: COLLECTION_NAMES tem prioridade
        raw_names = os.environ.get("COLLECTION_NAMES", "").strip()
        if not raw_names:
            raw_names = os.environ.get("COLLECTION_NAME", "").strip()

        missing = []
        if not qdrant_url:     missing.append("QDRANT_URL")
        if not qdrant_api_key: missing.append("QDRANT_API_KEY")
        if not raw_names:      missing.append("COLLECTION_NAMES (ou COLLECTION_NAME)")
        if missing:
            raise EnvironmentError(
                f"[Settings] Variáveis obrigatórias ausentes no .env: {missing}\n"
                f"Exemplo para multi-coleção:\n"
                f"  COLLECTION_NAMES=petrobras_rag_teoria,normas_tecnicas_publicas"
            )

        # Normaliza: split por vírgula, strip de espaços, remove vazios
        collection_names = tuple(
            n.strip() for n in raw_names.split(",") if n.strip()
        )

        if hf_token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
            os.environ.setdefault("HF_TOKEN", hf_token)

        return cls(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_names=collection_names,
            embedding_model=embedding_model,
            hf_token=hf_token,
            llm_model=llm_model,
        )

    def masked_repr(self) -> str:
        def _mask(s: str) -> str:
            return (s[:6] + "****") if len(s) > 6 else "****"
        return (
            f"Settings(\n"
            f"  QDRANT_URL       = {self.qdrant_url}\n"
            f"  QDRANT_API_KEY   = {_mask(self.qdrant_api_key)}\n"
            f"  COLLECTION_NAMES = {list(self.collection_names)}\n"
            f"  EMBEDDING_MODEL  = {self.embedding_model}\n"
            f"  HF_TOKEN         = {_mask(self.hf_token) if self.hf_token else '(não definido)'}\n"
            f"  LLM_MODEL        = {self.llm_model}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDoc:
    doc_id: str
    score: float
    payload: dict[str, Any]
    is_relevant: bool = False


@dataclass
class QueryResult:
    query_id: str
    query_text: str
    retrieved: list[RetrievedDoc]
    relevant_ids: set[str]

    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    collection_name: str
    num_queries: int
    k_values: list[int]
    embedding_model: str = ""

    mean_precision: dict[int, float] = field(default_factory=dict)
    mean_mrr: float = 0.0
    mean_ndcg: dict[int, float] = field(default_factory=dict)
    mean_hit_rate: dict[int, float] = field(default_factory=dict)

    per_query_results: list[QueryResult] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for qr in d["per_query_results"]:
            qr["relevant_ids"] = list(qr["relevant_ids"])
        return d


@dataclass
class CollectionDelta:
    """
    Diferença absoluta e relativa entre duas coleções para uma métrica/K.
    Δabs = B - A  |  Δrel% = (B - A) / A * 100  (A = coleção de referência)
    """
    metric: str          # ex: "P@5", "MRR", "nDCG@10"
    score_a: float
    score_b: float
    delta_abs: float
    delta_rel_pct: float
    winner: str          # nome da coleção vencedora


@dataclass
class ComparisonReport:
    """
    Comparação side-by-side entre N coleções avaliadas com o mesmo ground truth
    e o mesmo embedding model.

    Estrutura de scores:
        scores[collection_name]["mrr"]     → float
        scores[collection_name]["P@K"]     → float  (K em k_values)
        scores[collection_name]["nDCG@K"]  → float
        scores[collection_name]["HR@K"]    → float
    """
    embedding_model: str
    k_values: list[int]
    num_queries: int
    reports: dict[str, EvaluationReport]           # keyed by collection_name
    deltas: list[CollectionDelta] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "embedding_model": self.embedding_model,
            "k_values":        self.k_values,
            "num_queries":     self.num_queries,
            "timestamp":       self.timestamp,
            "scores": {
                col: {
                    "mrr":        rep.mean_mrr,
                    **{f"precision_at_{k}": rep.mean_precision[k] for k in self.k_values},
                    **{f"ndcg_at_{k}":      rep.mean_ndcg[k]      for k in self.k_values},
                    **{f"hit_rate_at_{k}":  rep.mean_hit_rate[k]  for k in self.k_values},
                }
                for col, rep in self.reports.items()
            },
            "deltas": [asdict(d) for d in self.deltas],
        }


# ---------------------------------------------------------------------------
# Ground Truth Dataset
# ---------------------------------------------------------------------------

class GroundTruthDataset:
    """
    JSON de ground truth:
    [
      {
        "query_id": "q001",
        "query_text": "espessura mínima N-115 tubulação carbono",
        "relevant_doc_ids": ["chunk_id_a", "chunk_id_b"],
        "metadata_filter": {"norma": "N-115"}   // opcional
      }
    ]
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: list[dict] = []
        self._load()

    def _load(self) -> None:
        last_err: Exception = Exception("Arquivo vazio ou inacessível")
        for encoding in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                with open(self.path, encoding=encoding) as f:
                    self._data = json.load(f)
                print(f"[GroundTruth] {len(self._data)} queries (encoding={encoding}): {self.path}")
                return
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                last_err = e
        raise ValueError(f"[GroundTruth] Falha ao decodificar {self.path}: {last_err}")

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    @staticmethod
    def create_template(output_path: str | Path, num_examples: int = 3) -> None:
        template = [
            {
                "query_id": f"q{i:03d}",
                "query_text": f"<INSIRA SUA QUERY AQUI {i}>",
                "relevant_doc_ids": ["<chunk_id_1>", "<chunk_id_2>"],
                "metadata_filter": {"norma": "<N-115 | N-279 | ...>"},
            }
            for i in range(1, num_examples + 1)
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        print(f"[GroundTruth] Template salvo em: {output_path}")


# ---------------------------------------------------------------------------
# Metric Calculators — funções puras
# ---------------------------------------------------------------------------

def precision_at_k(relevance_vector: list[int], k: int) -> float:
    """P@K = |{rel docs em top-K}| / K"""
    if k <= 0:
        raise ValueError("k deve ser > 0")
    return sum(relevance_vector[:k]) / k


def reciprocal_rank(relevance_vector: list[int]) -> float:
    """RR = 1/rank_primeiro_relevante (base-1). 0.0 se nenhum relevante."""
    for rank, rel in enumerate(relevance_vector, start=1):
        if rel == 1:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance_vector: list[int], k: int) -> float:
    """DCG@K = Σ_{i=1}^{K} rel_i / log2(i+1)"""
    return sum(
        rel / math.log2(i + 1)
        for i, rel in enumerate(relevance_vector[:k], start=1)
    )


def ndcg_at_k(relevance_vector: list[int], k: int) -> float:
    """nDCG@K = DCG@K / IDCG@K. Retorna 0.0 se IDCG=0."""
    actual_dcg = dcg_at_k(relevance_vector, k)
    ideal_dcg  = dcg_at_k(sorted(relevance_vector, reverse=True), k)
    return 0.0 if ideal_dcg == 0.0 else actual_dcg / ideal_dcg


def hit_rate_at_k(relevance_vector: list[int], k: int) -> float:
    """HR@K = 1.0 se ∃ relevante em top-K, else 0.0."""
    return 1.0 if any(r == 1 for r in relevance_vector[:k]) else 0.0


# ---------------------------------------------------------------------------
# RAG Evaluator (online — conecta ao Qdrant)
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Avaliação end-to-end contra uma única coleção Qdrant.
    Para avaliar múltiplas coleções, use MultiCollectionEvaluator.

    Args:
        collection_name:  coleção alvo (sobrescreve cfg.collection_name).
        settings:         Settings pré-carregado (opcional).
        env_path:         Caminho explícito para .env (opcional).
        k_values:         Cortes K (padrão: [1, 3, 5, 10]).
        top_k_retrieve:   Docs buscados no Qdrant; deve ser >= max(k_values).
        chunk_id_field:   Campo no payload Qdrant com o ID do chunk.
    """

    def __init__(
        self,
        collection_name: str,
        settings: Settings | None = None,
        env_path: str | Path | None = None,
        k_values: list[int] | None = None,
        top_k_retrieve: int = 10,
        chunk_id_field: str = "chunk_id",
    ):
        self.cfg = settings or Settings.from_env(env_path)

        self._collection_name = collection_name
        self.k_values        = sorted(k_values or [1, 3, 5, 10])
        self.top_k_retrieve  = max(top_k_retrieve, max(self.k_values))
        self.chunk_id_field  = chunk_id_field

        print(f"[RAGEvaluator] Conectando ao Qdrant: {self.cfg.qdrant_url}")
        self.qdrant = QdrantClient(
            url=self.cfg.qdrant_url,
            api_key=self.cfg.qdrant_api_key,
        )

        # ── Validação 1: coleção existe ──────────────────────────────────────
        available = self._list_available_collections()
        if collection_name not in available:
            raise ValueError(
                f"[RAGEvaluator] Coleção '{collection_name}' não existe no cluster.\n"
                f"Coleções disponíveis: {available}\n"
                f"Dica: atualize COLLECTION_NAMES no .env."
            )

        # ── Validação 2: introspecta campos indexados e dimensão do vetor ────
        # Ambas ANTES de carregar o embedder: fail-fast sem desperdiçar memória
        self._indexed_fields: set[str]  = self._introspect_indexed_fields()
        self._expected_dim:   int       = self._introspect_vector_dim()

        # ── Carrega embedder e valida dimensão ───────────────────────────────
        import warnings
        print(f"[RAGEvaluator] Carregando embedding: {self.cfg.embedding_model}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.embedder = SentenceTransformer(self.cfg.embedding_model)

        actual_dim = self.embedder.get_sentence_embedding_dimension()
        if actual_dim != self._expected_dim:
            raise ValueError(
                f"[RAGEvaluator] INCOMPATIBILIDADE DE DIMENSÃO DE VETOR\n"
                f"  Coleção '{collection_name}' espera: {self._expected_dim} dims\n"
                f"  Modelo '{self.cfg.embedding_model}' produz: {actual_dim} dims\n\n"
                f"  Você precisa usar o MESMO modelo utilizado durante a ingestão.\n"
                f"  Modelos comuns por dimensão:\n"
                f"    384  → sentence-transformers/all-MiniLM-L6-v2\n"
                f"    768  → sentence-transformers/all-mpnet-base-v2\n"
                f"    1024 → BAAI/bge-large-en-v1.5  |  BAAI/bge-m3\n"
                f"    1536 → text-embedding-ada-002 (OpenAI)\n\n"
                f"  Corrija EMBEDDING_MODEL no .env e reexecute."
            )

        print(
            f"[RAGEvaluator] Dimensão validada: {actual_dim}d ✓ | "
            f"Coleção '{collection_name}' OK."
        )

    def _list_available_collections(self) -> list[str]:
        return [c.name for c in self.qdrant.get_collections().collections]

    def _introspect_indexed_fields(self) -> set[str]:
        """
        Retorna o conjunto de campos com payload index na coleção alvo.

        Qdrant exige PayloadIndex para qualquer campo usado em FieldCondition.
        Sem índice a API retorna:
          "Bad request: Index required but not found for <campo>"

        Usamos esse conjunto em _build_filter_safe() para silenciar condições
        sobre campos não-indexados em vez de lançar exceção em runtime.
        """
        try:
            info   = self.qdrant.get_collection(self._collection_name)
            schema = info.payload_schema or {}
            # payload_schema pode ser um objeto Pydantic ou dict dependendo
            # da versão do client — normalizamos para dict de qualquer forma
            if hasattr(schema, "__iter__"):
                indexed = set(schema)
            else:
                indexed = set()
            if indexed:
                print(f"[RAGEvaluator] Campos indexados em '{self._collection_name}': {sorted(indexed)}")
            else:
                print(
                    f"[RAGEvaluator] AVISO: nenhum payload index encontrado em "
                    f"'{self._collection_name}'.\n"
                    f"  → metadata_filter das queries será IGNORADO.\n"
                    f"  → Execute create_payload_indexes.py para habilitar filtros."
                )
            return indexed
        except Exception as e:
            print(f"[RAGEvaluator] AVISO: não foi possível listar índices: {e}")
            return set()

    def _introspect_vector_dim(self) -> int:
        """
        Lê a dimensão do vetor configurada na coleção Qdrant.

        Qdrant armazena a config em:
          collection_info.config.params.vectors

        Dois formatos possíveis dependendo de como a coleção foi criada:
          A) Vetor único (named=False):
             vectors = VectorsConfig(size=1024, distance=Cosine)
          B) Vetores nomeados (named=True):
             vectors = {"text": VectorsConfig(size=1024, ...), ...}

        Retorna a dimensão do primeiro vetor encontrado.
        Retorna 0 se não conseguir determinar (não bloqueia a execução,
        mas a validação de dim será ignorada).
        """
        try:
            info   = self.qdrant.get_collection(self._collection_name)
            params = info.config.params
            vec    = params.vectors

            # Formato A: VectorsConfig direto (atributo .size)
            if hasattr(vec, "size"):
                dim = int(vec.size)
                print(f"[RAGEvaluator] Dimensão vetorial de '{self._collection_name}': {dim}d")
                return dim

            # Formato B: dict de vetores nomeados → pega o primeiro
            if isinstance(vec, dict) and vec:
                first_key = next(iter(vec))
                dim = int(vec[first_key].size)
                print(
                    f"[RAGEvaluator] Dimensão vetorial de '{self._collection_name}' "
                    f"(vetor '{first_key}'): {dim}d"
                )
                return dim

            print(f"[RAGEvaluator] AVISO: não foi possível determinar dimensão vetorial.")
            return 0

        except Exception as e:
            print(f"[RAGEvaluator] AVISO: erro ao ler dimensão vetorial: {e}")
            return 0

    # --- Retrieve ---

    def _embed(self, text: str) -> list[float]:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def _build_filter_safe(self, metadata_filter: dict | None) -> Filter | None:
        """
        Constrói um Filter do Qdrant descartando campos NÃO-indexados.

        Lógica de segurança:
          1. Se metadata_filter é None/vazio → retorna None (sem filtro).
          2. Para cada (campo, valor) do filtro:
             - Se campo está em self._indexed_fields → inclui na condição.
             - Se campo NÃO está                    → loga warning e descarta.
          3. Se nenhum campo sobrou após filtragem → retorna None.

        Isso garante que uma query com metadata_filter parcialmente indexado
        funcione parcialmente em vez de falhar completamente.
        """
        if not metadata_filter:
            return None

        valid_conditions   = []
        skipped_fields: list[str] = []

        for campo, valor in metadata_filter.items():
            if campo in self._indexed_fields:
                valid_conditions.append(
                    FieldCondition(key=campo, match=MatchValue(value=valor))
                )
            else:
                skipped_fields.append(campo)

        if skipped_fields:
            print(
                f"  [FILTER-WARN] Campos sem índice ignorados: {skipped_fields}\n"
                f"  → Execute create_payload_indexes.py para criá-los."
            )

        if not valid_conditions:
            return None

        return Filter(must=valid_conditions)

    def retrieve(
        self,
        query_text: str,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedDoc]:
        """
        Busca documentos no Qdrant.
        Filtros em campos não-indexados são descartados com warning
        em vez de lançar BadRequest.
        """
        qdrant_filter = self._build_filter_safe(metadata_filter)

        hits = self.qdrant.search(
            collection_name=self._collection_name,
            query_vector=self._embed(query_text),
            limit=self.top_k_retrieve,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            RetrievedDoc(
                doc_id=hit.payload.get(self.chunk_id_field, str(hit.id)),
                score=hit.score,
                payload=hit.payload,
            )
            for hit in hits
        ]

    # --- Evaluate single query ---

    def evaluate_query(
        self,
        query_id: str,
        query_text: str,
        relevant_ids: set[str],
        metadata_filter: dict | None = None,
    ) -> QueryResult:
        retrieved = self.retrieve(query_text, metadata_filter)
        for doc in retrieved:
            doc.is_relevant = doc.doc_id in relevant_ids

        rv = [1 if d.is_relevant else 0 for d in retrieved]
        result = QueryResult(
            query_id=query_id,
            query_text=query_text,
            retrieved=retrieved,
            relevant_ids=relevant_ids,
            mrr=reciprocal_rank(rv),
        )
        for k in self.k_values:
            result.precision_at_k[k] = precision_at_k(rv, k)
            result.ndcg_at_k[k]      = ndcg_at_k(rv, k)
            result.hit_rate_at_k[k]  = hit_rate_at_k(rv, k)
        return result

    # --- Evaluate full dataset ---

    def evaluate(
        self,
        ground_truth: GroundTruthDataset,
        verbose: bool = True,
    ) -> EvaluationReport:
        import datetime
        all_results: list[QueryResult] = []

        for entry in ground_truth:
            qid     = entry["query_id"]
            qtext   = entry["query_text"]
            rel_ids = set(entry["relevant_doc_ids"])
            meta    = entry.get("metadata_filter")

            if verbose:
                print(f"  [{self._collection_name}] {qid}: '{qtext[:65]}'")

            qr = self.evaluate_query(qid, qtext, rel_ids, meta)
            all_results.append(qr)

            if verbose:
                self._log_query(qr)

        report = EvaluationReport(
            collection_name=self._collection_name,
            num_queries=len(all_results),
            k_values=self.k_values,
            # Usa o modelo real do embedder atribuído a esta coleção.
            # _col_model não existe em RAGEvaluator standalone — fallback para cfg.
            embedding_model=getattr(self, "_embedding_model_name", self.cfg.embedding_model),
            timestamp=datetime.datetime.utcnow().isoformat(),
            mean_mrr=float(np.mean([r.mrr for r in all_results])),
        )
        for k in self.k_values:
            report.mean_precision[k] = float(np.mean([r.precision_at_k[k] for r in all_results]))
            report.mean_ndcg[k]      = float(np.mean([r.ndcg_at_k[k]      for r in all_results]))
            report.mean_hit_rate[k]  = float(np.mean([r.hit_rate_at_k[k]  for r in all_results]))
        report.per_query_results = all_results
        return report

    def _log_query(self, r: QueryResult) -> None:
        found = sum(1 for d in r.retrieved if d.is_relevant)
        k = self.k_values[-1]
        print(
            f"    → MRR={r.mrr:.3f} | P@{k}={r.precision_at_k[k]:.3f} | "
            f"nDCG@{k}={r.ndcg_at_k[k]:.3f} | Relevantes: {found}/{len(r.relevant_ids)}"
        )


# ---------------------------------------------------------------------------
# Mapa canônico: dimensão vetorial → modelo de embedding mais provável
#
# Lógica: ao detectar dimensão X na coleção, este mapa resolve qual
# SentenceTransformer usar para produzir vetores compatíveis.
# O .env EMBEDDING_MODEL é o "default" para coleções de dimensão conhecida
# se o usuário quiser forçar um modelo específico por dimensão, pode sobrescrever
# via EMBEDDING_MODEL_<DIM> (ex: EMBEDDING_MODEL_384=meu-modelo-customizado).
# ---------------------------------------------------------------------------

_DIM_TO_MODEL: dict[int, str] = {
    384:  "sentence-transformers/all-MiniLM-L6-v2",
    768:  "sentence-transformers/all-mpnet-base-v2",
    1024: "BAAI/bge-large-en-v1.5",
    1536: "text-embedding-ada-002",    # OpenAI — requer client diferente
}


def _resolve_model_for_dim(dim: int, cfg_model: str) -> str:
    """
    Retorna o modelo correto para uma dimensão vetorial.

    Prioridade:
      1. Variável de ambiente EMBEDDING_MODEL_<DIM>  (override por dimensão)
      2. cfg_model (EMBEDDING_MODEL do .env) se a dimensão do modelo bater
      3. _DIM_TO_MODEL[dim] (tabela canônica)
      4. cfg_model como fallback (aviso será emitido se dim não bater)
    """
    # Override explícito por dimensão: EMBEDDING_MODEL_1024=BAAI/bge-m3
    env_override = os.environ.get(f"EMBEDDING_MODEL_{dim}", "").strip()
    if env_override:
        return env_override

    # Tabela canônica
    return _DIM_TO_MODEL.get(dim, cfg_model)


# ---------------------------------------------------------------------------
# Multi-Collection Evaluator
# ---------------------------------------------------------------------------

class MultiCollectionEvaluator:
    """
    Avalia o mesmo ground truth em múltiplas coleções Qdrant com dimensões
    vetoriais potencialmente DIFERENTES.

    Problema real do projeto:
      petrobras_rag_teoria     → 1024d (ingerido com BAAI/bge-large-en-v1.5)
      normas_tecnicas_publicas → 384d  (ingerido com all-MiniLM-L6-v2)

    Solução: cada coleção tem seu próprio embedder, resolvido automaticamente
    pela dimensão vetorial detectada via get_collection() → config.params.vectors.

    Pool de embedders:
      _embedder_pool: dict[int, SentenceTransformer]
        ex: {1024: <bge-large>, 384: <MiniLM>}

      Embedders são carregados sob demanda e reutilizados entre coleções de
      mesma dimensão — evita carregar o mesmo modelo duas vezes.

    Override por dimensão via .env:
      EMBEDDING_MODEL_384=sentence-transformers/meu-modelo-384d
      EMBEDDING_MODEL_1024=BAAI/bge-m3

    Args:
        collection_names: Lista de coleções a avaliar. Se None, usa
                          cfg.collection_names do .env.
        settings:         Settings pré-carregado (opcional).
        env_path:         Caminho para .env (opcional).
        k_values:         Cortes K.
        top_k_retrieve:   Docs buscados por query.
        chunk_id_field:   Campo de ID no payload Qdrant.
    """

    def __init__(
        self,
        collection_names: list[str] | None = None,
        settings: Settings | None = None,
        env_path: str | Path | None = None,
        k_values: list[int] | None = None,
        top_k_retrieve: int = 10,
        chunk_id_field: str = "chunk_id",
    ):
        import warnings
        self.cfg = settings or Settings.from_env(env_path)
        print(self.cfg.masked_repr())

        self.collection_names = collection_names or list(self.cfg.collection_names)
        self.k_values         = sorted(k_values or [1, 3, 5, 10])
        self.top_k_retrieve   = max(top_k_retrieve, max(self.k_values))
        self.chunk_id_field   = chunk_id_field

        # Pool de embedders: dim → SentenceTransformer
        # Carregado lazy em _get_or_load_embedder()
        self._embedder_pool: dict[int, tuple[SentenceTransformer, str]] = {}
        # tuple = (model_instance, model_name_str)

        # Conexão Qdrant compartilhada entre todas as coleções
        print(f"[MultiEval] Conectando ao Qdrant: {self.cfg.qdrant_url}")
        self._qdrant = QdrantClient(
            url=self.cfg.qdrant_url,
            api_key=self.cfg.qdrant_api_key,
        )
        available = [c.name for c in self._qdrant.get_collections().collections]
        print(f"[MultiEval] Coleções disponíveis no cluster: {available}")

        missing_cols = [c for c in self.collection_names if c not in available]
        if missing_cols:
            raise ValueError(
                f"[MultiEval] Coleções não encontradas: {missing_cols}\n"
                f"Disponíveis: {available}"
            )

        # Pré-carrega embedders para todas as dimensões necessárias
        # (fail-fast: melhor descobrir modelo ausente antes de rodar queries)
        print(f"\n[MultiEval] Detectando dimensões vetoriais e carregando embedders...")
        self._col_dim:   dict[str, int] = {}    # collection → dim
        self._col_model: dict[str, str] = {}    # collection → model_name

        for col in self.collection_names:
            dim = self._detect_dim(col)
            self._col_dim[col] = dim
            model_name = _resolve_model_for_dim(dim, self.cfg.embedding_model)
            self._col_model[col] = model_name
            self._get_or_load_embedder(dim, model_name)

        print(f"\n[MultiEval] Mapa coleção → embedder:")
        for col in self.collection_names:
            dim = self._col_dim[col]
            print(f"  {col:<35} {dim:>5}d  →  {self._col_model[col]}")

        print(f"\n[MultiEval] Pronto. {len(self.collection_names)} coleção(ões).")

    def _detect_dim(self, collection_name: str) -> int:
        """Lê dimensão vetorial da coleção via get_collection()."""
        try:
            info   = self._qdrant.get_collection(collection_name)
            params = info.config.params
            vec    = params.vectors
            if hasattr(vec, "size"):
                return int(vec.size)
            if isinstance(vec, dict) and vec:
                return int(vec[next(iter(vec))].size)
        except Exception as e:
            print(f"  [WARN] Não foi possível detectar dim de '{collection_name}': {e}")
        return 0

    def _get_or_load_embedder(
        self, dim: int, model_name: str
    ) -> SentenceTransformer:
        """
        Retorna embedder do pool se já carregado para esta dimensão.
        Carrega e armazena no pool caso contrário.

        Pool key = dim (não model_name): se duas coleções com mesma dimensão
        usarem modelos diferentes, o segundo modelo sobrescreve o primeiro.
        Para forçar modelos distintos com mesma dim, use override por dim
        (EMBEDDING_MODEL_384=...) no .env.
        """
        import warnings
        if dim not in self._embedder_pool:
            print(f"  [Load] {model_name}  ({dim}d)...")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                embedder = SentenceTransformer(model_name)
            actual_dim = embedder.get_sentence_embedding_dimension()
            if actual_dim != dim and dim != 0:
                raise ValueError(
                    f"[MultiEval] Modelo '{model_name}' produz {actual_dim}d "
                    f"mas a coleção espera {dim}d.\n"
                    f"Defina EMBEDDING_MODEL_{dim}=<modelo-correto> no .env."
                )
            self._embedder_pool[dim] = (embedder, model_name)
            print(f"  [OK]   {model_name}  ({actual_dim}d)")
        return self._embedder_pool[dim][0]

    def _make_evaluator(self, collection_name: str) -> RAGEvaluator:
        """
        Constrói RAGEvaluator para uma coleção específica, injetando o
        embedder correto para a dimensão vetorial desta coleção.
        """
        dim        = self._col_dim[collection_name]
        model_name = self._col_model[collection_name]
        embedder   = self._get_or_load_embedder(dim, model_name)

        ev = object.__new__(RAGEvaluator)
        ev.cfg                    = self.cfg
        ev._collection_name       = collection_name
        ev.k_values               = self.k_values
        ev.top_k_retrieve         = self.top_k_retrieve
        ev.chunk_id_field         = self.chunk_id_field
        ev.qdrant                 = self._qdrant
        ev.embedder               = embedder
        ev._expected_dim          = dim
        ev._embedding_model_name  = model_name   # nome real para o relatório
        ev._indexed_fields        = ev._introspect_indexed_fields()
        return ev

    def evaluate_all(
        self,
        ground_truth: GroundTruthDataset,
        verbose: bool = True,
    ) -> ComparisonReport:
        """
        Avalia todas as coleções e retorna ComparisonReport com deltas.
        """
        import datetime

        reports: dict[str, EvaluationReport] = {}

        for col in self.collection_names:
            print(f"\n[MultiEval] ── Avaliando coleção: {col} ──")
            ev     = self._make_evaluator(col)
            report = ev.evaluate(ground_truth, verbose=verbose)
            reports[col] = report

        ts = datetime.datetime.utcnow().isoformat()
        comparison = ComparisonReport(
            embedding_model=self.cfg.embedding_model,
            k_values=self.k_values,
            num_queries=len(ground_truth),
            reports=reports,
            timestamp=ts,
        )

        # Calcula deltas somente se exatamente 2 coleções (A vs B)
        if len(self.collection_names) == 2:
            comparison.deltas = _compute_deltas(
                reports[self.collection_names[0]],
                reports[self.collection_names[1]],
                self.k_values,
            )

        return comparison


def _compute_deltas(
    rep_a: EvaluationReport,
    rep_b: EvaluationReport,
    k_values: list[int],
) -> list[CollectionDelta]:
    """
    Δabs = B - A
    Δrel% = (B - A) / A * 100    (se A=0 → Δrel% = +inf ou 0.0)
    """
    def _delta(metric: str, a: float, b: float) -> CollectionDelta:
        d_abs = b - a
        d_rel = ((b - a) / a * 100) if a != 0.0 else (float("inf") if b > 0 else 0.0)
        winner = rep_b.collection_name if b > a else (rep_a.collection_name if a > b else "empate")
        return CollectionDelta(
            metric=metric,
            score_a=round(a, 4),
            score_b=round(b, 4),
            delta_abs=round(d_abs, 4),
            delta_rel_pct=round(d_rel, 2),
            winner=winner,
        )

    deltas = [_delta("MRR", rep_a.mean_mrr, rep_b.mean_mrr)]
    for k in k_values:
        deltas.append(_delta(f"P@{k}",     rep_a.mean_precision[k], rep_b.mean_precision[k]))
        deltas.append(_delta(f"nDCG@{k}",  rep_a.mean_ndcg[k],      rep_b.mean_ndcg[k]))
        deltas.append(_delta(f"HR@{k}",    rep_a.mean_hit_rate[k],  rep_b.mean_hit_rate[k]))
    return deltas


# ---------------------------------------------------------------------------
# Report Printer & Persister
# ---------------------------------------------------------------------------

class MetricsReporter:

    @staticmethod
    def print_summary(report: EvaluationReport) -> None:
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  RELATÓRIO — {report.collection_name}")
        print(f"  Embedding : {report.embedding_model}")
        print(f"  Queries   : {report.num_queries} | {report.timestamp}")
        print(sep)
        print(f"  MRR: {report.mean_mrr:.4f}")
        print()
        header = f"{'K':>5} | {'Precision@K':>12} | {'nDCG@K':>10} | {'HitRate@K':>10}"
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        for k in report.k_values:
            print(
                f"  {k:>5} | {report.mean_precision[k]:>12.4f} | "
                f"{report.mean_ndcg[k]:>10.4f} | {report.mean_hit_rate[k]:>10.4f}"
            )
        print(sep)
        worst = sorted(report.per_query_results, key=lambda r: r.mrr)[:3]
        print("\n  ⚠  Top-3 piores queries (MRR mais baixo):")
        for qr in worst:
            print(f"    [{qr.query_id}] MRR={qr.mrr:.3f} | '{qr.query_text[:70]}'")
        print()

    @staticmethod
    def print_comparison(comp: ComparisonReport) -> None:
        """
        Imprime tabela side-by-side com scores de todas as coleções
        e, se 2 coleções, exibe coluna de Δ e destaca vencedor.
        """
        cols      = list(comp.reports.keys())
        reports   = comp.reports
        k_values  = comp.k_values
        sep       = "=" * (65 + 18 * len(cols))
        col_w     = 16

        print(f"\n{sep}")
        print(f"  COMPARAÇÃO MULTI-COLEÇÃO")
        print(f"  Embedding : {comp.embedding_model}")
        print(f"  Queries   : {comp.num_queries} | {comp.timestamp}")
        print(sep)

        # Cabeçalho
        header = f"  {'Métrica':<14}" + "".join(f"{c:>{col_w}}" for c in cols)
        if len(cols) == 2:
            header += f"{'Δabs':>{col_w}}{'Δrel%':>{col_w}}{'Vencedor':<20}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        # Linhas de métricas
        def _row(metric: str, scores: list[float]) -> str:
            line = f"  {metric:<14}" + "".join(f"{s:>{col_w}.4f}" for s in scores)
            if len(cols) == 2 and comp.deltas:
                d = next((x for x in comp.deltas if x.metric == metric), None)
                if d:
                    delta_str = f"{d.delta_abs:>+.4f}"
                    rel_str   = f"{d.delta_rel_pct:>+.1f}%"
                    winner    = f"  ✓ {d.winner}" if d.delta_abs != 0 else "  = empate"
                    line += f"{delta_str:>{col_w}}{rel_str:>{col_w}}{winner:<20}"
            return line

        print(_row("MRR", [reports[c].mean_mrr for c in cols]))
        for k in k_values:
            print(_row(f"P@{k}",    [reports[c].mean_precision[k] for c in cols]))
            print(_row(f"nDCG@{k}", [reports[c].mean_ndcg[k]      for c in cols]))
            print(_row(f"HR@{k}",   [reports[c].mean_hit_rate[k]  for c in cols]))

        print(sep)

        # Resumo de vitórias (apenas 2 coleções)
        if len(cols) == 2 and comp.deltas:
            wins = {cols[0]: 0, cols[1]: 0, "empate": 0}
            for d in comp.deltas:
                key = d.winner if d.winner in wins else "empate"
                wins[key] += 1
            print(f"\n  Placar: {cols[0]}={wins[cols[0]]} | "
                  f"{cols[1]}={wins[cols[1]]} | empates={wins['empate']}")
            overall_winner = cols[0] if wins[cols[0]] > wins[cols[1]] else (
                cols[1] if wins[cols[1]] > wins[cols[0]] else "empate"
            )
            print(f"  Vencedor geral: {overall_winner}\n")

    @staticmethod
    def save_json(report: EvaluationReport, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[Reporter] JSON → {output_path}")

    @staticmethod
    def save_comparison_json(comp: ComparisonReport, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comp.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[Reporter] Comparação JSON → {output_path}")

    @staticmethod
    def save_csv(report: EvaluationReport, output_path: str | Path) -> None:
        import csv
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for qr in report.per_query_results:
            for k in report.k_values:
                rows.append({
                    "collection":        report.collection_name,
                    "query_id":          qr.query_id,
                    "query_text":        qr.query_text[:100],
                    "embedding_model":   report.embedding_model,
                    "mrr":               round(qr.mrr, 4),
                    "k":                 k,
                    "precision_at_k":    round(qr.precision_at_k[k], 4),
                    "ndcg_at_k":         round(qr.ndcg_at_k[k], 4),
                    "hit_rate_at_k":     round(qr.hit_rate_at_k[k], 4),
                    "relevant_found":    sum(1 for d in qr.retrieved if d.is_relevant),
                    "total_relevant_gt": len(qr.relevant_ids),
                })
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Reporter] CSV → {output_path}")

    @staticmethod
    def save_comparison_csv(comp: ComparisonReport, output_path: str | Path) -> None:
        """CSV com uma linha por (coleção × métrica × K) para pivot no Excel."""
        import csv
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for col, rep in comp.reports.items():
            rows.append({
                "collection": col, "metric": "MRR", "k": "-",
                "score": round(rep.mean_mrr, 4),
                "embedding_model": comp.embedding_model,
            })
            for k in comp.k_values:
                for metric, val in [
                    (f"P@{k}",    rep.mean_precision[k]),
                    (f"nDCG@{k}", rep.mean_ndcg[k]),
                    (f"HR@{k}",   rep.mean_hit_rate[k]),
                ]:
                    rows.append({
                        "collection": col, "metric": metric, "k": k,
                        "score": round(val, 4),
                        "embedding_model": comp.embedding_model,
                    })
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Reporter] Comparação CSV → {output_path}")


# ---------------------------------------------------------------------------
# Offline Evaluator — testes unitários e CI (sem Qdrant, sem .env)
# ---------------------------------------------------------------------------

class OfflineRAGEvaluator:
    """
    Aceita qualquer Callable[[str], list[tuple[str, float]]] como retriever.
    Não carrega .env, não instancia Qdrant nem SentenceTransformer.
    """

    def __init__(
        self,
        retriever_fn: Callable[[str], list[tuple[str, float]]],
        k_values: list[int] | None = None,
    ):
        self.retriever_fn = retriever_fn
        self.k_values = sorted(k_values or [1, 3, 5, 10])

    def evaluate_query(
        self,
        query_id: str,
        query_text: str,
        relevant_ids: set[str],
    ) -> QueryResult:
        raw = self.retriever_fn(query_text)
        retrieved = [
            RetrievedDoc(doc_id=did, score=sc, payload={})
            for did, sc in raw
        ]
        for doc in retrieved:
            doc.is_relevant = doc.doc_id in relevant_ids

        rv = [1 if d.is_relevant else 0 for d in retrieved]
        result = QueryResult(
            query_id=query_id,
            query_text=query_text,
            retrieved=retrieved,
            relevant_ids=relevant_ids,
            mrr=reciprocal_rank(rv),
        )
        for k in self.k_values:
            result.precision_at_k[k] = precision_at_k(rv, k)
            result.ndcg_at_k[k]      = ndcg_at_k(rv, k)
            result.hit_rate_at_k[k]  = hit_rate_at_k(rv, k)
        return result