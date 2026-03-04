"""
test_rag_metrics.py
===================
Testes unitários e smoke tests.

Cobertura:
  1. Funções puras de métricas (sem I/O)
  2. Settings.from_env() com .env mockado via monkeypatch
  3. OfflineRAGEvaluator — fluxo completo sem Qdrant
  4. Smoke test executável diretamente: python test_rag_metrics.py

Execução:
  python -m pytest test_rag_metrics.py -v        # modo pytest
  python test_rag_metrics.py                     # modo smoke test standalone
"""

import math
import os
import pytest

from rag_metrics import (
    Settings,
    precision_at_k,
    reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    OfflineRAGEvaluator,
    EvaluationReport,
    MetricsReporter,
)


# ---------------------------------------------------------------------------
# Fixture: variáveis de ambiente mínimas para Settings (sem .env real)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_env(monkeypatch):
    """Injeta variáveis mínimas para Settings.from_env() não falhar em CI."""
    monkeypatch.setenv("QDRANT_URL",        "https://mock.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY",    "mock_api_key_123456")
    monkeypatch.setenv("COLLECTION_NAME",   "Mota_engil_multimodal")
    monkeypatch.setenv("EMBEDDING_MODEL",   "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("HF_TOKEN",          "hf_mocktoken")
    monkeypatch.setenv("LLM_MODEL",         "llama3:latest")


# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

class TestSettings:
    def test_from_env_reads_all_keys(self, mock_env):
        cfg = Settings.from_env()
        assert cfg.qdrant_url      == "https://mock.qdrant.io"
        assert cfg.collection_name == "Mota_engil_multimodal"
        assert cfg.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.llm_model       == "llama3:latest"

    def test_missing_required_raises(self, monkeypatch):
        """QDRANT_URL ausente → EnvironmentError."""
        monkeypatch.delenv("QDRANT_URL",      raising=False)
        monkeypatch.delenv("QDRANT_API_KEY",  raising=False)
        monkeypatch.delenv("COLLECTION_NAME", raising=False)
        with pytest.raises(EnvironmentError, match="QDRANT_URL"):
            Settings.from_env()

    def test_masked_repr_hides_key(self, mock_env):
        cfg = Settings.from_env()
        rep = cfg.masked_repr()
        assert "mock_api_key_123456" not in rep
        assert "****" in rep

    def test_hf_token_injected_into_env(self, mock_env):
        Settings.from_env()
        # HF_TOKEN deve estar disponível para huggingface_hub
        assert os.environ.get("HUGGING_FACE_HUB_TOKEN") == "hf_mocktoken"

    def test_frozen_immutable(self, mock_env):
        cfg = Settings.from_env()
        with pytest.raises(Exception):   # FrozenInstanceError (dataclass frozen=True)
            cfg.collection_name = "outro"  # type: ignore


# ---------------------------------------------------------------------------
# 2. Precision@K
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k([1, 1, 1], k=3) == pytest.approx(1.0)

    def test_none_relevant(self):
        assert precision_at_k([0, 0, 0, 0, 0], k=5) == pytest.approx(0.0)

    def test_partial_2_of_5(self):
        # [1,0,1,0,0] → 2/5
        assert precision_at_k([1, 0, 1, 0, 0], k=5) == pytest.approx(0.4)

    def test_k_truncates_vector(self):
        # k=2, vetor=[1,0,1,1] → considera só [1,0] → 1/2
        assert precision_at_k([1, 0, 1, 1], k=2) == pytest.approx(0.5)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k deve ser > 0"):
            precision_at_k([1, 0], k=0)


# ---------------------------------------------------------------------------
# 3. MRR / Reciprocal Rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    def test_rank_1(self):
        assert reciprocal_rank([1, 0, 0]) == pytest.approx(1.0)

    def test_rank_2(self):
        assert reciprocal_rank([0, 1, 0]) == pytest.approx(0.5)

    def test_rank_3(self):
        assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3, rel=1e-5)

    def test_no_relevant(self):
        assert reciprocal_rank([0, 0, 0]) == pytest.approx(0.0)

    def test_multiple_relevant_uses_first(self):
        # Rank do PRIMEIRO relevante (pos 2) → 1/2
        assert reciprocal_rank([0, 1, 1]) == pytest.approx(0.5)

    def test_mean_mrr_two_queries(self):
        # q1: rank 1 → RR=1.0 | q2: rank 2 → RR=0.5 | MRR=0.75
        rr1 = reciprocal_rank([1, 0, 0])
        rr2 = reciprocal_rank([0, 1, 0])
        assert (rr1 + rr2) / 2 == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 4. DCG / nDCG
# ---------------------------------------------------------------------------

class TestDCG:
    def test_manual_dcg(self):
        # [1,0,1] → 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5 = 1.5
        expected = 1 / math.log2(2) + 0 + 1 / math.log2(4)
        assert dcg_at_k([1, 0, 1], k=3) == pytest.approx(expected)

    def test_dcg_truncation(self):
        # k=2 de [1,0,1] → só [1,0] → 1/log2(2) = 1.0
        assert dcg_at_k([1, 0, 1], k=2) == pytest.approx(1.0)


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k([1, 1, 1], k=3) == pytest.approx(1.0)

    def test_no_relevant_returns_zero(self):
        # Evita ZeroDivisionError
        assert ndcg_at_k([0, 0, 0], k=3) == pytest.approx(0.0)

    def test_relevant_at_rank_3_vs_ideal_rank_1(self):
        # DCG@3([0,0,1]) = 1/log2(4) = 0.5
        # IDCG@3([1,0,0]) = 1/log2(2) = 1.0
        # nDCG = 0.5
        assert ndcg_at_k([0, 0, 1], k=3) == pytest.approx(0.5)

    def test_partial_relevance_formula(self):
        rv = [1, 0, 1, 0, 0]
        ideal = sorted(rv, reverse=True)
        expected = dcg_at_k(rv, 5) / dcg_at_k(ideal, 5)
        assert ndcg_at_k(rv, k=5) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. Hit Rate@K
# ---------------------------------------------------------------------------

class TestHitRate:
    def test_hit_at_k1(self):
        assert hit_rate_at_k([1, 0, 0], k=1) == 1.0

    def test_no_hit(self):
        assert hit_rate_at_k([0, 0, 0], k=3) == 0.0

    def test_relevant_outside_k_is_miss(self):
        # Relevante na pos 4, k=3 → miss
        assert hit_rate_at_k([0, 0, 0, 1], k=3) == 0.0

    def test_hit_at_boundary(self):
        assert hit_rate_at_k([0, 0, 1], k=3) == 1.0


# ---------------------------------------------------------------------------
# 6. OfflineRAGEvaluator — fluxo end-to-end sem Qdrant
# ---------------------------------------------------------------------------

class TestOfflineEvaluator:
    """
    Corpus sintético controlado:
      q001 → relevantes: doc_A, doc_B
      q002 → relevante : doc_C (não está no corpus → miss)
    """

    CORPUS = {
        "q001": [("doc_A", 0.95), ("doc_X", 0.80), ("doc_B", 0.75), ("doc_Y", 0.60), ("doc_Z", 0.50)],
        "q002": [("doc_X", 0.90), ("doc_Y", 0.85), ("doc_Z", 0.80), ("doc_A", 0.70), ("doc_B", 0.65)],
    }

    @staticmethod
    def _retriever(query_text: str):
        qid = "q001" if "espessura" in query_text else "q002"
        return TestOfflineEvaluator.CORPUS[qid]

    def _evaluator(self, k_values=None):
        return OfflineRAGEvaluator(
            retriever_fn=self._retriever,
            k_values=k_values or [1, 3, 5],
        )

    def test_q001_mrr_is_1(self):
        ev = self._evaluator()
        r = ev.evaluate_query("q001", "espessura mínima tubulação", {"doc_A", "doc_B"})
        # doc_A está em rank 1 → MRR = 1.0
        assert r.mrr == pytest.approx(1.0)

    def test_q001_precision_at_3(self):
        ev = self._evaluator()
        r = ev.evaluate_query("q001", "espessura mínima tubulação", {"doc_A", "doc_B"})
        # top-3: [doc_A✓, doc_X✗, doc_B✓] → 2/3
        assert r.precision_at_k[3] == pytest.approx(2 / 3)

    def test_q001_hit_rate_at_1(self):
        ev = self._evaluator()
        r = ev.evaluate_query("q001", "espessura mínima tubulação", {"doc_A", "doc_B"})
        assert r.hit_rate_at_k[1] == 1.0

    def test_q002_full_miss(self):
        ev = self._evaluator()
        r = ev.evaluate_query("q002", "inspeção solda qualificação", {"doc_C"})
        # doc_C não está no corpus → MRR=0, P@5=0, HR=0
        assert r.mrr == pytest.approx(0.0)
        assert r.precision_at_k[5] == pytest.approx(0.0)
        assert r.hit_rate_at_k[5] == 0.0

    def test_collection_name_in_report(self, mock_env):
        """embedding_model no relatório deve refletir o .env."""
        cfg = Settings.from_env()
        ev = self._evaluator()
        r = ev.evaluate_query("q001", "espessura mínima tubulação", {"doc_A"})

        import datetime, numpy as np
        report = EvaluationReport(
            collection_name=cfg.collection_name,
            num_queries=1,
            k_values=[1, 3, 5],
            embedding_model=cfg.embedding_model,
            timestamp=datetime.datetime.utcnow().isoformat(),
            mean_mrr=r.mrr,
            mean_precision={k: r.precision_at_k[k] for k in [1, 3, 5]},
            mean_ndcg={k: r.ndcg_at_k[k] for k in [1, 3, 5]},
            mean_hit_rate={k: r.hit_rate_at_k[k] for k in [1, 3, 5]},
            per_query_results=[r],
        )
        assert report.collection_name == "Mota_engil_multimodal"
        assert "all-MiniLM-L6-v2" in report.embedding_model


# ---------------------------------------------------------------------------
# Smoke test standalone (python test_rag_metrics.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    import numpy as np

    print("\n" + "=" * 65)
    print("  SMOKE TEST — OfflineRAGEvaluator + MetricsReporter")
    print("  Embedding  : sentence-transformers/all-MiniLM-L6-v2  (.env)")
    print("  Collection : Mota_engil_multimodal                   (.env)")
    print("=" * 65)

    # Simula os valores do .env sem precisar do arquivo físico
    os.environ.setdefault("QDRANT_URL",      "https://b9b4.qdrant.io")
    os.environ.setdefault("QDRANT_API_KEY",  "eyJh****")
    os.environ.setdefault("COLLECTION_NAME", "Mota_engil_multimodal")
    os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    os.environ.setdefault("HF_TOKEN",        "hf_****")
    os.environ.setdefault("LLM_MODEL",       "llama3:latest")

    cfg = Settings.from_env()

    # Corpus sintético mimetizando queries reais do projeto
    MOCK_CORPUS = {
        "q001": [
            ("n115_sec4_chunk_12", 0.92), ("irr_a", 0.85),
            ("n115_sec4_chunk_13", 0.78), ("irr_b", 0.70), ("irr_c", 0.60),
        ],
        "q002": [
            ("irr_d", 0.88), ("irr_e", 0.82), ("n279_sec6_chunk_05", 0.75),
            ("irr_f", 0.65), ("irr_g", 0.55),
        ],
        "q003": [
            ("irr_h", 0.95), ("irr_i", 0.90), ("irr_j", 0.85),
            ("irr_k", 0.80), ("irr_l", 0.75),   # nenhum relevante
        ],
    }

    GT = [
        ("q001", "espessura mínima N-115 tubulação carbono",        {"n115_sec4_chunk_12", "n115_sec4_chunk_13"}),
        ("q002", "qualificação soldadores N-279 ensaios não destrutivos", {"n279_sec6_chunk_05"}),
        ("q003", "suporte guia tubulação offshore ASTM A36",          {"suporte_chunk_22"}),
    ]

    def smoke_retriever(query_text: str):
        for qid, fragment in [("q001", "N-115"), ("q002", "N-279"), ("q003", "suporte")]:
            if fragment in query_text:
                return MOCK_CORPUS[qid]
        return []

    evaluator = OfflineRAGEvaluator(retriever_fn=smoke_retriever, k_values=[1, 3, 5])

    results = []
    for qid, qtext, rel_ids in GT:
        r = evaluator.evaluate_query(qid, qtext, rel_ids)
        results.append(r)
        print(
            f"  [{r.query_id}] MRR={r.mrr:.3f} | "
            f"P@5={r.precision_at_k[5]:.3f} | "
            f"nDCG@5={r.ndcg_at_k[5]:.3f} | "
            f"HR@5={r.hit_rate_at_k[5]:.1f}"
        )

    report = EvaluationReport(
        collection_name=cfg.collection_name,
        num_queries=len(results),
        k_values=[1, 3, 5],
        embedding_model=cfg.embedding_model,
        timestamp=datetime.datetime.utcnow().isoformat(),
        mean_mrr=float(np.mean([r.mrr for r in results])),
        mean_precision={k: float(np.mean([r.precision_at_k[k] for r in results])) for k in [1, 3, 5]},
        mean_ndcg={k:      float(np.mean([r.ndcg_at_k[k]      for r in results])) for k in [1, 3, 5]},
        mean_hit_rate={k:  float(np.mean([r.hit_rate_at_k[k]  for r in results])) for k in [1, 3, 5]},
        per_query_results=results,
    )

    MetricsReporter.print_summary(report)
    print("Smoke test concluído com sucesso.")