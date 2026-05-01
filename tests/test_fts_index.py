import pytest
from smriti_memcore.fts_index import FTSIndex


@pytest.fixture
def fts():
    index = FTSIndex(":memory:")
    yield index
    index.close()


class TestFTSRebuildIdempotency:
    def test_rebuild_twice_same_row_count(self, fts, make_memory):
        memories = [make_memory(f"topic {i}") for i in range(5)]
        fts.rebuild(memories)
        assert fts.needs_rebuild(5) is False
        fts.rebuild(memories)
        assert fts.needs_rebuild(5) is False

    def test_search_identical_after_rebuild(self, fts, make_memory):
        memories = [make_memory(f"topic {i}") for i in range(5)]
        fts.rebuild(memories)
        results1 = [r[0] for r in fts.search("topic")]
        fts.rebuild(memories)
        results2 = [r[0] for r in fts.search("topic")]
        assert results1 == results2


class TestFTSExactTermSearch:
    def test_exact_term_appears_in_results(self, fts, make_memory):
        target = make_memory("ticket YEP-293 auth regression in login flow")
        noise = [make_memory(f"unrelated memory about topic {i}") for i in range(10)]
        fts.rebuild([target] + noise)
        results = fts.search("YEP-293", top_k=5)
        ids = [r[0] for r in results]
        assert target.id in ids

    def test_no_results_when_term_absent(self, fts, make_memory):
        memories = [make_memory(f"topic {i} no special terms") for i in range(10)]
        fts.rebuild(memories)
        results = fts.search("YEP-293", top_k=5)
        assert results == []


class TestRRFMerge:
    def _engine(self, palace, vector_store, working_memory):
        from smriti_memcore.models import SmritiConfig
        from smriti_memcore.retrieval import RetrievalEngine
        return RetrievalEngine(
            palace=palace, working_memory=working_memory,
            vector_store=vector_store, config=SmritiConfig(),
        )

    def test_fts_only_memory_in_merged_pool(
        self, palace, vector_store, working_memory, make_memory
    ):
        engine = self._engine(palace, vector_store, working_memory)
        fts_only = make_memory("fts-only memory")
        vector_mem = make_memory("vector memory")
        palace.place_memory(vector_mem)

        merged = engine._rrf_merge(
            vector_candidates=[vector_mem],
            fts_results=[(fts_only.id, -1.0)],
            pool_size=10,
        )
        assert fts_only.id in merged

    def test_both_list_memory_scores_higher(
        self, palace, vector_store, working_memory, make_memory
    ):
        engine = self._engine(palace, vector_store, working_memory)
        both = make_memory("in both lists")
        fts_only = make_memory("in fts only")
        palace.place_memory(both)

        merged = engine._rrf_merge(
            vector_candidates=[both],
            fts_results=[(both.id, -1.0), (fts_only.id, -2.0)],
            pool_size=10,
        )
        assert merged.index(both.id) < merged.index(fts_only.id)


class TestHybridSearchEndToEnd:
    def test_exact_term_found_in_hybrid_results(self, tmp_dir):
        from smriti_memcore.core import SMRITI
        from smriti_memcore.models import MemorySource, SmritiConfig

        config = SmritiConfig(storage_path=tmp_dir)
        smriti = SMRITI(config=config)

        smriti.encode(
            "ticket YEP-293 causes auth regression in login flow",
            source=MemorySource.USER_STATED,
            use_llm=False,
        )
        for i in range(10):
            smriti.encode(
                f"unrelated topic about thing number {i}",
                source=MemorySource.USER_STATED,
                use_llm=False,
            )

        results = smriti.recall("YEP-293", top_k=5)
        assert any("YEP-293" in m.content for m in results), \
            "Hybrid search must find YEP-293 in top-5"
        smriti.close()
