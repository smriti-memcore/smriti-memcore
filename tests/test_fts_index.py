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
        target = make_memory("ticket YEP293 auth regression in login flow")
        noise = [make_memory(f"unrelated memory about topic {i}") for i in range(10)]
        fts.rebuild([target] + noise)
        results = fts.search("YEP293", top_k=5)
        ids = [r[0] for r in results]
        assert target.id in ids

    def test_no_results_when_term_absent(self, fts, make_memory):
        memories = [make_memory(f"topic {i} no special terms") for i in range(10)]
        fts.rebuild(memories)
        results = fts.search("YEP293", top_k=5)
        assert results == []
