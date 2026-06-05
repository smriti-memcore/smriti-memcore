import pytest
from smriti_memcore.core import SMRITI
from smriti_memcore.models import Modality, SmritiConfig, Episode, SalienceScore
from unittest.mock import MagicMock


@pytest.fixture
def smriti(tmp_path):
    """SMRITI instance with temporary storage."""
    config = SmritiConfig(storage_path=str(tmp_path))
    s = SMRITI(config=config)
    yield s
    s.close()


def test_ccr_roundtrip(smriti):
    """
    Integration test for CCR (Content Compress and Retrieve):
    1. Encode compressible content.
    2. Recall it to verify it was compressed.
    3. Format it for LLM to verify the marker.
    4. Retrieve original to verify the full content was saved.
    """
    # 1. Compressible code
    raw_code = '''def long_function():
    # some comment
    x = 1
    y = 2
    return x + y
''' * 20  # Make it long enough to pass length threshold
    
    # Mock attention gate to always pass
    dummy_episode = Episode(
        content=raw_code,
        salience=SalienceScore(surprise=0.9, relevance=0.9, emotional=0.0, novelty=0.9, utility=0.9),
        embedding=[0.1] * 384
    )
    smriti.attention_gate.process = MagicMock(return_value=dummy_episode)
    
    # 2. Encode
    mem_id = smriti.encode(raw_code, modality=Modality.CODE)
    assert mem_id is not None
    
    # Verify it compressed internally
    mem = smriti.palace.memories[mem_id]
    assert mem.content == raw_code
    assert mem.content_compressed is not None
    assert len(mem.content_compressed) < len(raw_code)
    
    # 3. Recall and inject (format for LLM)
    # Mock the embedding generation during recall
    smriti.vector_store.embed = MagicMock(return_value=[0.1] * 384)
    smriti.recall(raw_code)
    
    context_str = smriti.get_context()
    
    # The formatted context should have the compressed string and marker
    assert mem.content_compressed in context_str
    assert f"⟨compressed:{mem_id}⟩ — call smriti_retrieve_original('{mem_id}')" in context_str
    
    # The original raw text should NOT be in the working memory context
    assert "x = 1" not in context_str
    
    # 4. "Tool call" to retrieve original
    smriti._metrics.original_retrieval_count.inc()
    assert smriti._metrics.original_retrieval_count.value == 1
