import json
from smriti_memcore.models import Modality
from smriti_memcore.compressors.router import ContentRouter
from smriti_memcore.compressors.json_crusher import crush_json
from smriti_memcore.compressors.code_crusher import crush_code


def test_json_crusher_strips_boilerplate():
    """Test that boilerplate keys are removed."""
    raw = json.dumps({
        "data": "important",
        "@context": "noise",
        "links": [{"href": "url"}],
        "pagination": {"next": "page2"}
    })
    
    crushed = crush_json(raw)
    parsed = json.loads(crushed)
    
    assert "data" in parsed
    assert "@context" not in parsed
    assert "links" not in parsed
    assert "pagination" not in parsed


def test_json_crusher_truncates_arrays():
    """Test that homogeneous arrays are truncated to array_limit."""
    # 5 identical objects
    raw = json.dumps([{"id": i, "val": "x"} for i in range(5)])
    
    crushed = crush_json(raw, array_limit=2)
    parsed = json.loads(crushed)
    
    assert len(parsed) == 3  # 2 items + 1 summary string
    assert parsed[0]["id"] == 0
    assert parsed[1]["id"] == 1
    assert "... (3 more items" in parsed[2]


def test_code_crusher_strips_bodies():
    """Test that function bodies and docstrings are stripped."""
    raw = '''
import os
import sys

def calculate_stats(data: list) -> dict:
    """
    Calculate mean and variance.
    This is a long docstring.
    """
    if not data:
        return {}
    mean = sum(data) / len(data)
    # Return the result
    return {"mean": mean}

class Engine:
    def __init__(self, size: int = 10):
        self.size = size
        
    def start(self):
        print("Vroom")
'''
    crushed = crush_code(raw)
    
    # Imports summarized
    assert "# (2 imports omitted)" in crushed
    # Signatures preserved
    assert "def calculate_stats(data: list) -> dict:" in crushed
    assert "class Engine:" in crushed
    assert "def __init__(self, size: int = 10):" in crushed
    # Bodies replaced with ...
    assert "return {" not in crushed
    assert "print" not in crushed
    assert "mean = " not in crushed
    # Docstrings removed
    assert "Calculate mean and variance" not in crushed


def test_content_router_dispatch():
    """Test that the router sends content to the right compressor and handles thresholds."""
    router = ContentRouter(min_compression_ratio=0.9, min_content_length=50)
    
    # 1. CODE routing
    code = "def foo():\n" + "    pass\n" * 20
    res = router.compress(code, Modality.CODE)
    assert res.compressor_used == "code_crusher"
    assert res.compressed is not None
    assert res.ratio < 0.9
    
    # 2. JSON routing
    data = json.dumps([{"id": i} for i in range(20)])
    res = router.compress(data, Modality.STRUCTURED)
    assert res.compressor_used == "json_crusher"
    assert res.compressed is not None
    
    # 3. TEXT routing (passthrough)
    text = "This is a long prose text. " * 10
    res = router.compress(text, Modality.TEXT)
    assert res.compressor_used == "none"
    assert res.compressed is None
    assert res.skipped_reason == "modality_not_compressible"
    
    # 4. Too short
    short = "def a(): pass"
    res = router.compress(short, Modality.CODE)
    assert res.compressor_used == "none"
    assert res.compressed is None
    assert res.skipped_reason == "below_min_length"
