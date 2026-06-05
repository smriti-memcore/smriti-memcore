"""
SMRITI Compressors — Content compression for memory optimization.

Inspired by the Headroom library (https://github.com/chopratejas/headroom),
reimplemented as lightweight, dependency-free modules.

Modules:
    json_crusher : JSON schema-aware compression (SmartCrusher-inspired)
    code_crusher : AST-based Python code compression (CodeCompressor-inspired)
    router       : ContentRouter — dispatches by Modality enum
"""

from smriti_memcore.compressors.router import ContentRouter, CompressionResult
from smriti_memcore.compressors.json_crusher import crush_json
from smriti_memcore.compressors.code_crusher import crush_code

__all__ = [
    "ContentRouter",
    "CompressionResult",
    "crush_json",
    "crush_code",
]
