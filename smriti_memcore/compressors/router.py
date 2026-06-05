"""
SMRITI Compressors — Content Router.

Routes incoming content to the appropriate compressor based on the
SMRITI Modality enum. Applies a minimum compression ratio threshold
to avoid storing two copies when compression is negligible.

Inspired by Headroom's CCR (Content Compress and Retrieve) architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from smriti_memcore.models import Modality
from smriti_memcore.compressors.json_crusher import crush_json
from smriti_memcore.compressors.code_crusher import crush_code

logger = logging.getLogger(__name__)

# Minimum compression ratio to justify storing a compressed copy.
# If compressed_len / original_len > this threshold, compression is skipped.
_MIN_COMPRESSION_RATIO = 0.80

# Minimum content length to attempt compression.
# Short content doesn't benefit from compression.
_MIN_CONTENT_LENGTH = 200


@dataclass
class CompressionResult:
    """Result of a compression attempt."""
    compressed: Optional[str]   # Compressed text, or None if skipped
    original_length: int        # Length of original content (chars)
    compressed_length: int      # Length of compressed content (chars), 0 if skipped
    ratio: float                # compressed_length / original_length (0.0 if skipped)
    compressor_used: str        # "json_crusher", "code_crusher", "none"
    skipped_reason: str         # "" if compressed, reason if skipped


class ContentRouter:
    """
    Routes content to the appropriate compressor based on Modality.

    Routing table:
        CODE       → CodeCrusher (AST-based Python compressor)
        STRUCTURED → JsonCrusher (JSON schema-aware pruner)
        TEXT       → Passthrough (no compression in Phase 1)
        IMAGE      → Passthrough

    Compression is skipped if:
        - Content is shorter than MIN_CONTENT_LENGTH (200 chars)
        - Compression ratio exceeds MIN_COMPRESSION_RATIO (80%)
        - The compressor fails (falls back to passthrough)
    """

    def __init__(
        self,
        min_compression_ratio: float = _MIN_COMPRESSION_RATIO,
        min_content_length: int = _MIN_CONTENT_LENGTH,
    ):
        self.min_compression_ratio = min_compression_ratio
        self.min_content_length = min_content_length

    def compress(
        self,
        content: str,
        modality: Modality,
    ) -> CompressionResult:
        """
        Compress content using the appropriate algorithm for its modality.

        Args:
            content: Raw content text.
            modality: SMRITI Modality enum value.

        Returns:
            CompressionResult with compressed text (or None if skipped).
        """
        original_length = len(content)

        # Skip short content — not worth compressing
        if original_length < self.min_content_length:
            return CompressionResult(
                compressed=None,
                original_length=original_length,
                compressed_length=0,
                ratio=0.0,
                compressor_used="none",
                skipped_reason="below_min_length",
            )

        # Route to compressor
        if modality == Modality.CODE:
            return self._try_compress(content, original_length, crush_code, "code_crusher")

        elif modality == Modality.STRUCTURED:
            return self._try_compress(content, original_length, crush_json, "json_crusher")

        else:
            # TEXT, IMAGE — passthrough (no compression in Phase 1)
            return CompressionResult(
                compressed=None,
                original_length=original_length,
                compressed_length=0,
                ratio=0.0,
                compressor_used="none",
                skipped_reason="modality_not_compressible",
            )

    def _try_compress(
        self,
        content: str,
        original_length: int,
        compress_fn,
        compressor_name: str,
    ) -> CompressionResult:
        """Attempt compression and check ratio threshold."""
        try:
            compressed = compress_fn(content)
        except Exception as e:
            logger.warning(f"{compressor_name} failed: {e}")
            return CompressionResult(
                compressed=None,
                original_length=original_length,
                compressed_length=0,
                ratio=0.0,
                compressor_used=compressor_name,
                skipped_reason=f"compressor_error: {e}",
            )

        compressed_length = len(compressed)
        ratio = compressed_length / original_length if original_length > 0 else 1.0

        # Check if compression is worth it
        if ratio > self.min_compression_ratio:
            logger.debug(
                f"{compressor_name}: ratio {ratio:.2f} above threshold "
                f"{self.min_compression_ratio}, skipping"
            )
            return CompressionResult(
                compressed=None,
                original_length=original_length,
                compressed_length=compressed_length,
                ratio=ratio,
                compressor_used=compressor_name,
                skipped_reason="below_threshold",
            )

        logger.info(
            f"{compressor_name}: {original_length} → {compressed_length} chars "
            f"({ratio:.1%} of original)"
        )

        return CompressionResult(
            compressed=compressed,
            original_length=original_length,
            compressed_length=compressed_length,
            ratio=ratio,
            compressor_used=compressor_name,
            skipped_reason="",
        )
