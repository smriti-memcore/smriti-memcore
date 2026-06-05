"""
SMRITI Compressors — JSON Crusher.

A schema-aware JSON compressor inspired by Headroom's SmartCrusher.
Reduces JSON payloads by removing redundant structure while preserving
the key-value information that matters for LLM comprehension.

Strategies:
  1. Truncate homogeneous arrays to first N items + count annotation.
  2. Collapse deeply nested objects beyond a depth limit.
  3. Strip boilerplate keys (e.g., "@context", "$schema", "links", "href").
  4. Inline short leaf objects onto a single line.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Keys commonly found in API responses that add noise, not signal
_BOILERPLATE_KEYS: Set[str] = {
    "@context", "$schema", "$id", "$ref",
    "_links", "_embedded", "links", "href", "self",
    "pagination", "paging", "next_page", "prev_page",
    "total_pages", "page_size", "has_more",
    "request_id", "trace_id", "correlation_id",
    "x-request-id", "x-trace-id",
}

# Maximum items to keep from a homogeneous array
_DEFAULT_ARRAY_LIMIT = 3

# Maximum nesting depth before collapsing
_DEFAULT_DEPTH_LIMIT = 5


def crush_json(
    text: str,
    *,
    array_limit: int = _DEFAULT_ARRAY_LIMIT,
    depth_limit: int = _DEFAULT_DEPTH_LIMIT,
    strip_boilerplate: bool = True,
    boilerplate_keys: Optional[Set[str]] = None,
) -> str:
    """
    Compress a JSON string by removing structural redundancy.

    Args:
        text: Raw JSON string.
        array_limit: Max items kept from homogeneous arrays (rest summarised).
        depth_limit: Max nesting depth before objects are collapsed to summaries.
        strip_boilerplate: Whether to remove common API boilerplate keys.
        boilerplate_keys: Custom set of keys to strip (merged with defaults).

    Returns:
        Compressed JSON string, or the original text if parsing fails.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.debug("JSON crush: input is not valid JSON, returning as-is")
        return text

    keys_to_strip = _BOILERPLATE_KEYS.copy()
    if boilerplate_keys:
        keys_to_strip |= boilerplate_keys

    crushed = _crush_node(
        data,
        depth=0,
        array_limit=array_limit,
        depth_limit=depth_limit,
        strip_boilerplate=strip_boilerplate,
        boilerplate_keys=keys_to_strip,
    )

    return json.dumps(crushed, indent=2, ensure_ascii=False, default=str)


def _crush_node(
    node: Any,
    depth: int,
    array_limit: int,
    depth_limit: int,
    strip_boilerplate: bool,
    boilerplate_keys: Set[str],
) -> Any:
    """Recursively compress a JSON node."""

    # Depth limit — collapse to summary
    if depth >= depth_limit:
        return _summarize_node(node)

    if isinstance(node, dict):
        result = {}
        for key, value in node.items():
            # Strip boilerplate keys
            if strip_boilerplate and key.lower() in {k.lower() for k in boilerplate_keys}:
                continue

            result[key] = _crush_node(
                value,
                depth=depth + 1,
                array_limit=array_limit,
                depth_limit=depth_limit,
                strip_boilerplate=strip_boilerplate,
                boilerplate_keys=boilerplate_keys,
            )
        return result

    elif isinstance(node, list):
        if not node:
            return []

        # Check if array is homogeneous (all items have the same structure)
        if len(node) > array_limit and _is_homogeneous(node):
            # Keep first N items + count annotation
            crushed_items = [
                _crush_node(
                    item,
                    depth=depth + 1,
                    array_limit=array_limit,
                    depth_limit=depth_limit,
                    strip_boilerplate=strip_boilerplate,
                    boilerplate_keys=boilerplate_keys,
                )
                for item in node[:array_limit]
            ]
            crushed_items.append(f"... ({len(node) - array_limit} more items, {len(node)} total)")
            return crushed_items
        else:
            # Non-homogeneous or short — process all
            return [
                _crush_node(
                    item,
                    depth=depth + 1,
                    array_limit=array_limit,
                    depth_limit=depth_limit,
                    strip_boilerplate=strip_boilerplate,
                    boilerplate_keys=boilerplate_keys,
                )
                for item in node
            ]

    else:
        # Primitive — return as-is
        return node


def _is_homogeneous(items: List[Any]) -> bool:
    """Check if all items in a list have the same structural type."""
    if not items:
        return True

    first = items[0]

    # All dicts with the same keys
    if isinstance(first, dict):
        first_keys = set(first.keys())
        return all(
            isinstance(item, dict) and set(item.keys()) == first_keys
            for item in items[1:min(5, len(items))]  # Check first 5 for efficiency
        )

    # All same primitive type
    first_type = type(first)
    return all(isinstance(item, first_type) for item in items[1:min(5, len(items))])


def _summarize_node(node: Any) -> str:
    """Create a concise summary for a deeply nested node."""
    if isinstance(node, dict):
        keys = list(node.keys())
        if len(keys) <= 3:
            return f"{{{', '.join(keys)}}}"
        return f"{{{', '.join(keys[:3])}, ... ({len(keys)} keys)}}"
    elif isinstance(node, list):
        return f"[... ({len(node)} items)]"
    else:
        s = str(node)
        if len(s) > 80:
            return s[:77] + "..."
        return s
