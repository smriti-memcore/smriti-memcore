"""
SMRITI Compressors — Code Crusher.

An AST-based Python code compressor inspired by Headroom's CodeCompressor.
Strips non-essential code elements while preserving the structural interface
that an LLM needs to understand what the code does.

What's preserved:
  - Function/method signatures (name, args, return type)
  - Class definitions and inheritance
  - Decorator names
  - Top-level assignments and constants
  - Type aliases

What's stripped:
  - Function/method bodies (replaced with `...`)
  - Docstrings
  - Comments
  - Import statements (summarised as a count)
  - Blank lines
  - Inline type: ignore / noqa annotations
"""

from __future__ import annotations

import ast
import logging
import re
import textwrap
from typing import List, Optional

logger = logging.getLogger(__name__)


def crush_code(text: str, *, keep_bodies: bool = False) -> str:
    """
    Compress Python source code by extracting structural signatures.

    Args:
        text: Raw Python source code.
        keep_bodies: If True, keep function bodies (useful for short functions).

    Returns:
        Compressed code string, or the original text if AST parsing fails.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        logger.debug("Code crush: AST parse failed, returning as-is")
        return text

    lines: List[str] = []
    import_count = 0

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_count += 1

        elif isinstance(node, ast.ClassDef):
            lines.append(_format_class(node, keep_bodies=keep_bodies))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(_format_function(node, indent=0, keep_bodies=keep_bodies))

        elif isinstance(node, ast.Assign):
            line = _format_assign(node, text)
            if line:
                lines.append(line)

        elif isinstance(node, ast.AnnAssign):
            line = _format_ann_assign(node, text)
            if line:
                lines.append(line)

    # Build output
    parts: List[str] = []
    if import_count > 0:
        parts.append(f"# ({import_count} imports omitted)")
        parts.append("")

    parts.extend(lines)

    result = "\n".join(parts).strip()

    # If the result is empty (e.g., file with only imports), return a summary
    if not result:
        return f"# Python module: {import_count} imports, no classes/functions"

    return result


def _format_class(node: ast.ClassDef, *, keep_bodies: bool = False) -> str:
    """Format a class definition with its methods."""
    parts: List[str] = []

    # Decorators
    for decorator in node.decorator_list:
        parts.append(f"@{_unparse_safe(decorator)}")

    # Class header
    bases = [_unparse_safe(b) for b in node.bases]
    keywords = [f"{kw.arg}={_unparse_safe(kw.value)}" for kw in node.keywords if kw.arg]
    all_args = bases + keywords
    if all_args:
        parts.append(f"class {node.name}({', '.join(all_args)}):")
    else:
        parts.append(f"class {node.name}:")

    # Class body — extract methods and class-level assignments
    has_content = False
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(_format_function(child, indent=1, keep_bodies=keep_bodies))
            has_content = True
        elif isinstance(child, ast.Assign):
            line = _format_assign(child, "")
            if line:
                parts.append(f"    {line}")
                has_content = True
        elif isinstance(child, ast.AnnAssign):
            line = _format_ann_assign(child, "")
            if line:
                parts.append(f"    {line}")
                has_content = True

    if not has_content:
        parts.append("    ...")

    parts.append("")  # Blank line after class
    return "\n".join(parts)


def _format_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    indent: int = 0,
    *,
    keep_bodies: bool = False,
) -> str:
    """Format a function/method signature."""
    prefix = "    " * indent
    parts: List[str] = []

    # Decorators
    for decorator in node.decorator_list:
        parts.append(f"{prefix}@{_unparse_safe(decorator)}")

    # Function signature
    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    args_str = _format_args(node.args)
    returns = f" -> {_unparse_safe(node.returns)}" if node.returns else ""

    sig = f"{prefix}{async_prefix}def {node.name}({args_str}){returns}:"

    # Wrap long signatures
    if len(sig) > 88:
        sig = _wrap_signature(prefix, async_prefix, node.name, node.args, returns)

    parts.append(sig)

    if keep_bodies:
        # Include body (for short functions)
        body_lines = _get_body_source(node)
        if body_lines:
            for line in body_lines:
                parts.append(f"{prefix}    {line}")
        else:
            parts.append(f"{prefix}    ...")
    else:
        parts.append(f"{prefix}    ...")

    return "\n".join(parts)


def _format_args(args: ast.arguments) -> str:
    """Format function arguments into a string."""
    parts: List[str] = []

    # Regular args
    num_defaults = len(args.defaults)
    num_args = len(args.args)
    for i, arg in enumerate(args.args):
        ann = f": {_unparse_safe(arg.annotation)}" if arg.annotation else ""
        default_idx = i - (num_args - num_defaults)
        if default_idx >= 0:
            default = f" = {_unparse_safe(args.defaults[default_idx])}"
        else:
            default = ""
        parts.append(f"{arg.arg}{ann}{default}")

    # *args
    if args.vararg:
        ann = f": {_unparse_safe(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{args.vararg.arg}{ann}")
    elif args.kwonlyargs:
        parts.append("*")

    # Keyword-only args
    for i, arg in enumerate(args.kwonlyargs):
        ann = f": {_unparse_safe(arg.annotation)}" if arg.annotation else ""
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            default = f" = {_unparse_safe(args.kw_defaults[i])}"
        else:
            default = ""
        parts.append(f"{arg.arg}{ann}{default}")

    # **kwargs
    if args.kwarg:
        ann = f": {_unparse_safe(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{args.kwarg.arg}{ann}")

    return ", ".join(parts)


def _wrap_signature(
    prefix: str, async_prefix: str, name: str, args: ast.arguments, returns: str
) -> str:
    """Wrap a long function signature across multiple lines."""
    header = f"{prefix}{async_prefix}def {name}("
    arg_str = _format_args(args)
    arg_parts = [a.strip() for a in arg_str.split(",")]
    inner_prefix = prefix + "    "

    lines = [header]
    for i, part in enumerate(arg_parts):
        comma = "," if i < len(arg_parts) - 1 else ""
        lines.append(f"{inner_prefix}{part}{comma}")
    lines.append(f"{prefix}){returns}:")

    return "\n".join(lines)


def _format_assign(node: ast.Assign, source: str) -> Optional[str]:
    """Format a top-level assignment."""
    try:
        targets = [_unparse_safe(t) for t in node.targets]
        value_str = _unparse_safe(node.value)
        # Truncate long values
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."
        return f"{' = '.join(targets)} = {value_str}"
    except Exception:
        return None


def _format_ann_assign(node: ast.AnnAssign, source: str) -> Optional[str]:
    """Format an annotated assignment (e.g., type aliases)."""
    try:
        target = _unparse_safe(node.target)
        ann = _unparse_safe(node.annotation)
        if node.value:
            value_str = _unparse_safe(node.value)
            if len(value_str) > 60:
                value_str = value_str[:57] + "..."
            return f"{target}: {ann} = {value_str}"
        return f"{target}: {ann}"
    except Exception:
        return None


def _get_body_source(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
    """Extract the body of a function as source lines (excluding docstring)."""
    body = node.body

    # Skip docstring
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Constant, ast.Str)):
        body = body[1:]

    if not body:
        return []

    lines = []
    for stmt in body:
        try:
            lines.append(_unparse_safe(stmt))
        except Exception:
            lines.append("...")

    return lines


def _unparse_safe(node: Any) -> str:
    """Safely unparse an AST node to source code."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


# Make type hint work for Python 3.9 compatibility
from typing import Any
