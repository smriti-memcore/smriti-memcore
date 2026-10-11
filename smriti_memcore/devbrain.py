"""
DevBrain: Developer Agent Persistent Memory & Workspace Integration for SMRITI.

Provides:
- Workspace initialization (smriti init): Scaffolding .cursorrules, CLAUDE.md, .agents, and MCP configs.
- Git commit hooks (smriti hook install/uninstall): Non-blocking capture of commit context.
- Workspace-scoped memory rooms in the Semantic Palace.
"""

import json
import logging
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

HOOK_MARKER = "# SMRITI-DEVBRAIN-HOOK"

GIT_POST_COMMIT_SCRIPT = f"""
{HOOK_MARKER} - Auto-records commits into SMRITI DevBrain
(
  if command -v python3 >/dev/null 2>&1; then
    SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    MSG=$(git log -1 --pretty=%B 2>/dev/null | tr '\\n' ' ' | head -c 300)
    FILES=$(git diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null | tr '\\n' ', ' | head -c 200)
    REPO=$(basename "$(git rev-parse --show-toplevel 2>/dev/null || pwd)")
    
    python3 -c "
import sys
try:
    from smriti_memcore import SMRITI
    from smriti_memcore.models import MemorySource, Modality
    smriti = SMRITI()
    repo, sha, msg, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    content = f'Git Commit in [{{repo}}] ({{sha}}): {{msg}}. Touched: {{files}}'
    smriti.encode(content, source=MemorySource.DIRECT, modality=Modality.CODE, force=True)
    smriti.close()
except Exception:
    pass
" "$REPO" "$SHA" "$MSG" "$FILES" >/dev/null 2>&1 &
  fi
) >/dev/null 2>&1 &
# END-SMRITI-DEVBRAIN-HOOK
"""


def find_repo_root(start_path: Optional[str] = None) -> Tuple[Path, str]:
    """Find the root directory of the current git repository or directory."""
    target = Path(start_path).expanduser().resolve() if start_path else Path.cwd().resolve()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(target),
            capture_output=True,
            text=True,
            check=True,
        )
        repo_root = Path(res.stdout.strip())
        repo_name = repo_root.name
        return repo_root, repo_name
    except Exception:
        return target, target.name


def ensure_workspace_room(repo_name: str, storage_path: Optional[str] = None) -> str:
    """Ensure a dedicated Semantic Room exists for this workspace in SMRITI."""
    try:
        from smriti_memcore.core import SMRITI
        from smriti_memcore.models import SmritiConfig

        cfg = SmritiConfig(storage_path=os.path.expanduser(storage_path)) if storage_path else None
        smriti = SMRITI(config=cfg)
        topic = f"repo:{repo_name}"

        # Check existing rooms
        for room_id, room in smriti.palace.rooms.items():
            if room.topic == topic:
                smriti.close()
                return room_id

        # Create room if not present
        new_room = smriti.palace.create_room(topic=topic)
        smriti.save()
        room_id = new_room.id
        smriti.close()
        return room_id
    except Exception as e:
        logger.warning(f"Could not connect to SMRITI to create room: {e}")
        return f"repo:{repo_name}"


def init_workspace(
    target_path: Optional[str] = None,
    room_name: Optional[str] = None,
    storage_path: str = "~/.smriti/global",
    install_hook: bool = True,
) -> Dict[str, Any]:
    """
    Initialize Smriti DevBrain for the workspace.
    Generates .cursorrules, .cursor/rules/smriti.mdc, CLAUDE.md, and project MCP configs.
    """
    repo_root, repo_default_name = find_repo_root(target_path)
    active_name = room_name or repo_default_name
    room_topic = f"repo:{active_name}"

    # 1. Ensure Room in Smriti
    room_id = ensure_workspace_room(active_name, storage_path=storage_path)

    created_files = []

    # 2. Scaffolding .cursorrules
    cursorrules_path = repo_root / ".cursorrules"
    cursor_content = f"""# SMRITI DevBrain: Persistent Memory for Cursor
# Room: {room_topic} (ID: {room_id})

You are connected to SMRITI Long-Term Memory.
Before answering any user question or starting any coding task:
1. ALWAYS call `smriti_recall` with 2-3 keywords relevant to the user request.
2. If this workspace has prior architecture decisions, conventions, or solved bugs, incorporate them.
3. When you finalize a key architecture choice, resolve an edge case bug, or learn a user preference:
   Call `smriti_encode` with clear, concise findings so future sessions remember it.
4. Keep memories concise and factual.
"""
    if not cursorrules_path.exists():
        cursorrules_path.write_text(cursor_content, encoding="utf-8")
        created_files.append(".cursorrules")

    # 2b. Modern Cursor MDC rule (.cursor/rules/smriti.mdc)
    cursor_rules_dir = repo_root / ".cursor" / "rules"
    cursor_rules_dir.mkdir(parents=True, exist_ok=True)
    mdc_path = cursor_rules_dir / "smriti.mdc"
    if not mdc_path.exists():
        mdc_content = f"""---
description: SMRITI DevBrain Persistent Memory rules for coding agent
globs: *
alwaysApply: true
---

# SMRITI Memory Protocol

- Call `smriti_recall` on task start with keywords related to the task.
- Call `smriti_encode` when an architectural decision is finalized, a bug is solved, or a pattern is established.
- Current Workspace Room: `{room_topic}`
"""
        mdc_path.write_text(mdc_content, encoding="utf-8")
        created_files.append(".cursor/rules/smriti.mdc")

    # 3. Project-level .cursor/mcp.json (if not present)
    cursor_mcp = repo_root / ".cursor" / "mcp.json"
    if not cursor_mcp.exists():
        py_bin = (
            Path.home() / ".smriti" / "venv" / "bin" / "python3"
            if (Path.home() / ".smriti" / "venv" / "bin" / "python3").exists()
            else sys.executable
        )
        mcp_data = {
            "mcpServers": {
                "smriti-memory": {
                    "command": str(py_bin),
                    "args": ["-m", "smriti_memcore.integrations.mcp_server"],
                    "env": {
                        "SMRITI_STORAGE_PATH": os.path.expanduser(storage_path),
                        "SMRITI_LLM_MODEL": os.environ.get("SMRITI_LLM_MODEL", "mistral"),
                        "SMRITI_LLM_API_KEY": os.environ.get("SMRITI_LLM_API_KEY", ""),
                    },
                }
            }
        }
        cursor_mcp.write_text(json.dumps(mcp_data, indent=2), encoding="utf-8")
        created_files.append(".cursor/mcp.json")

    # 4. Scaffolding CLAUDE.md
    claude_md = repo_root / "CLAUDE.md"
    claude_section = f"""
## SMRITI DevBrain Memory Integration
- **Workspace Room**: `{room_topic}`
- **On every message**: Call `smriti_recall` with 2-3 keywords before responding.
- **On key milestones**: Call `smriti_encode` when decisions are made, bugs are resolved, or constraints are introduced.
"""
    if not claude_md.exists():
        claude_md.write_text(f"# {active_name} Development Guidelines\n{claude_section}", encoding="utf-8")
        created_files.append("CLAUDE.md")
    else:
        existing = claude_md.read_text(encoding="utf-8")
        if "SMRITI DevBrain Memory" not in existing:
            claude_md.write_text(f"{existing}\n{claude_section}", encoding="utf-8")
            created_files.append("CLAUDE.md (updated)")

    # 5. Scaffolding .agents/rules/smriti.md (Gemini / Antigravity)
    agents_dir = repo_root / ".agents" / "rules"
    agents_dir.mkdir(parents=True, exist_ok=True)
    agent_rule = agents_dir / "smriti.md"
    if not agent_rule.exists():
        agent_rule.write_text(
            f"""# Smriti DevBrain Rules
- Active Workspace: `{room_topic}`
- Recall memory on session startup using `smriti_recall`.
- Encode decisions and bug solutions using `smriti_encode`.
""",
            encoding="utf-8",
        )
        created_files.append(".agents/rules/smriti.md")

    # 6. Optional Git Hook
    hook_installed = False
    if install_hook:
        hook_installed = install_git_hook(repo_root)

    return {
        "repo_name": active_name,
        "repo_root": str(repo_root),
        "room_topic": room_topic,
        "room_id": room_id,
        "created_files": created_files,
        "git_hook_installed": hook_installed,
    }


def install_git_hook(repo_root: Optional[Path] = None) -> bool:
    """Install git post-commit hook for automatic memory capture."""
    root = repo_root or find_repo_root()[0]
    git_dir = root / ".git"
    if not git_dir.exists() or not git_dir.is_dir():
        return False

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    post_commit = hooks_dir / "post-commit"

    if post_commit.exists():
        content = post_commit.read_text(encoding="utf-8")
        if HOOK_MARKER in content:
            return True  # Already installed
        new_content = content + "\n" + GIT_POST_COMMIT_SCRIPT
    else:
        new_content = "#!/usr/bin/env bash\n" + GIT_POST_COMMIT_SCRIPT

    post_commit.write_text(new_content, encoding="utf-8")
    # Make executable
    st = post_commit.stat()
    post_commit.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return True


def uninstall_git_hook(repo_root: Optional[Path] = None) -> bool:
    """Uninstall git post-commit hook."""
    root = repo_root or find_repo_root()[0]
    post_commit = root / ".git" / "hooks" / "post-commit"
    if not post_commit.exists():
        return False

    content = post_commit.read_text(encoding="utf-8")
    if HOOK_MARKER not in content:
        return False

    # Remove the block
    lines = content.splitlines()
    new_lines = []
    in_block = False
    for line in lines:
        if HOOK_MARKER in line:
            in_block = True
            continue
        if "END-SMRITI-DEVBRAIN-HOOK" in line:
            in_block = False
            continue
        if not in_block:
            new_lines.append(line)

    cleaned = "\n".join(new_lines).strip()
    if cleaned and cleaned != "#!/usr/bin/env bash":
        post_commit.write_text(cleaned + "\n", encoding="utf-8")
    else:
        post_commit.unlink()
    return True
