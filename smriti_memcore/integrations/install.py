"""
SMRITI Universal MCP Installer for AI Coding Agents and IDEs.

Supports:
- Claude Code & Claude Desktop
- Cursor IDE
- Windsurf / Cascade
- Google Gemini & Codex (Antigravity)
- Continue.dev (VS Code & JetBrains)
- Zed Editor
- Roo Code & Cline
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_python_bin(custom_python: Optional[str] = None) -> str:
    """Find the best Python executable for running the MCP server."""
    if custom_python:
        p = Path(custom_python).expanduser().resolve()
        if p.exists() and os.access(p, os.X_OK):
            return str(p)

    smriti_venv_py = Path.home() / ".smriti" / "venv" / "bin" / "python3"
    if smriti_venv_py.exists() and os.access(smriti_venv_py, os.X_OK):
        return str(smriti_venv_py)

    return sys.executable


def patch_json_config(file_path: Path, patcher: Callable[[Dict[str, Any]], bool]) -> bool:
    """
    Safely load, patch, and save a JSON configuration file.
    Preserves other existing servers and keys.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}

        modified = patcher(data)
        if modified:
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return modified
    except Exception as e:
        logger.warning(f"Failed to patch {file_path}: {e}")
        return False


# ── Client Handlers ──────────────────────────────────────────────────────────

class ClientConfigurator:
    def __init__(self, python_bin: str, storage_path: str, llm_model: str, llm_api_key: str, obsidian_path: str = ""):
        self.python_bin = python_bin
        self.storage_path = os.path.expanduser(storage_path)
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.obsidian_path = os.path.expanduser(obsidian_path) if obsidian_path else ""

    def _get_env(self) -> Dict[str, str]:
        env = {
            "PYTHONPATH": "",
            "SMRITI_STORAGE_PATH": self.storage_path,
            "SMRITI_LLM_MODEL": self.llm_model,
            "SMRITI_LLM_API_KEY": self.llm_api_key,
        }
        if self.obsidian_path:
            env["SMRITI_OBSIDIAN_PATH"] = self.obsidian_path
        return env

    def _std_server_def(self) -> Dict[str, Any]:
        return {
            "command": self.python_bin,
            "args": ["-m", "smriti_memcore.integrations.mcp_server"],
            "env": self._get_env(),
        }

    # 1. Claude Code
    def configure_claude_code(self) -> List[str]:
        modified = []
        claude_json = Path.home() / ".claude.json"

        def patcher(cfg: Dict[str, Any]) -> bool:
            servers = cfg.setdefault("mcpServers", {})
            servers["smriti"] = self._std_server_def()
            return True

        if patch_json_config(claude_json, patcher):
            modified.append(str(claude_json))

        # Instructions in CLAUDE.md
        claude_md = Path.home() / ".claude" / "CLAUDE.md"
        claude_md.parent.mkdir(parents=True, exist_ok=True)
        rule_text = """
## SMRITI Memory
- On every message: Call `smriti_recall` with 2-3 keywords before responding.
- Call `smriti_encode` when a decision is made, a bug is solved, or new context emerges.
"""
        existing = claude_md.read_text(encoding="utf-8") if claude_md.exists() else "# Global Claude Instructions\n"
        if "SMRITI Memory" not in existing:
            claude_md.write_text(existing + "\n" + rule_text, encoding="utf-8")
            modified.append(str(claude_md))

        # Hooks in settings.json
        settings_path = Path.home() / ".claude" / "settings.json"
        def hook_patcher(cfg: Dict[str, Any]) -> bool:
            hooks = cfg.setdefault("hooks", {})
            ups = hooks.setdefault("UserPromptSubmit", [])
            hook_exists = any("smriti_recall" in str(h) for h in ups)
            if not hook_exists:
                ups.append({
                    "matcher": "",
                    "hooks": [{
                        "type": "command",
                        "command": "echo '{\"hookSpecificOutput\": {\"hookEventName\": \"UserPromptSubmit\", \"additionalContext\": \"SMRITI MEMORY: Call smriti_recall with 2-3 keywords from the user message before answering.\"}}'",
                        "statusMessage": "Loading SMRITI memories..."
                    }]
                })
                return True
            return False

        if patch_json_config(settings_path, hook_patcher):
            modified.append(str(settings_path))

        return modified

    # 2. Claude Desktop
    def configure_claude_desktop(self) -> List[str]:
        plat = sys.platform
        paths = []
        if plat == "darwin":
            paths.append(Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json")
        elif plat == "linux":
            paths.append(Path.home() / ".config" / "Claude" / "claude_desktop_config.json")
        elif plat == "win32":
            appdata = os.environ.get("APPDATA")
            if appdata:
                paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")

        modified = []
        for p in paths:
            def patcher(cfg: Dict[str, Any]) -> bool:
                servers = cfg.setdefault("mcpServers", {})
                servers["smriti"] = self._std_server_def()
                return True
            if patch_json_config(p, patcher):
                modified.append(str(p))
        return modified

    # 3. Cursor IDE
    def configure_cursor(self) -> List[str]:
        targets = [
            Path.home() / ".cursor" / "mcp.json",
            Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "cursor.mcp" / "mcp.json",
        ]
        modified = []
        for p in targets:
            # Only patch if directory or file exists or standard ~/.cursor
            if p.parent.exists() or p == targets[0]:
                def patcher(cfg: Dict[str, Any]) -> bool:
                    servers = cfg.setdefault("mcpServers", {})
                    servers["smriti-memory"] = self._std_server_def()
                    return True
                if patch_json_config(p, patcher):
                    modified.append(str(p))

        # Global ~/.cursorrules
        cursorrules = Path.home() / ".cursorrules"
        rule_content = """# SMRITI Persistent Memory for Cursor
Always call `smriti_recall` with keywords from the user query before writing code.
Call `smriti_encode` when resolving issues or recording project conventions.
"""
        if not cursorrules.exists():
            cursorrules.write_text(rule_content, encoding="utf-8")
            modified.append(str(cursorrules))

        return modified

    # 4. Windsurf / Cascade
    def configure_windsurf(self) -> List[str]:
        targets = [
            Path.home() / ".codeium" / "windsurf" / "mcp_config.json",
        ]
        modified = []
        for p in targets:
            if p.parent.exists():
                def patcher(cfg: Dict[str, Any]) -> bool:
                    servers = cfg.setdefault("mcpServers", {})
                    servers["smriti-memory"] = self._std_server_def()
                    return True
                if patch_json_config(p, patcher):
                    modified.append(str(p))
        return modified

    # 5. Gemini / Antigravity
    def configure_gemini(self) -> List[str]:
        targets = [
            Path.home() / ".gemini" / "config" / "mcp_config.json",
            Path.home() / ".gemini" / "antigravity" / "mcp_config.json",
            Path.home() / ".gemini" / "antigravity-ide" / "mcp_config.json",
        ]
        modified = []
        for p in targets:
            if p.parent.exists():
                def patcher(cfg: Dict[str, Any]) -> bool:
                    servers = cfg.setdefault("mcpServers", {})
                    servers["smriti-memory"] = self._std_server_def()
                    return True
                if patch_json_config(p, patcher):
                    modified.append(str(p))

        # AGENTS.md
        agents_md = Path.home() / ".gemini" / "config" / "AGENTS.md"
        if agents_md.parent.exists():
            rule_header = "## Smriti Memory System Hooks"
            existing = agents_md.read_text(encoding="utf-8") if agents_md.exists() else "# Global Customizations\n"
            if rule_header not in existing:
                agents_md.write_text(
                    existing + f"\n\n{rule_header}\n1. Call `smriti_recall` on session start.\n2. Call `smriti_encode` on milestones.\n3. Call `smriti_consolidate` after encoding.\n",
                    encoding="utf-8"
                )
                modified.append(str(agents_md))

        return modified

    # 6. Continue.dev (Grok, Ollama, DeepSeek, etc.)
    def configure_continue(self) -> List[str]:
        targets = [Path.home() / ".continue" / "config.json"]
        modified = []
        for p in targets:
            if p.parent.exists():
                def patcher(cfg: Dict[str, Any]) -> bool:
                    exp = cfg.setdefault("experimental", {})
                    mcps = exp.setdefault("modelContextProtocolServers", [])
                    # Check if already present
                    already = any(s.get("name") == "smriti-memory" for s in mcps if isinstance(s, dict))
                    if not already:
                        mcps.append({
                            "name": "smriti-memory",
                            "transport": {
                                "type": "stdio",
                                "command": self.python_bin,
                                "args": ["-m", "smriti_memcore.integrations.mcp_server"],
                                "env": self._get_env(),
                            }
                        })
                        return True
                    return False
                if patch_json_config(p, patcher):
                    modified.append(str(p))
        return modified

    # 7. Zed Editor
    def configure_zed(self) -> List[str]:
        targets = [Path.home() / ".config" / "zed" / "settings.json"]
        modified = []
        for p in targets:
            if p.parent.exists():
                def patcher(cfg: Dict[str, Any]) -> bool:
                    cs = cfg.setdefault("context_servers", {})
                    cs["smriti-memory"] = {
                        "command": {
                            "path": self.python_bin,
                            "args": ["-m", "smriti_memcore.integrations.mcp_server"],
                            "env": self._get_env(),
                        }
                    }
                    return True
                if patch_json_config(p, patcher):
                    modified.append(str(p))
        return modified

    # 8. Cline & Roo Code
    def configure_cline(self) -> List[str]:
        targets = [
            Path.home() / "Library" / "Application Support" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json",
            Path.home() / "Library" / "Application Support" / "Code" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline" / "settings" / "cline_mcp_settings.json",
        ]
        modified = []
        for p in targets:
            if p.parent.exists():
                def patcher(cfg: Dict[str, Any]) -> bool:
                    servers = cfg.setdefault("mcpServers", {})
                    srv = self._std_server_def()
                    srv["disabled"] = False
                    srv["autoApprove"] = []
                    servers["smriti-memory"] = srv
                    return True
                if patch_json_config(p, patcher):
                    modified.append(str(p))
        return modified


def detect_installed_clients() -> Dict[str, bool]:
    """Scan the environment for known AI coding clients and IDEs."""
    status = {}

    # Claude Code
    status["claude_code"] = (Path.home() / ".claude.json").exists() or (Path.home() / ".claude").exists()

    # Claude Desktop
    d_mac = Path.home() / "Library" / "Application Support" / "Claude"
    d_lin = Path.home() / ".config" / "Claude"
    status["claude_desktop"] = d_mac.exists() or d_lin.exists()

    # Cursor
    c_dir = Path.home() / ".cursor"
    c_mac = Path.home() / "Library" / "Application Support" / "Cursor"
    status["cursor"] = c_dir.exists() or c_mac.exists()

    # Windsurf
    w_dir = Path.home() / ".codeium" / "windsurf"
    status["windsurf"] = w_dir.exists()

    # Gemini / Antigravity
    g_dir = Path.home() / ".gemini"
    status["gemini"] = g_dir.exists()

    # Continue.dev
    con_dir = Path.home() / ".continue"
    status["continue"] = con_dir.exists()

    # Zed
    z_dir = Path.home() / ".config" / "zed"
    status["zed"] = z_dir.exists()

    # Cline / Roo Code
    cl_dir = Path.home() / "Library" / "Application Support" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev"
    roo_dir = Path.home() / "Library" / "Application Support" / "Code" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline"
    status["cline"] = cl_dir.exists() or roo_dir.exists()

    return status


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="smriti-install",
        description="SMRITI DevBrain Universal Installer for Coding Agents and IDEs",
    )
    parser.add_argument("--all", "-a", action="store_true", help="Install to all detected clients automatically")
    parser.add_argument("--client", "-c", help="Comma-separated clients (claude_code, cursor, windsurf, gemini, etc.)")
    parser.add_argument("--python", help="Path to Python binary to run the MCP server")
    parser.add_argument("--storage", default="~/.smriti/global", help="Storage directory (default: ~/.smriti/global)")
    parser.add_argument("--model", default="mistral", help="Consolidation LLM model (default: mistral)")
    parser.add_argument("--api-key", default="", help="LLM API key for cloud models")
    parser.add_argument("--obsidian", default="", help="Obsidian vault directory for palace sync")
    parser.add_argument("--yes", "-y", action="store_true", help="Do not prompt for confirmation")

    args = parser.parse_args(argv)

    print("\n🧠 SMRITI DevBrain: Universal Agent Memory Installer\n" + "=" * 55)

    py_bin = resolve_python_bin(args.python)
    print(f"  Python Runtime : {py_bin}")
    print(f"  Memory Storage : {args.storage}")
    print(f"  Default Model  : {args.model}")
    print("=" * 55)

    detected = detect_installed_clients()
    print("\n🔍 Detecting Installed AI Agents & IDEs:")
    for client_name, found in detected.items():
        icon = "✓ Found" if found else "· Not found"
        print(f"  [{icon}] {client_name}")

    configurator = ClientConfigurator(
        python_bin=py_bin,
        storage_path=args.storage,
        llm_model=args.model,
        llm_api_key=args.api_key,
        obsidian_path=args.obsidian,
    )

    client_map: Dict[str, Callable[[], List[str]]] = {
        "claude_code": configurator.configure_claude_code,
        "claude_desktop": configurator.configure_claude_desktop,
        "cursor": configurator.configure_cursor,
        "windsurf": configurator.configure_windsurf,
        "gemini": configurator.configure_gemini,
        "continue": configurator.configure_continue,
        "zed": configurator.configure_zed,
        "cline": configurator.configure_cline,
    }

    selected = []
    if args.client:
        requested = [c.strip().lower() for c in args.client.split(",")]
        for r in requested:
            if r in client_map:
                selected.append(r)
            else:
                print(f"  ⚠ Unknown client '{r}' (available: {', '.join(client_map.keys())})")
    elif args.all or args.yes:
        selected = [k for k, v in detected.items() if v]
    else:
        # Default: all detected
        selected = [k for k, v in detected.items() if v]

    if not selected:
        print("\nNo matching clients found to configure.")
        return

    print(f"\n🚀 Configuring SMRITI Persistent Memory for: {', '.join(selected)}\n")

    total_patched = 0
    for c in selected:
        fn = client_map.get(c)
        if fn:
            modified = fn()
            if modified:
                print(f"  ✓ {c.upper()}:")
                for m in modified:
                    print(f"     → Patched {m}")
                total_patched += len(modified)
            else:
                print(f"  · {c.upper()}: Already up to date")

    print("\n" + "=" * 55)
    print(f"🎉 DevBrain Setup Complete! ({total_patched} config files updated)")
    print("   Restart your IDE or agent session to activate SMRITI memory.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
