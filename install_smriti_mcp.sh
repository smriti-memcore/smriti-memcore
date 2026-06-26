#!/usr/bin/env bash
# install_smriti_mcp.sh — Register the SMRITI MCP server with Claude Code and Claude Desktop
#
# Usage:
#   bash install_smriti_mcp.sh
#
# What it does:
#   1. Creates a dedicated venv at ~/.smriti/venv
#   2. Installs smriti-memcore[mcp] into it
#   3. Patches ~/.claude.json to register the smriti MCP server (Claude Code)
#   4. Patches Claude Desktop config if detected (absolute paths required)
#   5. Patches ~/.claude/settings.json to add recall/encode hooks
#   6. Patches ~/.claude/CLAUDE.md with SMRITI memory instructions
#
# Requirements: Python 3.9+, Claude Code and/or Claude Desktop

set -euo pipefail

VENV_DIR="$HOME/.smriti/venv"

# ── Helpers ───────────────────────────────────────────────────────────────────

info()    { echo "[smriti] $*"; }
warn()    { echo "[smriti] ⚠ $*" >&2; }
success() { echo "[smriti] ✓ $*"; }
error()   { echo "[smriti] ✗ $*" >&2; exit 1; }

# ── 1. Create dedicated venv ──────────────────────────────────────────────────

PY=$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 || command -v python || true)
[[ -z "$PY" ]] && error "Python 3.10+ not found. Install it first."

PY_VERSION=$("$PY" -c "import sys; print(sys.version_info.minor)")
[[ "$PY_VERSION" -lt 10 ]] && error "Python 3.10+ required (found 3.$PY_VERSION)."

if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
    info "Creating venv at $VENV_DIR..."
    "$PY" -m venv "$VENV_DIR"
    success "Venv created"
else
    info "Using existing venv at $VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python3"

# ── 2. Install package into venv ──────────────────────────────────────────────

info "Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip --quiet
success "pip upgraded"

info "Installing smriti-memcore[mcp]..."
"$PYTHON" -m pip install "smriti-memcore[mcp]" --quiet --upgrade
# Ensure mcp is installed even if the PyPI release pre-dates the extra
"$PYTHON" -c "import mcp" 2>/dev/null || "$PYTHON" -m pip install "mcp>=1.0.0" --quiet
success "smriti-memcore[mcp] installed"

# Verify imports
"$PYTHON" -c "import smriti_memcore" 2>/dev/null \
    || error "smriti_memcore not importable after install — check pip output above."
"$PYTHON" -c "import mcp" 2>/dev/null \
    || error "mcp not importable after install — check pip output above."

success "Using Python: $PYTHON"

# ── 3. Prompt for LLM config ──────────────────────────────────────────────────

echo ""
echo "Which LLM should SMRITI use for memory consolidation?"
echo "  1) mistral (local Ollama — default, no API key needed)"
echo "  2) claude-haiku-4-5-20251001 (Anthropic API key required)"
echo "  3) gpt-4o-mini (OpenAI API key required)"
echo "  4) Enter custom model name"
echo ""
read -rp "Choice [1]: " MODEL_CHOICE
MODEL_CHOICE="${MODEL_CHOICE:-1}"

case "$MODEL_CHOICE" in
    1) LLM_MODEL="mistral";                   LLM_API_KEY="" ;;
    2) LLM_MODEL="claude-haiku-4-5-20251001"; read -rsp "Anthropic API key: " LLM_API_KEY; echo ;;
    3) LLM_MODEL="gpt-4o-mini";               read -rsp "OpenAI API key: " LLM_API_KEY; echo ;;
    4) read -rp "Model name: " LLM_MODEL;     read -rsp "API key (leave blank for Ollama): " LLM_API_KEY; echo ;;
    *) LLM_MODEL="mistral";                   LLM_API_KEY="" ;;
esac

# ── Validate Ollama model is available (if using local Ollama) ─────────────
if [[ -z "$LLM_API_KEY" ]]; then
    if curl -s --connect-timeout 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        AVAILABLE_MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "
import json,sys
data = json.load(sys.stdin)
names = [m['name'].split(':')[0] for m in data.get('models', [])]
print('\n'.join(names))
" 2>/dev/null)
        if [[ -n "$AVAILABLE_MODELS" ]]; then
            if ! echo "$AVAILABLE_MODELS" | grep -qx "$LLM_MODEL"; then
                warn "Ollama is running but model '$LLM_MODEL' is not pulled."
                echo "  Available models: $(echo "$AVAILABLE_MODELS" | tr '\n' ' ')"
                FIRST_MODEL=$(echo "$AVAILABLE_MODELS" | head -1)
                read -rp "  Use '$FIRST_MODEL' instead? [Y/n]: " USE_FIRST
                if [[ "${USE_FIRST:-Y}" =~ ^[Yy]$ ]]; then
                    LLM_MODEL="$FIRST_MODEL"
                    success "Using Ollama model: $LLM_MODEL"
                else
                    warn "Proceeding with '$LLM_MODEL' — run 'ollama pull $LLM_MODEL' before using SMRITI consolidation."
                fi
            else
                success "Ollama model '$LLM_MODEL' is available."
            fi
        fi
    else
        warn "Ollama not running at localhost:11434. Consolidation LLM calls will fail until Ollama is started."
        warn "Start Ollama and run 'ollama pull $LLM_MODEL', or re-run this script to choose a cloud model."
    fi
fi

read -rp "Memory storage path [~/.smriti/global]: " STORAGE_PATH
STORAGE_PATH="${STORAGE_PATH:-~/.smriti/global}"

echo ""
echo "SMRITI can export your Semantic Palace to an Obsidian vault (optional)."
echo "This enables the smriti_sync_obsidian MCP tool and Obsidian graph view."
echo ""
read -rp "Obsidian vault Palace directory (sets SMRITI_OBSIDIAN_PATH, leave blank to skip): " OBSIDIAN_PATH

# ── 4. Patch ~/.claude.json ───────────────────────────────────────────────────

info "Registering smriti MCP server in ~/.claude.json..."

"$PYTHON" - <<PYEOF
import json, os

claude_json = os.path.expanduser("~/.claude.json")

if os.path.exists(claude_json):
    with open(claude_json) as f:
        config = json.load(f)
else:
    config = {}

if "mcpServers" not in config:
    config["mcpServers"] = {}

env = {
    "PYTHONPATH": "",
    "SMRITI_STORAGE_PATH": "$STORAGE_PATH",
    "SMRITI_LLM_MODEL": "$LLM_MODEL",
    "SMRITI_LLM_API_KEY": "$LLM_API_KEY",
}

obsidian_path = "$OBSIDIAN_PATH".strip()
if obsidian_path:
    env["SMRITI_OBSIDIAN_PATH"] = obsidian_path

config["mcpServers"]["smriti"] = {
    "command": "$PYTHON",
    "args": ["-m", "smriti_memcore.integrations.mcp_server"],
    "env": env,
}

with open(claude_json, "w") as f:
    json.dump(config, f, indent=2)

print(f"[smriti] ✓ Written to {claude_json}")
PYEOF

# ── 5. Patch Claude Desktop config (if detected) ─────────────────────────────

DESKTOP_CONFIG_MACOS="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
DESKTOP_CONFIG_LINUX="$HOME/.config/Claude/claude_desktop_config.json"

if [[ -f "$DESKTOP_CONFIG_MACOS" ]]; then
    DESKTOP_CONFIG="$DESKTOP_CONFIG_MACOS"
elif [[ -f "$DESKTOP_CONFIG_LINUX" ]]; then
    DESKTOP_CONFIG="$DESKTOP_CONFIG_LINUX"
else
    DESKTOP_CONFIG=""
fi

if [[ -n "$DESKTOP_CONFIG" ]]; then
    echo ""
    read -rp "Claude Desktop detected — also register smriti there? [Y/n]: " CONFIGURE_DESKTOP
    CONFIGURE_DESKTOP="${CONFIGURE_DESKTOP:-Y}"

    if [[ "$CONFIGURE_DESKTOP" =~ ^[Yy]$ ]]; then
        info "Registering smriti MCP server in Claude Desktop config..."

        # Claude Desktop does not expand ~ — resolve to absolute paths
        ABS_STORAGE_PATH="${STORAGE_PATH/#\~/$HOME}"
        ABS_OBSIDIAN_PATH="${OBSIDIAN_PATH/#\~/$HOME}"

        "$PYTHON" - <<PYEOF
import json, os

desktop_config = "$DESKTOP_CONFIG"

if os.path.exists(desktop_config):
    with open(desktop_config) as f:
        config = json.load(f)
else:
    config = {}

if "mcpServers" not in config:
    config["mcpServers"] = {}

env = {
    "PYTHONPATH": "",
    "SMRITI_STORAGE_PATH": "$ABS_STORAGE_PATH",
    "SMRITI_LLM_MODEL": "$LLM_MODEL",
    "SMRITI_LLM_API_KEY": "$LLM_API_KEY",
}

obsidian_path = "$ABS_OBSIDIAN_PATH".strip()
if obsidian_path:
    env["SMRITI_OBSIDIAN_PATH"] = obsidian_path

config["mcpServers"]["smriti"] = {
    "command": "$PYTHON",
    "args": ["-m", "smriti_memcore.integrations.mcp_server"],
    "env": env,
}

with open(desktop_config, "w") as f:
    json.dump(config, f, indent=2)

print(f"[smriti] ✓ Written to {desktop_config}")
PYEOF
        success "Claude Desktop configured — restart it to activate smriti"
    fi
fi

# ── 5b. Patch Gemini (Antigravity) and Codex (Antigravity-IDE) configs ───────

GEMINI_DIR="$HOME/.gemini"
if [[ -d "$GEMINI_DIR" ]]; then
    echo ""
    read -rp "Gemini & Codex configs detected — register smriti-memory there? [Y/n]: " CONFIGURE_GEMINI_CODEX
    CONFIGURE_GEMINI_CODEX="${CONFIGURE_GEMINI_CODEX:-Y}"

    if [[ "$CONFIGURE_GEMINI_CODEX" =~ ^[Yy]$ ]]; then
        info "Registering smriti-memory MCP server in Gemini & Codex configs..."

        ABS_STORAGE_PATH="${STORAGE_PATH/#\~/$HOME}"
        ABS_OBSIDIAN_PATH="${OBSIDIAN_PATH/#\~/$HOME}"

        "$PYTHON" - <<PYEOF
import json, os

def patch_config(filepath):
    filepath = os.path.expanduser(filepath)
    if os.path.islink(filepath):
        filepath = os.path.realpath(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                config = json.load(f)
        except Exception:
            config = {}
    else:
        config = {}
        
    if "mcpServers" not in config:
        config["mcpServers"] = {}
        
    env = {
        "PYTHONPATH": "",
        "SMRITI_STORAGE_PATH": "$ABS_STORAGE_PATH",
        "SMRITI_LLM_MODEL": "$LLM_MODEL",
        "SMRITI_LLM_API_KEY": "$LLM_API_KEY",
    }
    obsidian_path = "$ABS_OBSIDIAN_PATH".strip()
    if obsidian_path:
        env["SMRITI_OBSIDIAN_PATH"] = obsidian_path
        
    config["mcpServers"]["smriti-memory"] = {
        "command": "$PYTHON",
        "args": ["-m", "smriti_memcore.integrations.mcp_server"],
        "env": env,
    }
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[smriti] ✓ Written to {filepath}")

# Register with Gemini (Antigravity) configs
patch_config("~/.gemini/config/mcp_config.json")
patch_config("~/.gemini/antigravity/mcp_config.json")

# Register with Codex (Antigravity-IDE) config
patch_config("~/.gemini/antigravity-ide/mcp_config.json")
PYEOF

        # Append global agent rules for Gemini/Codex
        AGENTS_MD="$GEMINI_DIR/config/AGENTS.md"
        RULE_HEADER="## Smriti Memory System Hooks"
        if [[ -d "$GEMINI_DIR/config" ]]; then
            if [[ ! -f "$AGENTS_MD" ]]; then
                echo "# Global Customizations" > "$AGENTS_MD"
            fi
            
            if ! grep -q "$RULE_HEADER" "$AGENTS_MD" 2>/dev/null; then
                cat << 'RULEEOF' >> "$AGENTS_MD"

## Smriti Memory System Hooks
1. **Automatic Memory Encoding**:
   - Automatically invoke `smriti_encode` (or `amp.encode`) whenever a significant project milestone is achieved, a key technical decision is finalized, or a physical hardware configuration is successfully calibrated.
2. **Context Recall**:
   - Proactively call `smriti_recall` (or `amp.recall`) at the beginning of a session or when exploring a new workspace area to retrieve historical context, past setups, or hardware parameters.
3. **Consolidation**:
   - Automatically run `smriti_consolidate` after encoding memories at the end of a session to clean the working slots and trigger the background reflection/forgotten indexing cycle.
RULEEOF
                success "Smriti memory hooks added to $AGENTS_MD"
            else
                info "Smriti memory hooks already present in $AGENTS_MD — skipping"
            fi
        fi
    fi
fi

# ── 6. Smoke test ─────────────────────────────────────────────────────────────

info "Verifying server starts..."
if "$PYTHON" -c "
import os
os.environ['SMRITI_STORAGE_PATH'] = '/tmp/smriti_install_test'
os.environ['SMRITI_LLM_MODEL'] = '$LLM_MODEL'
os.environ['SMRITI_LLM_API_KEY'] = '$LLM_API_KEY'
from smriti_memcore.integrations.mcp_server import build_smriti_config
cfg = build_smriti_config()
assert cfg.llm_model == '$LLM_MODEL'
print('ok')
" 2>/dev/null | grep -q ok; then
    success "Server verified"
else
    echo "[smriti] ⚠ Could not verify server — check your LLM config after launch"
fi

# ── 7. Patch ~/.claude/settings.json with hooks ───────────────────────────────

echo ""
echo "SMRITI can configure automatic memory hooks in ~/.claude/settings.json"
echo "so Claude recalls memories on every message and encodes takeaways at session end."
echo "This modifies your global Claude Code settings (affects all projects)."
echo ""
read -rp "Configure automatic memory hooks? [y/N]: " CONFIGURE_HOOKS
CONFIGURE_HOOKS="${CONFIGURE_HOOKS:-N}"

if [[ "$CONFIGURE_HOOKS" =~ ^[Yy]$ ]]; then
info "Configuring Claude Code hooks in ~/.claude/settings.json..."

"$PYTHON" - <<PYEOF
import json, os

settings_path = os.path.expanduser("~/.claude/settings.json")
os.makedirs(os.path.dirname(settings_path), exist_ok=True)

if os.path.exists(settings_path):
    with open(settings_path) as f:
        settings = json.load(f)
else:
    settings = {}

if "hooks" not in settings:
    settings["hooks"] = {}

# UserPromptSubmit — inject smriti_recall reminder into Claude's context
smriti_prompt_group = {
    "matcher": "",
    "hooks": [{
        "type": "command",
        "command": "echo '{\"hookSpecificOutput\": {\"hookEventName\": \"UserPromptSubmit\", \"additionalContext\": \"SMRITI MEMORY PROTOCOL: (1) BEFORE responding: call smriti_recall with 2-3 keywords from the user message. (2) AFTER responding: call smriti_encode for any of these that occurred this turn: user stated a preference or fact, a decision was made, a problem was solved, a key design choice was finalized, or new project context was revealed. Do not skip smriti_encode when these conditions are met.\"}}'",
        "statusMessage": "Loading SMRITI memories..."
    }]
}

if "UserPromptSubmit" not in settings["hooks"]:
    settings["hooks"]["UserPromptSubmit"] = []

# Only add if not already present
existing_cmds = []
for group in settings["hooks"]["UserPromptSubmit"]:
    for h in group.get("hooks", []):
        existing_cmds.append(h.get("command", ""))
if not any("smriti_recall" in cmd for cmd in existing_cmds):
    settings["hooks"]["UserPromptSubmit"].append(smriti_prompt_group)

# SessionStart — load working memory for the current project directory
smriti_session_group = {
    "matcher": "",
    "hooks": [{
        "type": "command",
        "command": r"""bash -c 'echo "{\"systemMessage\": \"SMRITI MEMORY: Session starting in \$(pwd). Call smriti_get_context to load working memory for this project, then smriti_recall with the project name and relevant keywords.\"}"'""",
        "statusMessage": "Loading SMRITI context..."
    }]
}

if "SessionStart" not in settings["hooks"]:
    settings["hooks"]["SessionStart"] = []

existing_session_cmds = []
for group in settings["hooks"]["SessionStart"]:
    for h in group.get("hooks", []):
        existing_session_cmds.append(h.get("command", ""))
if not any("smriti_get_context" in cmd for cmd in existing_session_cmds):
    settings["hooks"]["SessionStart"].append(smriti_session_group)

# Stop — remind Claude to encode takeaways
smriti_stop_group = {
    "matcher": "",
    "hooks": [{
        "type": "command",
        "command": "echo '{\"systemMessage\": \"SMRITI: Remember to call smriti_encode to store key facts, decisions, and context from this session before ending.\"}'",
        "statusMessage": "Saving SMRITI memory..."
    }]
}

if "Stop" not in settings["hooks"]:
    settings["hooks"]["Stop"] = []

existing_stop_cmds = []
for group in settings["hooks"]["Stop"]:
    for h in group.get("hooks", []):
        existing_stop_cmds.append(h.get("command", ""))
if not any("smriti_encode" in cmd for cmd in existing_stop_cmds):
    settings["hooks"]["Stop"].append(smriti_stop_group)

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)

print(f"[smriti] ✓ Hooks written to {settings_path}")
PYEOF

# ── 8. Patch ~/.claude/CLAUDE.md with SMRITI memory instructions ───────────────

info "Adding SMRITI memory instructions to ~/.claude/CLAUDE.md..."

CLAUDE_MD="$HOME/.claude/CLAUDE.md"
SMRITI_SECTION="
## SMRITI Memory

Use the smriti MCP tools to maintain memory across sessions.

**On every user message:** Call \`smriti_recall\` with 2-3 keywords before responding.

**Call \`smriti_encode\` immediately when any of these occur:**
- User states a preference, constraint, or personal fact (\"I prefer...\", \"we always...\", \"don't do X\")
- A technical decision is finalized (architecture choice, library selection, approach chosen)
- A bug root cause is identified or a problem is solved
- New project context is revealed (goals, deadlines, team setup, infra details)
- A workflow or process is established (\"from now on...\", \"the way we do this is...\")

**Do not wait until end of session** — encode facts as they emerge, mid-conversation.

**At session end:** Call \`smriti_encode\` with a summary of key takeaways not yet encoded."

if [[ ! -f "$CLAUDE_MD" ]]; then
    echo "# Global Claude Instructions" > "$CLAUDE_MD"
fi

if ! grep -q "SMRITI Memory" "$CLAUDE_MD" 2>/dev/null; then
    echo "$SMRITI_SECTION" >> "$CLAUDE_MD"
    success "SMRITI instructions added to $CLAUDE_MD"
else
    info "SMRITI instructions already present in $CLAUDE_MD — skipping"
fi

else
    info "Skipping hooks setup — you can add them manually via /hooks in Claude Code"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SMRITI MCP server registered successfully!"
echo " • Claude Code:                  restart, then run /mcp"
echo " • Claude Desktop:               restart, check Settings → Developer"
echo " • Gemini (Antigravity) & Codex: restart your IDE session"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
