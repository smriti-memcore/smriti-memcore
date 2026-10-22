"""
SMRITI Unified CLI.
Unified command-line interface for the SMRITI cognitive memory engine,
persona management, interactive chat, MCP server installation, and UI.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def print_banner():
    banner = r"""
  ____  __  __ ____  ___ _____ ___ 
 / ___||  \/  |  _ \|_ _|_   _|_ _|
 \___ \| |\/| | |_) || |  | |  | | 
  ___) | |  | |  _ < | |  | |  | | 
 |____/|_|  |_|_| \_\___| |_| |___|
 Neuro-Inspired Long-Term Memory Engine
"""
    print(banner)


def cmd_pull(args: argparse.Namespace):
    from smriti_memcore.personas.hub import pull_persona
    persona_id = args.persona
    print(f"📦 Fetching persona '{persona_id}'...")
    try:
        target_dir, manifest = pull_persona(persona_id, force=args.force)
        print(f"\n✅ Successfully pulled '{manifest.name}' ({manifest.id}) v{manifest.version}")
        print(f"   Tagline: {manifest.tagline}")
        print(f"   Rooms: {manifest.total_rooms} | Memories: {manifest.total_memories}")
        print(f"   Cached at: {target_dir}")
        print(f"\n🚀 Start chatting: smriti chat {manifest.id}\n")
    except Exception as e:
        print(f"❌ Failed to pull persona '{persona_id}': {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list(args: argparse.Namespace):
    from smriti_memcore.personas.hub import list_local_personas, list_official_personas

    print("\n🏛️  SMRITI Persona Catalog\n" + "=" * 60)

    # 1. Official Registry
    official = list_official_personas()
    print("🌐 Available in Official Registry (pull on-demand):")
    for pid, info in official.items():
        print(f"  • {pid:<18} : {info['name']} — {info['description'][:65]}...")

    # 2. Local Installed
    print("\n💾 Installed Locally in Cache:")
    local_personas = list_local_personas()
    if not local_personas:
        print("  (No personas installed yet. Run 'smriti pull <persona>' to download one.)")
    else:
        for m in local_personas:
            print(f"  • {m.id:<18} : {m.name} ({m.total_rooms} rooms, {m.total_memories} memories)")
    print("\n" + "=" * 60 + "\n")


def cmd_chat(args: argparse.Namespace):
    from smriti_memcore.personas.chat import run_persona_chat
    run_persona_chat(persona_id=args.persona, model=args.model)


def cmd_ui(args: argparse.Namespace):
    print(f"🌐 Launching SMRITI Visual Memory Graph UI on port {args.port}...")
    from smriti_memcore.ui.server import launch
    launch(storage_path=args.storage, port=args.port, open_browser=not args.no_browser)


def cmd_consolidate(args: argparse.Namespace):
    from smriti_memcore.core import SMRITI
    from smriti_memcore.models import ConsolidationDepth, SmritiConfig
    print("🧠 Running SMRITI System 2 Background Consolidation...")
    cfg = SmritiConfig()
    if args.storage:
        cfg.storage_path = args.storage
    smriti = SMRITI(config=cfg)
    depth = ConsolidationDepth(args.depth) if args.depth else None
    result = smriti.consolidate(depth=depth)
    smriti.save()
    smriti.close()
    print("\n✅ Consolidation completed successfully!")
    if isinstance(result, dict):
        for k, v in result.items():
            print(f"   • {k}: {v}")


def cmd_install(args: argparse.Namespace):
    print("🔌 Setting up SMRITI MCP Server...")
    from smriti_memcore.integrations.install import main as install_main
    install_main()


def cmd_stats(args: argparse.Namespace):
    from smriti_memcore.core import SMRITI
    smriti = SMRITI()
    h = smriti.palace.health()
    print("\n📊 SMRITI System Statistics\n" + "=" * 40)
    print(f"  Memories       : {h.get('memory_count', 0)}")
    print(f"  Semantic Rooms : {h.get('room_count', 0)}")
    print(f"  Palace Edges   : {h.get('edge_count', 0)}")
    print(f"  Landmarks      : {h.get('landmark_count', 0)}")
    print(f"  Episodes       : {smriti.episode_buffer.count}")
    print(f"  Vector Store   : {smriti.vector_store.size} entries")
    print("=" * 40 + "\n")
    smriti.close()


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="smriti",
        description="SMRITI: Neuro-Inspired Long-Term Memory Engine for AI Agents",
    )

    try:
        from importlib.metadata import version
        pkg_version = version("smriti-memcore")
    except Exception:
        pkg_version = "1.4.27"

    parser.add_argument("-v", "--version", action="version", version=f"smriti {pkg_version}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. pull
    p_pull = subparsers.add_parser("pull", help="Download a pre-curated personality memory bundle")
    p_pull.add_argument("persona", help="Persona identifier (e.g. einstein, newton, marcus-aurelius)")
    p_pull.add_argument("--force", "-f", action="store_true", help="Force re-download even if already cached")
    p_pull.set_defaults(func=cmd_pull)

    # 2. list
    p_list = subparsers.add_parser("list", help="List available and locally installed personas")
    p_list.set_defaults(func=cmd_list)

    # 3. chat
    p_chat = subparsers.add_parser("chat", help="Start an interactive conversation with a persona")
    p_chat.add_argument("persona", nargs="?", default="albert-einstein", help="Persona identifier (default: albert-einstein)")
    p_chat.add_argument("--model", "-m", default=None, help="Optional LLM model override")
    p_chat.set_defaults(func=cmd_chat)

    # 4. ui
    p_ui = subparsers.add_parser("ui", help="Launch the interactive D3.js memory graph visualization")
    p_ui.add_argument("--storage", "-s", default="~/.smriti/global", help="Path to storage directory (default: ~/.smriti/global)")
    p_ui.add_argument("--port", "-p", type=int, default=7799, help="Port to serve UI on (default: 7799)")
    p_ui.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    p_ui.set_defaults(func=cmd_ui)

    # 5. consolidate
    p_consolidate = subparsers.add_parser("consolidate", help="Trigger System 2 background consolidation")
    p_consolidate.add_argument("--depth", "-d", choices=["full", "light"], default=None, help="Consolidation depth")
    p_consolidate.add_argument("--storage", "-s", default=None, help="Path to storage directory (default: ~/.smriti/global)")
    p_consolidate.set_defaults(func=cmd_consolidate)

    # 6. install
    p_install = subparsers.add_parser("install", help="Install SMRITI MCP server for Claude, Antigravity, and Cursor")
    p_install.set_defaults(func=cmd_install)

    # 7. stats
    p_stats = subparsers.add_parser("stats", help="Display local memory palace and vector store statistics")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)

    if not args.command:
        print_banner()
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
