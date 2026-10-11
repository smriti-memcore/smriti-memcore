"""
SMRITI Persona Interactive Terminal Chat.
Allows developers and users to converse directly with pre-curated cognitive personas.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from smriti_memcore.core import SMRITI
from smriti_memcore.personas.hub import pull_persona
from smriti_memcore.personas.manifest import PersonaManifest


def run_persona_chat(persona_id: str = "albert-einstein", model: Optional[str] = None):
    print("=" * 70)
    print(f"🧠 SMRITI Persona Engine — Initializing '{persona_id}'...")
    print("=" * 70)

    smriti, manifest = SMRITI.from_persona(persona_id)

    print(f"\n🎭 Loaded Persona: {manifest.name} ({manifest.era})")
    print(f"📌 Tagline: {manifest.tagline}")
    print(f"🏛️  Rooms: {manifest.total_rooms} rooms | 🧠 Memories: {manifest.total_memories} pinned memories")
    print("\n🏛️  Cognitive Palace Rooms:")
    for r in manifest.core_rooms:
        print(f"   • {r}")

    print("\n💡 Suggested prompts to ask:")
    for p in manifest.suggested_prompts[:4]:
        print(f"   - \"{p}\"")

    print("\n" + "-" * 70)
    print("Type your questions below. (Type 'exit', 'quit', or 'q' to stop)")
    print("-" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nFarewell!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            farewell = manifest.metadata.get("farewell") or "Farewell, my friend. Thank you for this conversation."
            print(f"\n{manifest.name}: {farewell}\n")
            break

        # 1. Recall relevant memories from the persona's semantic palace
        recalled = smriti.recall(user_input, top_k=3)

        print(f"\n[{manifest.name}'s Memory Palace Primed]")
        if recalled:
            for i, m in enumerate(recalled, 1):
                room_name = smriti.palace.rooms.get(m.room_id).topic if m.room_id in smriti.palace.rooms else "Unknown Room"
                print(f"  [{i}] ({room_name}) {m.content[:100]}...")
        else:
            print("  (No direct anchor memories matched; reflecting...)")

        # 2. Build augmented response using configured LLM or heuristic synthesis
        context_block = "\n".join([f"- {m.content}" for m in recalled])
        system_prompt = f"{manifest.system_prompt_hint}\n\nRelevant memories from your life and thinking:\n{context_block}"

        print(f"\n{manifest.name}: ", end="", flush=True)
        try:
            # Generate response via smriti's LLM interface
            prompt = f"User asked: {user_input}\nRespond in character:"
            response = smriti.llm.generate(prompt=prompt, system=system_prompt)
            if hasattr(response, "text") and response.text:
                reply = response.text
            elif hasattr(response, "content") and response.content:
                reply = response.content
            else:
                reply = str(response)

            reply = reply.strip()
            # Clean possible prompt echoing from LLM output
            if reply.startswith(f"User asked: {user_input}"):
                reply = reply[len(f"User asked: {user_input}"):].strip()
            if reply.startswith(user_input):
                reply = reply[len(user_input):].strip()
            if reply.lower().startswith("respond in character:"):
                reply = reply[len("respond in character:"):].strip()

            print(reply.strip() + "\n")
        except Exception as e:
            # Fallback if no local/remote LLM is running
            print(
                f"(LLM offline: {e})\n"
                f"Based on my memories: {recalled[0].content if recalled else 'I ponder this deeply.'}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Chat with a SMRITI Personality Memory Palace")
    parser.add_argument(
        "--persona",
        "-p",
        default="albert-einstein",
        help="Persona identifier (e.g. albert-einstein, isaac-newton)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Optional LLM model override",
    )
    args = parser.parse_args()
    run_persona_chat(persona_id=args.persona, model=args.model)


if __name__ == "__main__":
    main()
