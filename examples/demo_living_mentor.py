#!/usr/bin/env python3
"""
SMRITI Living Mentors — Live Demonstration Script for Educators and Students.
Demonstrates:
  1. Professor's workflow: Packaging a course curriculum (slides, transcripts) into a .smritibundle.
  2. Student's workflow: Importing the bundle and experiencing 24/7 Socratic Office Hours.
  3. Real-time token streaming with pinpoint slide and video transcript citations.

Usage:
  python3 examples/demo_living_mentor.py          # Interactive mode
  python3 examples/demo_living_mentor.py --auto   # Automated presentation mode (for screen recording)
"""

import argparse
import sys
import time
from pathlib import Path

from smriti_memcore.core import SMRITI


def print_banner(text: str, char: str = "━", color: str = "\033[1;36m"):
    reset = "\033[0m"
    line = char * 72
    print(f"\n{color}{line}")
    print(f" {text}")
    print(f"{line}{reset}\n")


def print_step(num: int, title: str):
    print(f"\033[1;33m[Step {num}]\033[0m \033[1;37m{title}\033[0m")


def slow_type(text: str, delay: float = 0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def run_demo(auto: bool = False):
    print_banner("🎓 SMRITI Living Mentors: Offline Socratic Coursework Demo", "═", "\033[1;35m")
    print(
        "Welcome! This demo demonstrates how SMRITI bridges the gap between static\n"
        "course materials (PDF slides, video lectures) and 24/7 interactive student office hours.\n"
        "100% offline, privacy-first, with zero cloud hosting costs."
    )
    if not auto:
        input("\nPress [Enter] to begin Step 1: The Professor's Workflow...")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1: The Professor's Workflow
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("👨‍🏫 PART 1: The Professor's Workflow — Zero-Infra Course Packaging")
    print_step(1, "The Challenge: Static materials students struggle to digest.")
    print(
        "  • A typical course has 20+ slide decks, 10+ hours of video lectures, and readings.\n"
        "  • Students get stuck on homework at 2:00 AM when office hours are closed.\n"
        "  • Cloud-hosted chatbots cost recurring subscription fees and risk hallucinations.\n"
    )

    print_step(2, "The Solution: One-command packaging into a '.smritibundle'.")
    print("  The professor simply runs in their terminal:\n")
    print("    \033[1;32m$ smriti pack ./MIT-15.773-Deep-Learning -o mit-15-773.smritibundle\033[0m\n")
    print("  What SMRITI automatically performs in ~85 seconds:")
    print("    ✓ Scans 25 PDF slide decks and 11 video lecture caption tracks (.webvtt/.srt)")
    print("    ✓ Strips video timecodes and stitches speech into clean natural prose")
    print("    ✓ Maps lecture topics into 14 sequential Semantic Palace Rooms")
    print("    ✓ Generates 3,293 vector embeddings locally using Apple Metal GPU acceleration")
    print("    ✓ Produces a compact, self-contained 6.5 MB archive: 'mit-15-773.smritibundle'")
    print("\n  📦 The professor uploads this 6.5 MB file to Canvas, Blackboard, or Piazza.")

    if not auto:
        input("\nPress [Enter] to continue to Step 2: The Student's Experience...")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2: The Student's Experience
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("🎓 PART 2: The Student's Experience — 24/7 Socratic Office Hours")
    print_step(3, "Student downloads the bundle and installs SMRITI.")
    print("  The student runs two simple commands:\n")
    print("    \033[1;32m$ pip install smriti-memcore\033[0m")
    print("    \033[1;32m$ smriti pull ./mit-15-773.smritibundle\033[0m")
    print("    \033[1;32m$ smriti chat mit-15-773\033[0m\n")

    print_step(4, "Initializing the Cognitive Memory Palace...")
    time.sleep(0.5)

    smriti, manifest = SMRITI.from_persona("mit-15-773")
    speaker_label = manifest.author or manifest.name

    print(f"\n  🎭 \033[1mLoaded Persona\033[0m : {manifest.name} ({manifest.era})")
    print(f"  👨‍🏫 \033[1mInstructor\033[0m     : {speaker_label}")
    print(f"  🏛️  \033[1mPalace Rooms\033[0m   : {manifest.total_rooms} lecture rooms")
    print(f"  🧠 \033[1mTotal Memories\033[0m : {manifest.total_memories} curriculum memories")
    print(f"  🤖 \033[1mActive Model\033[0m   : {smriti.llm.default_model} (100% Local Ollama)")

    if not auto:
        input("\nPress [Enter] to run live simulated student questions...")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 3: Live Socratic Q&A Demonstration
    # ──────────────────────────────────────────────────────────────────────────
    demo_questions = [
        (
            "What is the starting input for image generation in Lecture 11?",
            "Shows pinpoint routing to Lecture 11 (Diffusion Models) with transcript & slide attribution.",
        ),
        (
            "How does backpropagation compute gradients through the network in Lecture 02?",
            "Shows deep mathematical & conceptual grounding from Lecture 02 slide decks.",
        ),
    ]

    for q_idx, (q_text, description) in enumerate(demo_questions, 1):
        print_banner(f"Live Query #{q_idx}: \"{q_text}\"", "─", "\033[1;34m")
        print(f"\033[3mGoal: {description}\033[0m\n")

        print(f"\033[1;32mStudent:\033[0m {q_text}\n")
        time.sleep(0.4)

        # 1. Recall
        recalled = smriti.recall(q_text, top_k=3)
        print(f"\033[1;36m[Semantic Memory Palace Primed]\033[0m")
        for i, m in enumerate(recalled, 1):
            room_topic = smriti.palace.rooms.get(m.room_id).topic if m.room_id in smriti.palace.rooms else "Palace"
            print(f"  [{i}] \033[1;33m({room_topic})\033[0m {m.content[:130]}...")

        # 2. Socratic Response
        context_block = "\n".join([f"- {m.content}" for m in recalled])
        system_prompt = (
            f"{manifest.system_prompt_hint}\n\n"
            f"RELEVANT CURRICULUM FROM YOUR PALACE:\n"
            f"{context_block}\n\n"
            f"GUIDANCE:\n"
            f"- Speak authentically as {speaker_label}.\n"
            f"- Guide the student Socratically with intuition from the course slides and video transcripts.\n"
            f"- Do NOT repeat the student's question."
        )
        prompt = f"Student: {q_text}\n\n{speaker_label}:"

        print(f"\n\033[1;35m{speaker_label}:\033[0m ", end="", flush=True)

        for token in smriti.llm.generate_stream(prompt=prompt, system=system_prompt):
            sys.stdout.write(token)
            sys.stdout.flush()
            if auto:
                time.sleep(0.01)
        print("\n")

        if not auto and q_idx < len(demo_questions):
            input("Press [Enter] for the next query...")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 4: Key Takeaways for Educators
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("✨ Key Takeaways for Professors & Institutions", "═", "\033[1;32m")
    print(
        "1. Zero Cloud Bills: Local inference means $0 OpenAI API bills for your department.\n"
        "2. Zero Privacy Violations: 100% offline compliance (FERPA & GDPR compliant).\n"
        "3. Socratic Integrity: Answers teach first principles rather than providing cheat sheets.\n"
        "4. Artifact Portability: A 6.5 MB file holds your entire course memory.\n\n"
        "To try it on your own course:\n"
        "  $ pip install smriti-memcore\n"
        "  $ smriti pack ./your-course-materials -o my-course.smritibundle\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SMRITI Living Mentors Demonstration")
    parser.add_argument("--auto", action="store_true", help="Run automated presentation mode")
    args = parser.parse_args()
    run_demo(auto=args.auto)
