"""
SMRITI Persona Compiler & Multimodal Coursework Ingestion Engine.
Compiles knowledge specifications (YAML, JSON, or directory trees of PDFs,
lecture transcripts, and notes) into self-contained .smritibundle cognitive artifacts.

Enables professors to package an entire course curriculum (slides, transcripts,
readings, syllabi) into a single downloadable bundle that students can chat with
24/7 as a Socratic TA.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ── File and Transcript Extractors ───────────────────────────────────────────

def clean_vtt_or_srt(text: str) -> str:
    """
    Cleans WebVTT (.vtt) and SubRip (.srt) video/audio transcripts by
    stripping timestamps, index counters, and formatting tags.
    Merges broken speech lines into natural paragraphs.
    """
    lines = text.splitlines()
    cleaned_lines = []
    # Pattern matching VTT/SRT timing lines like: 00:01:23.456 --> 00:01:28.000
    timestamp_pattern = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?[.,\d]*\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?[.,\d]*")

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        if line_s.upper().startswith("WEBVTT") or line_s.upper().startswith("NOTE"):
            continue
        if line_s.isdigit():  # Subtitle index number
            continue
        if timestamp_pattern.search(line_s):
            continue
        # Strip simple HTML styling tags (e.g. <v Speaker>, <i>, </b>)
        clean = re.sub(r"<[^>]+>", "", line_s)
        if clean:
            cleaned_lines.append(clean)

    # Join and group into cohesive paragraphs
    raw_text = " ".join(cleaned_lines)
    # Deduplicate excessive whitespace
    return re.sub(r"\s+", " ", raw_text).strip()


def extract_pdf_pages(file_path: Path) -> List[Tuple[int, str]]:
    """Extracts text per page from a PDF file using pypdf if available."""
    pages = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        for idx, page in enumerate(reader.pages, 1):
            txt = page.extract_text() or ""
            txt_clean = re.sub(r"\s+", " ", txt).strip()
            if txt_clean:
                pages.append((idx, txt_clean))
    except Exception as e:
        logger.warning(f"Could not extract PDF {file_path}: {e}")
    return pages


def extract_notebook_text(file_path: Path) -> str:
    """Extracts markdown text and key code documentation from a Jupyter Notebook."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        chunks = []
        for cell in data.get("cells", []):
            cell_type = cell.get("cell_type")
            source = "".join(cell.get("source", []))
            if cell_type == "markdown" and source.strip():
                chunks.append(source.strip())
            elif cell_type == "code" and source.strip():
                # Include short code snippets
                chunks.append(f"```python\n{source.strip()}\n```")
        return "\n\n".join(chunks)
    except Exception as e:
        logger.warning(f"Could not extract notebook {file_path}: {e}")
        return ""


def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """Splits text into cohesive chunks around sentence/paragraph boundaries."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_len = 0

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if current_len + len(s_clean) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [s_clean]
            current_len = len(s_clean)
        else:
            current_chunk.append(s_clean)
            current_len += len(s_clean) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# ── Directory Coursework Ingestion ────────────────────────────────────────────

def parse_coursework_directory(dir_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Scans a folder of course materials (PDFs, transcripts, notes, markdown)
    and constructs a structured Persona Palace specification.
    """
    root = Path(dir_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Path is not an existing directory: {root}")

    # Check for manifest or course metadata file
    meta_file = None
    for name in ["course.yaml", "course.json", "manifest.yaml", "manifest.json", "persona.yaml"]:
        p = root / name
        if p.exists():
            meta_file = p
            break

    spec = load_spec_file(meta_file) if meta_file else {}

    raw_name = root.name.replace("-", " ").replace("_", " ").title()
    course_name = spec.get("name") or raw_name
    course_id = spec.get("id") or root.name.lower().replace(" ", "-").replace("_", "-")

    # Group documents by subdirectory (Rooms)
    subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    rooms_data: List[Dict[str, Any]] = []

    if subdirs:
        # Hierarchical course layout: each folder is a unit/lecture (Room)
        subdirs.sort(key=lambda d: d.name)
        for subdir in subdirs:
            room_topic = subdir.name.replace("-", " ").replace("_", " ").title()
            room_memories: List[str] = []

            for f in sorted(subdir.rglob("*")):
                if not f.is_file() or f.name.startswith("."):
                    continue
                _ingest_file_into_memories(f, room_memories)

            if room_memories:
                rooms_data.append({
                    "topic": room_topic,
                    "description": f"Curriculum materials from {subdir.name}",
                    "memories": room_memories,
                })
    else:
        # Flat course layout: group by file or cluster
        files = [f for f in sorted(root.iterdir()) if f.is_file() and not f.name.startswith(".")]
        for f in files:
            if f == meta_file:
                continue
            room_topic = f.stem.replace("-", " ").replace("_", " ").title()
            room_memories: List[str] = []
            _ingest_file_into_memories(f, room_memories)

            if room_memories:
                rooms_data.append({
                    "topic": room_topic,
                    "description": f"Materials from {f.name}",
                    "memories": room_memories,
                })

    # Sequential edges between lectures
    edges = []
    for i in range(len(rooms_data) - 1):
        edges.append({
            "from_room": rooms_data[i]["topic"],
            "to_room": rooms_data[i + 1]["topic"],
            "relation": "builds_on",
            "weight": 0.85,
        })

    compiled_spec: Dict[str, Any] = {
        "id": course_id,
        "name": course_name,
        "author": spec.get("author", "Course Instructors"),
        "version": spec.get("version", "1.0.0"),
        "era": spec.get("era", "Current Academic Term"),
        "tagline": spec.get("tagline", f"Interactive Socratic Coursework Palace for {course_name}"),
        "description": spec.get("description", f"Self-contained 24/7 coursework and office hours memory palace for {course_name}."),
        "system_prompt_hint": spec.get(
            "system_prompt_hint",
            f"You are a warm, rigorous, and supportive Socratic teaching assistant for {course_name}. "
            f"Help students understand concepts deeply by guiding them with intuition, asking probing "
            f"questions, and citing the course slides, transcripts, and reading notes stored in your palace.",
        ),
        "suggested_prompts": spec.get("suggested_prompts", [
            f"Can you explain the main concepts introduced in {rooms_data[0]['topic']}?" if rooms_data else "What is the overview of this course?",
            "What are the most common misconceptions on this topic?",
            "Can you test my understanding with a challenging question?",
        ]),
        "rooms": rooms_data,
        "edges": edges,
    }
    return compiled_spec


def _ingest_file_into_memories(file_path: Path, memories: List[str]):
    """Extracts text chunks from supported document types and appends to memories."""
    ext = file_path.suffix.lower()
    fname = file_path.name

    if ext == ".pdf":
        pages = extract_pdf_pages(file_path)
        for page_num, text in pages:
            chunks = chunk_text(text, max_chars=450)
            for ch in chunks:
                memories.append(f"[{fname}: Page {page_num}] {ch}")

    elif ext in (".vtt", ".srt"):
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        clean = clean_vtt_or_srt(raw)
        chunks = chunk_text(clean, max_chars=500)
        for ch in chunks:
            memories.append(f"[Lecture Transcript: {fname}] {ch}")

    elif ext == ".ipynb":
        nb_text = extract_notebook_text(file_path)
        chunks = chunk_text(nb_text, max_chars=500)
        for ch in chunks:
            memories.append(f"[{fname}] {ch}")

    elif ext in (".md", ".txt", ".rst"):
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(raw, max_chars=500)
        for ch in chunks:
            memories.append(f"[{fname}] {ch}")


# ── Universal Loader & Compiler ──────────────────────────────────────────────

def load_spec_file(file_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Load specification from a YAML or JSON file."""
    if not file_path:
        return {}
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            return json.loads(text)
    else:
        return json.loads(text)


def compile_persona_spec(
    spec: Dict[str, Any],
    output_bundle_path: Optional[str] = None,
    install: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> str:
    """
    Compiles a persona dictionary spec into a .smritibundle archive.
    """
    from smriti_memcore.core import SMRITI
    from smriti_memcore.models import (
        Memory,
        MemorySource,
        MemoryStatus,
        Modality,
        SalienceScore,
        SmritiConfig,
        Visibility,
    )
    from smriti_memcore.personas.bundle import BUNDLE_EXTENSION, export_bundle, import_bundle
    from smriti_memcore.personas.manifest import PersonaManifest

    persona_id = spec.get("id") or "custom-persona"
    name = spec.get("name") or persona_id.replace("-", " ").title()
    version = str(spec.get("version", "1.0.0"))
    author = spec.get("author", "SMRITI Curators")
    era = str(spec.get("era", ""))
    tagline = spec.get("tagline", "")
    description = spec.get("description", "")
    system_prompt_hint = spec.get("system_prompt_hint", "")
    suggested_prompts = spec.get("suggested_prompts", [])
    metadata = spec.get("metadata", {})
    if "farewell" in spec and "farewell" not in metadata:
        metadata["farewell"] = spec["farewell"]

    if not output_bundle_path:
        output_bundle_path = f"{persona_id}{BUNDLE_EXTENSION}"
    if not output_bundle_path.endswith(BUNDLE_EXTENSION):
        output_bundle_path += BUNDLE_EXTENSION

    output_bundle_path = os.path.abspath(output_bundle_path)

    # Staging storage for the palace compilation
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "palace_build")
        os.makedirs(storage_path, exist_ok=True)

        config = SmritiConfig(
            storage_path=storage_path,
            working_memory_slots=7,
            embedding_model=embedding_model,
        )

        logger.info(f"Compiling Persona Palace for '{name}' ({persona_id})...")
        smriti = SMRITI(config=config)

        created_rooms: Dict[str, Any] = {}
        room_topic_to_id: Dict[str, str] = {}
        core_room_topics: List[str] = []

        # 1. Create Semantic Rooms
        raw_rooms = spec.get("rooms", [])
        for r_def in raw_rooms:
            topic = r_def.get("topic") if isinstance(r_def, dict) else str(r_def)
            if not topic:
                continue
            room = smriti.palace.create_room(topic=topic)
            created_rooms[topic] = room
            room_topic_to_id[topic] = room.id
            core_room_topics.append(topic)

        # 2. Place Memories
        total_pinned = 0
        for r_def in raw_rooms:
            if not isinstance(r_def, dict):
                continue
            topic = r_def.get("topic")
            room = created_rooms.get(topic)
            if not room:
                continue

            memories = r_def.get("memories", [])
            for m_item in memories:
                if isinstance(m_item, dict):
                    content = m_item.get("content", "").strip()
                    confidence = float(m_item.get("confidence", 0.95))
                    source_str = m_item.get("source", "direct")
                else:
                    content = str(m_item).strip()
                    confidence = 0.95
                    source_str = "direct"

                if not content:
                    continue

                source_enum = MemorySource.USER_STATED if source_str == "user_stated" else MemorySource.DIRECT
                embedding = smriti.vector_store.embed(content)

                mem = Memory(
                    content=content,
                    modality=Modality.TEXT,
                    source=source_enum,
                    status=MemoryStatus.PINNED,
                    visibility=Visibility.SHARED,
                    confidence=confidence,
                    room_id=room.id,
                    salience=SalienceScore(
                        surprise=0.3,
                        relevance=0.8,
                        emotional=0.4,
                        novelty=0.6,
                        utility=0.9,
                    ),
                    strength=1.8,
                    embedding=embedding,
                )
                smriti.palace.place_memory(mem)
                total_pinned += 1

        # 3. Add Typed Semantic Edges
        raw_edges = spec.get("edges", [])
        for e_def in raw_edges:
            from_t = e_def.get("from_room")
            to_t = e_def.get("to_room")
            relation = e_def.get("relation", "complements")
            weight = float(e_def.get("weight", 0.8))

            from_id = room_topic_to_id.get(from_t)
            to_id = room_topic_to_id.get(to_t)

            if from_id and to_id:
                smriti.palace.link_rooms(
                    room_a_id=from_id,
                    room_b_id=to_id,
                    relationship=relation,
                    strength=weight,
                )

        # 4. Save and Export
        smriti.save()

        manifest = PersonaManifest(
            id=persona_id,
            name=name,
            version=version,
            author=author,
            era=era,
            tagline=tagline,
            description=description,
            core_rooms=core_room_topics,
            suggested_prompts=suggested_prompts,
            system_prompt_hint=system_prompt_hint,
            total_memories=total_pinned,
            total_rooms=len(created_rooms),
            metadata=metadata,
        )

        os.makedirs(os.path.dirname(output_bundle_path) or ".", exist_ok=True)
        export_bundle(smriti, manifest, output_bundle_path)
        smriti.close()

    # 5. Optional auto-install to local cache
    if install:
        personas_cache = os.path.expanduser("~/.smriti/personas")
        target_dir = os.path.join(personas_cache, persona_id)
        import_bundle(output_bundle_path, target_dir)
        logger.info(f"Installed persona '{persona_id}' into {target_dir}")

    return output_bundle_path


def compile_path(
    source_path: Union[str, Path],
    output_bundle_path: Optional[str] = None,
    install: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    High-level entry point: compiles either a spec file (YAML/JSON)
    OR a coursework directory (with PDFs, transcripts, and notes) into a .smritibundle.
    """
    path = Path(source_path).expanduser().resolve()
    if path.is_dir():
        spec = parse_coursework_directory(path)
    else:
        spec = load_spec_file(path)

    out = compile_persona_spec(
        spec=spec,
        output_bundle_path=output_bundle_path,
        install=install,
    )
    return out, spec
