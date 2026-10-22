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
logging.getLogger("pypdf").setLevel(logging.ERROR)


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
            raw = page.extract_text() or ""
            # Handle surrogates and clean whitespace safely
            clean = raw.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            clean = re.sub(r"\s+", " ", clean).strip()
            # Filter out near-empty slide title pages (< 50 chars)
            if len(clean) >= 50:
                pages.append((idx, clean))
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

def normalize_lecture_key(key: str) -> str:
    """Normalizes lecture labels like Lecture 1, lec01 -> Lecture 01."""
    m = re.match(r"Lecture\s+(\d+)([a-zA-Z]?)", key, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        suffix = m.group(2).upper()
        return f"Lecture {num:02d}{suffix}".strip()
    return key


def parse_coursework_directory(dir_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Scans a folder of course materials (PDFs, transcripts, notes, markdown)
    and constructs a structured Persona Palace specification.
    """
    root = Path(dir_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Path is not an existing directory: {root}")

    # Check for course metadata (data.json, course.yaml, manifest.json)
    meta_file = None
    for name in ["data.json", "course.yaml", "course.json", "manifest.yaml", "manifest.json"]:
        p = root / name
        if p.exists():
            meta_file = p
            break
        # Also check immediate subfolders (e.g. 15.773-spring-2024/data.json)
        subs = list(root.glob(f"*/{name}"))
        if subs:
            meta_file = subs[0]
            break

    raw_meta = load_spec_file(meta_file) if meta_file else {}

    # Build file-to-title map from any data.json metadata in course tree (MIT OCW style)
    file_to_title: Dict[str, str] = {}
    for jf in root.rglob("data.json"):
        try:
            jdata = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
            title = jdata.get("title")
            if not title:
                continue
            for key in ["captions_file", "transcript_file", "file"]:
                target = jdata.get(key)
                if target:
                    fname = Path(target).name
                    file_to_title[fname] = title
        except Exception:
            pass

    # Extract metadata (with MIT OCW / academic portal support)
    course_name = raw_meta.get("course_title") or raw_meta.get("name") or root.name.replace("-", " ").replace("_", " ").title()
    course_id = raw_meta.get("id")
    if not course_id:
        if raw_meta.get("primary_course_number"):
            course_id = f"mit-{raw_meta['primary_course_number'].lower().replace('.', '-')}"
        elif raw_meta.get("course_title"):
            course_id = re.sub(r"[^a-z0-9]+", "-", raw_meta["course_title"].lower()).strip("-")
        else:
            course_id = root.name.lower().replace(" ", "-").replace("_", "-")
    
    author = "Course Instructors"
    if "instructors" in raw_meta and isinstance(raw_meta["instructors"], list) and raw_meta["instructors"]:
        first_inst = raw_meta["instructors"][0]
        if isinstance(first_inst, dict):
            author = first_inst.get("title") or f"{first_inst.get('first_name', '')} {first_inst.get('last_name', '')}".strip()
    elif raw_meta.get("author"):
        author = raw_meta["author"]

    course_desc = raw_meta.get("course_description") or raw_meta.get("description") or f"Coursework and office hours palace for {course_name}."

    # Scan for course files
    supported_exts = {".pdf", ".webvtt", ".vtt", ".srt", ".md", ".txt", ".ipynb"}
    ignored_names = {"robots.txt", "sitemap.xml", "favicon.ico", "license.txt", ".ds_store", "content_map.json"}
    files_to_ingest = []
    seen_signatures = set()

    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.name.startswith(".") or f.name.lower() in ignored_names:
            continue
        if f.suffix.lower() not in supported_exts:
            continue
        # Skip build/cache assets
        if any(part in f.parts for part in ("node_modules", ".git", ".smriti")):
            continue
        # Deduplicate identical files across root and subfolders
        sig = (f.name, f.stat().st_size)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        files_to_ingest.append(f)

    # Group files into Rooms (by lecture pattern, title mapping, or subfolder)
    grouped_rooms: Dict[str, List[Path]] = {}

    def _determine_room_topic(f: Path) -> str:
        title = file_to_title.get(f.name) or file_to_title.get(f.name.lower()) or ""
        targets = [title, f.name] if title else [f.name]
        for text in targets:
            m = re.search(r"(?:lec(?:ture)?)[-_:\s]?([0-9]{1,3}[a-zA-Z]|[0-9]{1,3}\.[0-9]+|[0-9]{1,3})", text, re.IGNORECASE)
            if m:
                val = m.group(1)
                if "." in val:
                    return f"Lecture {val}"
                num_part = re.sub(r"[a-zA-Z]", "", val)
                alpha_part = re.sub(r"[0-9]", "", val).upper()
                if num_part:
                    return f"Lecture {int(num_part):02d}{alpha_part}"
                return f"Lecture {val}"
            m2 = re.match(r"^\s*([0-9]{1,3}):\s*", text)
            if m2:
                num = int(m2.group(1))
                return f"Lecture {num:02d}"

        parent_name = f.parent.name
        if parent_name and parent_name not in (root.name, "resources", "static_resources", "download"):
            return parent_name.replace("-", " ").replace("_", " ").title()
        return f.stem.replace("-", " ").replace("_", " ").title()

    for f in files_to_ingest:
        room_topic = _determine_room_topic(f)
        grouped_rooms.setdefault(room_topic, []).append(f)

    rooms_data: List[Dict[str, Any]] = []

    for room_topic, flist in sorted(grouped_rooms.items()):
        room_memories: List[str] = []
        for f in flist:
            _ingest_file_into_memories(f, room_memories)

        if room_memories:
            rooms_data.append({
                "topic": room_topic,
                "description": f"Curriculum materials for {room_topic}",
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
        "author": author,
        "version": raw_meta.get("version", "1.0.0"),
        "era": raw_meta.get("term", raw_meta.get("era", "Current Term")),
        "tagline": f"Interactive Socratic Coursework Palace for {course_name}",
        "description": course_desc,
        "system_prompt_hint": (
            f"You are {author}, teaching {course_name}. "
            f"Adopt a supportive, rigorous, and Socratic pedagogical approach: validate the student's curiosity, "
            f"guide them toward first principles using practical intuition, and ask probing questions that "
            f"lead them to discover the answer. Ground your explanations directly in the lecture slides and transcripts stored in your palace."
        ),
        "suggested_prompts": [
            f"What are the core concepts covered in {rooms_data[0]['topic']}?" if rooms_data else "What is the overview of this course?",
            "Can you explain the intuition behind backpropagation and gradient descent?",
            "What is the difference between convolutional layers and fully connected layers?",
            "Can you quiz my understanding with a challenging conceptual question?",
        ],
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
                smriti.palace.place_memory(mem, room=room)
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
