"""
SMRITI Persona Compiler.
Compiles knowledge specifications (YAML, JSON, or markdown folders) into
self-contained .smritibundle cognitive artifacts.

Enables professors, counsellors, and domain experts to distribute their
cognitive palaces with zero code.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def load_spec_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load specification from a YAML or JSON file."""
    path = Path(file_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text)
        except ImportError:
            # Fallback basic json parser if yaml not installed
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

    Args:
        spec: Dictionary describing the persona, rooms, and memories.
        output_bundle_path: Target path for the .smritibundle file.
        install: If True, also installs the compiled bundle into ~/.smriti/personas/.
        embedding_model: Embedding model for semantic vectors.

    Returns:
        The canonical path to the generated bundle.
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
    from smriti_memcore.palace import TypedEdge
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
