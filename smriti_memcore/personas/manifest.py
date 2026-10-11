"""
SMRITI Persona Manifest.
Defines metadata, structure, and prompt hints for pre-curated personality memory bundles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PersonaManifest:
    """Metadata describing a pre-curated personality memory bundle."""
    id: str                                  # e.g. "albert-einstein"
    name: str                                # e.g. "Albert Einstein"
    version: str = "1.0.0"                   # Semantic version
    author: str = "SMRITI Curators"
    era: str = ""                            # e.g. "1879–1955"
    tagline: str = ""                        # One-line summary
    description: str = ""                    # Full bio and domain scope
    core_rooms: List[str] = field(default_factory=list)
    suggested_prompts: List[str] = field(default_factory=list)
    system_prompt_hint: str = ""             # Voice, tone, epistemic framing
    total_memories: int = 0
    total_rooms: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersonaManifest:
        valid_keys = {
            "id", "name", "version", "author", "era", "tagline",
            "description", "core_rooms", "suggested_prompts",
            "system_prompt_hint", "total_memories", "total_rooms",
            "created_at", "metadata",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> PersonaManifest:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
