"""
SMRITI Persona Hub.
Registry and on-demand download manager for pre-curated persona memory artifacts.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from typing import Dict, List, Optional, Tuple

from smriti_memcore.personas.bundle import (
    BUNDLE_EXTENSION,
    MANIFEST_FILENAME,
    import_bundle,
)
from smriti_memcore.personas.manifest import PersonaManifest

logger = logging.getLogger(__name__)

DEFAULT_PERSONAS_DIR = os.environ.get(
    "SMRITI_PERSONAS_DIR",
    os.path.expanduser("~/.smriti/personas"),
)

# Official registry index mapping persona IDs to official release URLs or metadata
OFFICIAL_REGISTRY: Dict[str, Dict[str, str]] = {
    "albert-einstein": {
        "name": "Albert Einstein",
        "description": "Historical memory palace of Albert Einstein (Relativity, Quantum debates, Spinoza's God, Pacifism)",
        "download_url": "https://github.com/smriti-memcore/smriti-memcore/releases/download/v1.4.26/albert-einstein.smritibundle",
    },
    "isaac-newton": {
        "name": "Isaac Newton",
        "description": "Historical memory palace of Sir Isaac Newton (Classical mechanics, Optics, Gravitation, Calculus)",
        "download_url": "https://github.com/smriti-memcore/smriti-memcore/releases/download/v1.4.26/isaac-newton.smritibundle",
    },
    "marcus-aurelius": {
        "name": "Marcus Aurelius",
        "description": "Philosophical memory palace of Marcus Aurelius (Meditations, Stoic ethics, Duty, Mortality)",
        "download_url": "https://github.com/smriti-memcore/smriti-memcore/releases/download/v1.4.26/marcus-aurelius.smritibundle",
    },
    "prof-systems": {
        "name": "Prof. Alex Chen",
        "description": "24/7 Socratic Office Hours companion for Distributed Systems, Consensus, and Storage Engines",
        "download_url": "https://github.com/smriti-memcore/smriti-memcore/releases/download/v1.4.26/prof-systems.smritibundle",
    },
    "mindful-counsellor": {
        "name": "Dr. Clara Vance",
        "description": "100% private, empathetic CBT & mindfulness companion for grounding, anxiety reframing, and calm",
        "download_url": "https://github.com/smriti-memcore/smriti-memcore/releases/download/v1.4.26/mindful-counsellor.smritibundle",
    },
}


def list_official_personas() -> Dict[str, Dict[str, str]]:
    """Returns the catalog of available personas in the official registry."""
    return OFFICIAL_REGISTRY.copy()


def list_local_personas(cache_dir: Optional[str] = None) -> List[PersonaManifest]:
    """Returns all personas installed in local cache directories."""
    candidate_roots = [cache_dir or DEFAULT_PERSONAS_DIR, os.path.abspath("data/personas/cache"), os.path.abspath("data/personas")]
    results: List[PersonaManifest] = []
    seen_ids = set()

    for root in candidate_roots:
        if not os.path.exists(root):
            continue
        for entry in os.listdir(root):
            folder = os.path.join(root, entry)
            manifest_file = os.path.join(folder, MANIFEST_FILENAME)
            if os.path.isdir(folder) and os.path.exists(manifest_file):
                try:
                    manifest = PersonaManifest.load(manifest_file)
                    if manifest.id not in seen_ids:
                        seen_ids.add(manifest.id)
                        results.append(manifest)
                except Exception as e:
                    logger.warning(f"Failed to load manifest from {manifest_file}: {e}")
    return results


def get_persona_dir(persona_id: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """Returns path to installed persona directory if it exists."""
    root = cache_dir or DEFAULT_PERSONAS_DIR
    candidate = os.path.join(root, persona_id)
    manifest_file = os.path.join(candidate, MANIFEST_FILENAME)
    if os.path.isdir(candidate) and os.path.exists(manifest_file):
        return candidate
    return None


def pull_persona(
    persona_id: str,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> Tuple[str, PersonaManifest]:
    """
    Ensures a persona is downloaded and ready in local storage.

    Args:
        persona_id: Identifier of the persona (e.g. 'albert-einstein' or alias 'einstein').
        cache_dir: Local directory to store personas (default: ~/.smriti/personas).
        force: If True, re-downloads and overwrites existing cached persona.

    Returns:
        Tuple of (persona_storage_dir, PersonaManifest)
    """
    normalized_id = persona_id.lower().strip()
    # Support simple aliases
    aliases = {
        "einstein": "albert-einstein",
        "newton": "isaac-newton",
        "marcus": "marcus-aurelius",
        "aurelius": "marcus-aurelius",
        "prof": "prof-systems",
        "systems": "prof-systems",
        "counsellor": "mindful-counsellor",
        "counselor": "mindful-counsellor",
        "mindful": "mindful-counsellor",
        "clara": "mindful-counsellor",
    }
    canonical_id = aliases.get(normalized_id, normalized_id)

    root = cache_dir or DEFAULT_PERSONAS_DIR
    try:
        os.makedirs(root, exist_ok=True)
    except (PermissionError, OSError):
        root = os.path.abspath("data/personas/cache")
        os.makedirs(root, exist_ok=True)

    # 0. Check if persona_id is a direct filepath to a .smritibundle archive or directory
    if persona_id.endswith(BUNDLE_EXTENSION) or "/" in persona_id or "\\" in persona_id:
        if not os.path.exists(persona_id):
            raise FileNotFoundError(f"Local bundle file not found: '{persona_id}'")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            temp_m = import_bundle(persona_id, td)
            cid = temp_m.id
        final_dir = os.path.join(root, cid)
        manifest = import_bundle(persona_id, final_dir)
        logger.info(f"Imported local bundle '{manifest.name}' into {final_dir}")
        return final_dir, manifest

    target_dir = os.path.join(root, canonical_id)

    if not force and os.path.exists(os.path.join(target_dir, MANIFEST_FILENAME)):
        manifest = PersonaManifest.load(os.path.join(target_dir, MANIFEST_FILENAME))
        logger.info(f"Persona '{canonical_id}' already present at {target_dir}")
        return target_dir, manifest

    # Check if local pre-built bundle exists in workspace/data or curation directory
    local_bundle = _find_local_candidate_bundle(canonical_id)
    if local_bundle:
        logger.info(f"Importing local bundle from {local_bundle} to {target_dir}")
        manifest = import_bundle(local_bundle, target_dir)
        return target_dir, manifest

    # Download from remote registry
    if canonical_id not in OFFICIAL_REGISTRY:
        raise ValueError(
            f"Persona '{canonical_id}' not found in registry. "
            f"Available personas: {list(OFFICIAL_REGISTRY.keys())}"
        )

    reg_entry = OFFICIAL_REGISTRY[canonical_id]
    url = reg_entry["download_url"]
    temp_archive = os.path.join(root, f"{canonical_id}{BUNDLE_EXTENSION}")

    logger.info(f"Downloading persona '{canonical_id}' from {url}...")
    try:
        urllib.request.urlretrieve(url, temp_archive)
        manifest = import_bundle(temp_archive, target_dir)
        return target_dir, manifest
    finally:
        if os.path.exists(temp_archive):
            os.remove(temp_archive)


def _find_local_candidate_bundle(persona_id: str) -> Optional[str]:
    """Searches workspace and common dirs for local pre-compiled bundle file."""
    search_paths = [
        f"data/personas/{persona_id}{BUNDLE_EXTENSION}",
        f"data/personas/{persona_id}",
        f"curation/{persona_id}/{persona_id}{BUNDLE_EXTENSION}",
        f"curation/{persona_id}",
        f"curation/{persona_id.replace('-', '_')}/{persona_id}{BUNDLE_EXTENSION}",
        f"curation/{persona_id.replace('-', '_')}",
        f"{persona_id}{BUNDLE_EXTENSION}",
    ]
    for p in search_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None
