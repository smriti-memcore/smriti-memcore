"""
SMRITI Persona Bundle Packaging.
Export, import, and load self-contained personality memory bundles (.smritibundle).
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
from typing import Optional, Tuple

from smriti_memcore.models import SmritiConfig
from smriti_memcore.personas.manifest import PersonaManifest

logger = logging.getLogger(__name__)

BUNDLE_EXTENSION = ".smritibundle"
MANIFEST_FILENAME = "manifest.json"


def export_bundle(
    smriti_instance,
    manifest: PersonaManifest,
    output_path: str,
) -> str:
    """
    Exports a SMRITI instance's memory state into a self-contained persona bundle.

    Args:
        smriti_instance: Active SMRITI instance to export.
        manifest: PersonaManifest metadata describing the persona.
        output_path: Destination path (e.g. 'einstein.smritibundle' or a folder).

    Returns:
        The canonical path to the exported bundle.
    """
    smriti_instance.save()

    # Update counts in manifest
    manifest.total_memories = len(smriti_instance.palace.memories)
    manifest.total_rooms = len(smriti_instance.palace.rooms)

    src_storage = smriti_instance.config.storage_path

    # Check whether output is archive or directory
    is_archive = output_path.endswith(BUNDLE_EXTENSION) or output_path.endswith(".tar.gz")

    if is_archive:
        # Create temp staging directory
        staging_dir = output_path + ".staging"
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=True)

        try:
            _copy_storage_components(src_storage, staging_dir)
            manifest.save(os.path.join(staging_dir, MANIFEST_FILENAME))

            # Package as tar.gz
            with tarfile.open(output_path, "w:gz") as tar:
                for item in os.listdir(staging_dir):
                    item_path = os.path.join(staging_dir, item)
                    tar.add(item_path, arcname=item)
            logger.info(f"Exported persona bundle archive to {output_path}")
            return output_path
        finally:
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir)
    else:
        # Export as directory
        os.makedirs(output_path, exist_ok=True)
        _copy_storage_components(src_storage, output_path)
        manifest.save(os.path.join(output_path, MANIFEST_FILENAME))
        logger.info(f"Exported persona bundle directory to {output_path}")
        return output_path


def import_bundle(
    bundle_path: str,
    target_storage_path: str,
) -> PersonaManifest:
    """
    Imports a persona bundle into a target storage directory.

    Args:
        bundle_path: Path to .smritibundle archive or bundle directory.
        target_storage_path: Storage path where the persona memory should be unpacked.

    Returns:
        The PersonaManifest of the imported bundle.
    """
    os.makedirs(target_storage_path, exist_ok=True)

    if tarfile.is_tarfile(bundle_path):
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(target_storage_path)
    elif os.path.isdir(bundle_path):
        for item in os.listdir(bundle_path):
            s = os.path.join(bundle_path, item)
            d = os.path.join(target_storage_path, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
    else:
        raise ValueError(f"Invalid bundle path: {bundle_path} is neither a tar archive nor directory")

    manifest_file = os.path.join(target_storage_path, MANIFEST_FILENAME)
    if not os.path.exists(manifest_file):
        raise FileNotFoundError(f"Missing {MANIFEST_FILENAME} in imported bundle at {target_storage_path}")

    manifest = PersonaManifest.load(manifest_file)
    logger.info(f"Imported persona '{manifest.name}' ({manifest.id}) into {target_storage_path}")
    return manifest


def _copy_storage_components(src_storage: str, dest_dir: str):
    """Copies palace, vectors, and episodes if present."""
    for component in ["palace", "vectors", "episodes"]:
        src_comp = os.path.join(src_storage, component)
        if os.path.exists(src_comp):
            dest_comp = os.path.join(dest_dir, component)
            if os.path.exists(dest_comp):
                shutil.rmtree(dest_comp)
            shutil.copytree(src_comp, dest_comp)
