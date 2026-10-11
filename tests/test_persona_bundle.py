"""
Test SMRITI Persona Bundles and Albert Einstein Memory Palace.
"""

from __future__ import annotations

import os
import shutil
import unittest

from smriti_memcore.core import SMRITI
from smriti_memcore.personas.manifest import PersonaManifest
from smriti_memcore.personas.bundle import import_bundle
from smriti_memcore.personas.hub import list_local_personas, pull_persona


class TestPersonaBundle(unittest.TestCase):
    def setUp(self):
        self.test_work_dir = os.path.abspath("build/test_einstein_session")
        if os.path.exists(self.test_work_dir):
            shutil.rmtree(self.test_work_dir)
        os.makedirs(self.test_work_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_work_dir):
            shutil.rmtree(self.test_work_dir)

    def test_einstein_bundle_loading_and_recall(self):
        bundle_path = os.path.abspath("data/personas/albert-einstein.smritibundle")
        self.assertTrue(os.path.exists(bundle_path), f"Bundle not found at {bundle_path}")

        # Test from_bundle
        smriti, manifest = SMRITI.from_bundle(bundle_path, storage_path=self.test_work_dir)
        self.assertEqual(manifest.id, "albert-einstein")
        self.assertEqual(manifest.name, "Albert Einstein")
        self.assertGreater(len(smriti.palace.rooms), 0)
        self.assertGreater(len(smriti.palace.memories), 0)

        # Test 1: Thought experiment on light beam
        recalled = smriti.recall("What thought experiment did you conceive about a beam of light?")
        self.assertTrue(len(recalled) > 0)
        top_text = " ".join([m.content for m in recalled])
        self.assertIn("Aarau", top_text)
        self.assertIn("beam of light", top_text)

        # Test 2: Dice quote & quantum mechanics
        recalled_dice = smriti.recall("Why do you say God does not play dice?")
        top_dice_text = " ".join([m.content for m in recalled_dice])
        self.assertIn("dice", top_dice_text)
        self.assertIn("Max Born", top_dice_text)

        # Test 3: Happiest thought of my life
        recalled_happiest = smriti.recall("What was the happiest thought of your life?")
        top_happy_text = " ".join([m.content for m in recalled_happiest])
        self.assertIn("happiest thought", top_happy_text.lower())
        self.assertIn("falls freely", top_happy_text)

        # Test 4: Spinoza's God
        recalled_god = smriti.recall("Do you believe in God?")
        top_god_text = " ".join([m.content for m in recalled_god])
        self.assertIn("Spinoza", top_god_text)

        # Test 5: Letter to Roosevelt
        recalled_fdr = smriti.recall("What was your letter to President Roosevelt about?")
        top_fdr_text = " ".join([m.content for m in recalled_fdr])
        self.assertIn("Roosevelt", top_fdr_text)
        self.assertIn("Szilárd", top_fdr_text)

        smriti.close()


if __name__ == "__main__":
    unittest.main()
