"""Tests for user-defined personas (middleware/personas.py)."""

import tempfile
import unittest
from pathlib import Path

import config
from personas import (
    apply_custom_personas,
    describe_personas,
    load_custom_personas,
    restore_builtin_personas,
)

EXAMPLE = Path(__file__).with_name("personas.example.yaml")

MINIMAL = """
testbot:
  name: "Testbot"
  system_prompt: "You are Testbot."
  base_speed: slow
  safety_weight: 1.0
  curiosity_weight: 1.0
  perimeter_weight: 1.0
  preferred_wall: left
  forward_burst_ms: 500
  hold_bias: 0.5
"""

FULL = """
finch:
  name: "Finch"
  emoji: "🦉"
  color: "#a16207"
  system_prompt: "You are Finch."
  move_bias: "Curious"
  base_speed: medium
  safety_weight: 1.2
  curiosity_weight: 1.5
  perimeter_weight: 0.6
  preferred_wall: none
  forward_burst_ms: 800
  hold_bias: 0.3
  turn_duration_ms: 550
  salience: "Prioritize interesting things."
  commentary_style: "warm narration"
"""


def _write_tmp(content):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name


class TestLoadCustomPersonas(unittest.TestCase):
    def tearDown(self):
        restore_builtin_personas()

    def test_minimal_applies_defaults(self):
        customs = load_custom_personas(_write_tmp(MINIMAL))
        bot = customs["testbot"]
        self.assertEqual(bot["emoji"], "\U0001f916")
        self.assertEqual(bot["color"], "#888888")
        self.assertEqual(bot["move_bias"], "")
        self.assertEqual(bot["turn_duration_ms"], 600)
        self.assertEqual(bot["policy"]["base_speed"], "slow")
        self.assertEqual(bot["policy"]["preferred_wall"], "left")

    def test_full_persona_preserved(self):
        customs = load_custom_personas(_write_tmp(FULL))
        finch = customs["finch"]
        self.assertEqual(finch["name"], "Finch")
        self.assertEqual(finch["policy"]["curiosity_weight"], 1.5)
        self.assertEqual(finch["turn_duration_ms"], 550)
        self.assertEqual(finch["salience"], "Prioritize interesting things.")

    def test_unknown_field_rejected(self):
        bad = MINIMAL.replace("hold_bias: 0.5", "hold_bias: 0.5\n  curiosity_wieght: 2.0")
        with self.assertRaisesRegex(ValueError, "curiosity_wieght"):
            load_custom_personas(_write_tmp(bad))

    def test_bad_values_rejected(self):
        cases = [
            MINIMAL.replace("slow", "warp"),
            MINIMAL.replace("left", "up"),
            MINIMAL.replace("hold_bias: 0.5", "hold_bias: 1.5"),
            MINIMAL.replace("forward_burst_ms: 500", "forward_burst_ms: 0"),
            MINIMAL.replace("safety_weight: 1.0", "safety_weight: -1"),
        ]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_custom_personas(_write_tmp(bad))

    def test_missing_required_rejected(self):
        bad = MINIMAL.replace('  system_prompt: "You are Testbot."\n', "")
        with self.assertRaisesRegex(ValueError, "system_prompt"):
            load_custom_personas(_write_tmp(bad))

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_custom_personas("/nonexistent/personas.yaml"), {})

    def test_malformed_yaml_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid YAML"):
            load_custom_personas(_write_tmp("testbot: [unclosed"))

    def test_example_file_loads_and_contains_finch(self):
        customs = load_custom_personas(EXAMPLE)
        finch = customs["finch"]
        self.assertEqual(finch["policy"]["base_speed"], "medium")
        self.assertEqual(finch["policy"]["curiosity_weight"], 1.5)
        self.assertEqual(finch["turn_duration_ms"], 550)


class TestApplyAndRestore(unittest.TestCase):
    def tearDown(self):
        restore_builtin_personas()

    def test_apply_merges_and_restore_removes(self):
        applied = apply_custom_personas(load_custom_personas(_write_tmp(FULL)))
        self.assertEqual(applied, ["finch"])
        self.assertIn("finch", config.PERSONALITIES)
        self.assertEqual(config.PERSONA_POLICIES["finch"]["curiosity_weight"], 1.5)
        self.assertEqual(config.TURN_DURATION_MS["finch"], 550)
        self.assertEqual(describe_personas()["finch"]["source"], "custom")
        self.assertEqual(describe_personas()["benson"]["source"], "builtin")
        restore_builtin_personas()
        self.assertNotIn("finch", config.PERSONALITIES)
        self.assertEqual(config.PERSONA_POLICIES["benson"]["safety_weight"], 1.7)

    def test_apply_override_then_restore_builtin(self):
        override = """
benson:
  name: "BENSON"
  system_prompt: "You are BENSON, slower."
  base_speed: slow
  safety_weight: 9.9
  curiosity_weight: 0.5
  perimeter_weight: 0.4
  preferred_wall: none
  forward_burst_ms: 700
  hold_bias: 0.8
"""
        apply_custom_personas(load_custom_personas(_write_tmp(override)))
        self.assertEqual(config.PERSONA_POLICIES["benson"]["safety_weight"], 9.9)
        restore_builtin_personas()
        self.assertEqual(config.PERSONA_POLICIES["benson"]["safety_weight"], 1.7)

    def test_failed_load_leaves_current_set_untouched(self):
        apply_custom_personas(load_custom_personas(_write_tmp(FULL)))
        self.assertIn("finch", config.PERSONALITIES)
        with self.assertRaises(ValueError):
            apply_custom_personas(path=_write_tmp("testbot: [unclosed"))
        self.assertIn("finch", config.PERSONALITIES)


if __name__ == "__main__":
    unittest.main()
