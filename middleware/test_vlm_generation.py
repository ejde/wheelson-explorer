"""
wheelson-explorer · middleware · test_vlm_generation.py
Tests for the 2026-10-25 generation-split fix: the loop used to compare
in-flight VLM observations against planning_generation, which bumps on
recovery triggers and transient firmware-busy rejections. Those fire
repeatedly inside one ~20 s VLM round trip, so every result arrived
"generation changed" and was discarded, pinning the loop in degraded
local mode. The consumption check now compares against vlm_generation,
which advances only on safety-critical regime changes (persona switch,
human takeover/resume, safety event). These tests lock the consumption
contract that wiring relies on: a result is current when its generation
matches the VLM counter, whatever the planning counter did meanwhile,
and the age window still applies. The wiring itself (which loop sites
bump vlm_generation) is verified live: strategy_source flips to vlm
and generation discards stop.
"""

import unittest

from skills import observation_is_current, vlm_discard_reason


def _update(generation=5, captured_at=100.0):
    return {"generation": generation, "captured_at": captured_at}


class TestVlmGenerationSplit(unittest.TestCase):
    def test_result_current_when_vlm_generation_matches(self):
        # A result captured at VLM generation 5 is current at VLM
        # generation 5, even after a 19 s round trip. Planning-only
        # bumps (recovery, busy) never reach this check: the caller
        # passes the VLM counter, not the planning counter.
        update = _update(generation=5)
        self.assertIsNone(vlm_discard_reason(update, 5, 119.0, 30.0))
        self.assertTrue(observation_is_current(update, 5, 119.0, 30.0))

    def test_vlm_generation_advance_rejects_with_reason(self):
        # A safety-critical regime change DOES invalidate: once the
        # VLM counter has moved to 6, a result captured at 5 is out,
        # and the reason names the generation change.
        reason = vlm_discard_reason(_update(generation=5), 6, 105.0, 30.0)
        self.assertIsNotNone(reason)
        self.assertIn("generation", reason)
        self.assertFalse(observation_is_current(_update(generation=5), 6, 105.0, 30.0))

    def test_age_window_still_applies(self):
        # Splitting the counters must not weaken the age check: a
        # generation-matching result older than the 30 s window is out.
        reason = vlm_discard_reason(_update(generation=5, captured_at=100.0), 5, 131.0, 30.0)
        self.assertIsNotNone(reason)
        self.assertIn("31.0s", reason)


if __name__ == "__main__":
    unittest.main()
