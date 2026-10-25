"""
wheelson-explorer · middleware · test_vlm_observation.py
Tests for the 2026-10-25 VLM validation fix: the old 18 s consumption
window discarded normal 18-20 s OpenRouter round trips (under the 20 s
timeout, over the TTL), pinning the loop in degraded local mode.
"""

import unittest

import config
from skills import observation_is_current, vlm_discard_reason


def _update(generation=3, captured_at=100.0):
    return {"generation": generation, "captured_at": captured_at}


class TestVlmDiscardReason(unittest.TestCase):
    def test_fresh_update_is_current(self):
        update = _update()
        self.assertIsNone(vlm_discard_reason(update, 3, 105.0, 30.0))
        self.assertTrue(observation_is_current(update, 3, 105.0, 30.0))

    def test_typical_vlm_latency_accepted_under_new_window(self):
        # A 19 s round trip: inside the 20 s provider timeout, but the old
        # 18 s window discarded it. The new 30 s window accepts it.
        update = _update(captured_at=100.0)
        self.assertIsNone(vlm_discard_reason(update, 3, 119.0, 30.0))
        reason = vlm_discard_reason(update, 3, 119.0, 18.0)
        self.assertIsNotNone(reason)
        self.assertIn("19.0s", reason)

    def test_age_over_new_window_rejected(self):
        reason = vlm_discard_reason(_update(captured_at=100.0), 3, 131.0, 30.0)
        self.assertIsNotNone(reason)
        self.assertIn("31.0s", reason)
        self.assertFalse(observation_is_current(_update(captured_at=100.0), 3, 131.0, 30.0))

    def test_boundary_age_equals_ttl_rejected(self):
        # The check is strict: age must be < ttl.
        self.assertIsNotNone(vlm_discard_reason(_update(captured_at=100.0), 3, 130.0, 30.0))

    def test_generation_mismatch_named(self):
        reason = vlm_discard_reason(_update(generation=2), 3, 105.0, 30.0)
        self.assertIsNotNone(reason)
        self.assertIn("generation", reason)

    def test_missing_captured_at_named_not_crash(self):
        reason = vlm_discard_reason({"generation": 3}, 3, 105.0, 30.0)
        self.assertIsNotNone(reason)
        self.assertIn("captured_at", reason)


class TestVlmObservationWindowConfig(unittest.TestCase):
    def test_window_covers_provider_timeout(self):
        # The invariant that prevents the squeeze from returning: the
        # consumption window must exceed the slowest allowed VLM call.
        self.assertGreater(
            config.VLM_OBSERVATION_MAX_AGE_SEC, config.OPENROUTER_VLM_TIMEOUT_SEC
        )
        self.assertGreater(
            config.VLM_OBSERVATION_MAX_AGE_SEC, config.SEMANTIC_TTL_SEC
        )


if __name__ == "__main__":
    unittest.main()
