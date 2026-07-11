"""
wheelson-explorer · middleware · test_explorer.py
Unit tests and simulation mocks for middleware logic.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from config import FIRMWARE_VISUAL_SAFETY
from scene import _reconcile_scene_with_telemetry, build_strategy_intent
from motion import WheelsonClient, MotionSupervisor, MotionPlan
from planner import IntentPlanner


class TestTelemetryReconciliation(unittest.TestCase):
    def test_vlm_source_trusts_vlm(self):
        """Verify that when the source is 'vlm', the VLM's hazard evaluation is trusted and not overridden by noisy visual_obstacle."""
        telemetry = {
            "visual_obstacle": True,
            "obstacle_left_ratio": 0.45,
            "obstacle_right_ratio": 0.45,
            "nav_state": "CRUISE",
        }
        # Under VLM source, hazard = 'none' should be preserved despite visual_obstacle=True
        frontier, traversability, hazard = _reconcile_scene_with_telemetry(
            frontier="forward",
            traversability="high",
            hazard="none",
            telemetry=telemetry,
            source="vlm",
        )
        self.assertEqual(hazard, "none")
        self.assertEqual(frontier, "forward")
        self.assertEqual(traversability, "high")

    def test_local_source_falls_back_to_telemetry(self):
        """Verify that when the source is 'local' (fallback), noisy telemetry overrides hazard evaluation."""
        telemetry = {
            "visual_obstacle": True,
            "obstacle_left_ratio": 0.45,
            "obstacle_right_ratio": 0.45,
            "nav_state": "CRUISE",
        }
        # Under local source, visual_obstacle=True should override hazard to 'soft' (or 'hard' if side_peak >= 0.35)
        frontier, traversability, hazard = _reconcile_scene_with_telemetry(
            frontier="forward",
            traversability="high",
            hazard="none",
            telemetry=telemetry,
            source="local",
        )
        self.assertEqual(hazard, "hard")  # overridden because visual_obstacle=True and side_peak = 0.45 >= 0.35
        self.assertEqual(frontier, "blocked")
        self.assertEqual(traversability, "medium")


class TestMotionSupervisorSafetyInjection(unittest.IsolatedAsyncioTestCase):
    async def test_post_move_injects_visual_safety(self):
        """Verify that _post_move automatically injects the visual_safety state from config into all move payloads."""
        mock_http = MagicMock()
        mock_http.post = AsyncMock()
        
        # Mock response from HTTP client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "action": "forward"}
        mock_http.post.return_value = mock_resp

        client = WheelsonClient(mock_http)
        
        # Set config settings to False for test
        with patch("config.FIRMWARE_VISUAL_SAFETY", False):
            payload = {"action": "forward", "duration_ms": 600}
            await client._post_move(payload)
            
            # Assert that "visual_safety": "false" was injected
            mock_http.post.assert_called_once()
            called_json = mock_http.post.call_args[1]["json"]
            self.assertEqual(called_json["visual_safety"], "false")
            self.assertEqual(called_json["action"], "forward")

        # Reset mock and test with True
        mock_http.post.reset_mock()
        with patch("config.FIRMWARE_VISUAL_SAFETY", True):
            payload = {"action": "left", "duration_ms": 500}
            await client._post_move(payload)
            
            # Assert that "visual_safety": "true" was injected
            mock_http.post.assert_called_once()
            called_json = mock_http.post.call_args[1]["json"]
            self.assertEqual(called_json["visual_safety"], "true")
            self.assertEqual(called_json["action"], "left")


class TestPlannerLatching(unittest.TestCase):
    def test_planner_latches_turn_plans(self):
        """Verify that turn plans are latched and returned during the latch duration."""
        planner = IntentPlanner()
        planner.set_persona("benson")
        
        intent = MagicMock()
        intent.hazard = "none"
        intent.frontier = "left"
        intent.traversability = "high"
        intent.source = "vlm"
        
        telemetry = {
            "visual_obstacle": False,
            "obstacle_left_ratio": 0.0,
            "obstacle_right_ratio": 0.0,
        }

        now = 100.0
        # First plan - benson should plan a turn to the left because frontier is left
        plan = planner.plan(intent, telemetry, cycle_count=20, now_ts=now)
        self.assertEqual(plan.action, "left")
        self.assertIn("benson:frontier_turn", plan.reason)
        
        # Second plan - immediately after, latch should be active, so same plan is returned
        plan2 = planner.plan(intent, telemetry, cycle_count=21, now_ts=now + 0.2)
        self.assertEqual(plan2.action, "left")
        self.assertIn("latched", plan2.reason)
        
        # Third plan - after latch duration (0.9s for turns), latch should be expired
        # Benson's forward chain streak resets, let's say frontier is now forward
        intent.frontier = "forward"
        plan3 = planner.plan(intent, telemetry, cycle_count=22, now_ts=now + 1.5)
        self.assertEqual(plan3.action, "forward")
        self.assertNotIn("latched", plan3.reason)

    def test_benson_safety_pause_time_based(self):
        """Verify that Benson's safety pause triggers after 9.6s of forward movement, regardless of cycle count."""
        planner = IntentPlanner()
        planner.set_persona("benson")

        intent = MagicMock()
        intent.hazard = "none"
        intent.frontier = "forward"
        intent.traversability = "high"
        intent.source = "vlm"

        telemetry = {
            "visual_obstacle": False,
            "obstacle_left_ratio": 0.0,
            "obstacle_right_ratio": 0.0,
        }

        now = 100.0
        # Start moving forward (first call)
        plan = planner.plan(intent, telemetry, cycle_count=0, now_ts=now)
        self.assertEqual(plan.action, "forward")
        
        # Advance time by 5.0 seconds (below 9.6s limit)
        plan2 = planner.plan(intent, telemetry, cycle_count=10, now_ts=now + 5.0)
        self.assertEqual(plan2.action, "forward")
        
        # Advance time by another 5.0 seconds (total 10.0s, which is >= 9.6s)
        # It should trigger the safety pause
        plan3 = planner.plan(intent, telemetry, cycle_count=20, now_ts=now + 10.0)
        self.assertEqual(plan3.action, "stop")
        self.assertIn("benson:safety_pause", plan3.reason)


if __name__ == "__main__":
    unittest.main()
