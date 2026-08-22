"""
wheelson-explorer · middleware · test_explorer.py
Unit tests and simulation mocks for middleware logic.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from config import FIRMWARE_VISUAL_SAFETY
from scene import _reconcile_scene_with_telemetry, build_strategy_intent
from motion import WheelsonClient, MotionSupervisor, MotionPlan
from planner import IntentPlanner
from recovery import RecoveryController


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
        """Verify that Benson's safety pause triggers after the jittered forward limit."""
        planner = IntentPlanner(rng_seed=7)
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

        # 5 s in: below even the minimum jittered threshold (9.6 * 0.75 = 7.2)
        plan2 = planner.plan(intent, telemetry, cycle_count=10, now_ts=now + 5.0)
        self.assertEqual(plan2.action, "forward")

        # 15 s in: above the maximum jittered threshold (9.6 * 1.25 = 12.0)
        plan3 = planner.plan(intent, telemetry, cycle_count=20, now_ts=now + 15.0)
        self.assertEqual(plan3.action, "stop")
        self.assertIn("benson:safety_pause", plan3.reason)

    def test_jitter_bounds_and_episode_stability(self):
        """Jittered thresholds stay within ±25% of base and hold for an episode."""
        planner = IntentPlanner(rng_seed=1)
        for _ in range(20):
            sampled = planner._sample("k", 4.0)
            self.assertGreaterEqual(sampled, 4.0 * 0.75)
            self.assertLessEqual(sampled, 4.0 * 1.25)
            again = planner._sample("k", 4.0)
            self.assertEqual(sampled, again)  # stable within episode
        planner._begin_episode_if_new()  # last action not forward → resample
        fresh = planner._episode_samples.get("k")
        self.assertIsNone(fresh)


class TestPlannerSteering(unittest.TestCase):
    def _intent(self, frontier="forward"):
        intent = MagicMock()
        intent.hazard = "none"
        intent.frontier = frontier
        intent.traversability = "high"
        intent.source = "vlm"
        return intent

    def test_steer_attaches_on_asymmetric_ratios(self):
        planner = IntentPlanner(rng_seed=3)
        planner.set_persona("sir_david")
        telemetry = {
            "visual_obstacle": False,
            "obstacle_left_ratio": 0.0,
            "obstacle_right_ratio": 0.30,
        }
        plan = planner.plan(self._intent(), telemetry, cycle_count=5, now_ts=100.0)
        self.assertEqual(plan.action, "forward")
        self.assertGreater(plan.steer, 0.0, "obstacle on right should veer left (positive steer)")
        self.assertIn("steer", plan.reason)

    def test_steer_deadband(self):
        planner = IntentPlanner(rng_seed=3)
        planner.set_persona("sir_david")
        telemetry = {
            "visual_obstacle": False,
            "obstacle_left_ratio": 0.01,
            "obstacle_right_ratio": 0.02,
        }
        plan = planner.plan(self._intent(), telemetry, cycle_count=5, now_ts=100.0)
        self.assertEqual(plan.steer, 0.0)

    def test_steer_clamped_to_max(self):
        steer = IntentPlanner(rng_seed=3)._compute_steer({
            "obstacle_left_ratio": 0.0,
            "obstacle_right_ratio": 1.0,
        })
        self.assertLessEqual(steer, 0.6)


class TestPlannerHysteresisAndGlance(unittest.TestCase):
    def _intent(self, hazard="none"):
        intent = MagicMock()
        intent.hazard = hazard
        intent.frontier = "forward"
        intent.traversability = "high"
        intent.source = "vlm"
        return intent

    def test_zog_brightness_hysteresis(self):
        planner = IntentPlanner(rng_seed=11)
        planner.set_persona("zog7")
        clear = {"visual_obstacle": False, "obstacle_left_ratio": 0.2, "obstacle_right_ratio": 0.2}
        bright = dict(clear, brightness="Bright")

        t = 50.0
        p1 = planner.plan(self._intent(), bright, 1, t); t += 0.4
        self.assertEqual(p1.action, "stop")          # freeze on bright
        p2 = planner.plan(self._intent(), bright, 2, t); t += 0.4
        self.assertEqual(p2.action, "stop")          # cooldown
        p3 = planner.plan(self._intent(), clear, 3, t); t += 0.4
        self.assertEqual(p3.action, "stop")          # only ONE clear frame: still paused
        p4 = planner.plan(self._intent(), clear, 4, t); t += 0.4
        self.assertNotIn("zog7:concealment_pause", p4.reason)  # two clears: resumed

    def test_klaus_wall_hysteresis_deadband(self):
        planner = IntentPlanner(rng_seed=5)
        planner.set_persona("klaus")
        mid = {"visual_obstacle": False, "obstacle_left_ratio": 0.09, "obstacle_right_ratio": 0.0}

        # Acquire wall above 0.12 (steps spaced > latch duration so every
        # cycle actually reaches the klaus policy).
        t = 50.0
        planner.plan(self._intent(), {"visual_obstacle": False, "obstacle_left_ratio": 0.2, "obstacle_right_ratio": 0.0}, 1, t)
        t += 2.0
        self.assertTrue(planner._klaus_wall_acquired)
        # Signal drifts into the deadband (0.06–0.12): no frantic re-seeking.
        p = planner.plan(self._intent(), mid, 2, t)
        t += 2.0
        self.assertIn("klaus:perimeter_sweep", p.reason)
        self.assertTrue(planner._klaus_wall_acquired)
        # Below hard floor: acquisition lost.
        low = {"visual_obstacle": False, "obstacle_left_ratio": 0.03, "obstacle_right_ratio": 0.0}
        p = planner.plan(self._intent(), low, 3, t)
        self.assertFalse(planner._klaus_wall_acquired)

    def test_idle_glance_replaces_hold(self):
        from unittest.mock import patch as _patch

        # Deterministic direct check on the helper.
        planner = IntentPlanner(rng_seed=13)
        planner.set_persona("benson")
        hold_plan = MotionPlan("stop", 0, "slow", "hold", "benson:safety_pause")
        with _patch("planner.IDLE_GLANCE_PROBABILITY", 1.0), \
             _patch("planner.IDLE_GLANCE_MIN_INTERVAL_SEC", 0.0):
            glanced = planner._maybe_idle_glance(hold_plan, 200.0)
        self.assertEqual(glanced.action, "left")   # side alternates; starts right→left
        self.assertEqual(glanced.reason, "idle:glance")
        self.assertEqual(glanced.mode, "turn")

        # Non-hold plans pass through untouched.
        fwd_plan = MotionPlan("forward", 0, "medium", "explore", "x")
        with _patch("planner.IDLE_GLANCE_PROBABILITY", 1.0), \
             _patch("planner.IDLE_GLANCE_MIN_INTERVAL_SEC", 0.0):
            untouched = planner._maybe_idle_glance(fwd_plan, 201.0)
        self.assertIs(untouched, fwd_plan)

        # With probability 1.0 wired into plan(), a benson safety-pause hold
        # becomes a glance turn instead.
        planner2 = IntentPlanner(rng_seed=13)
        planner2.set_persona("benson")
        intent = self._intent()
        clear = {"visual_obstacle": False, "obstacle_left_ratio": 0.0, "obstacle_right_ratio": 0.0}
        t = 100.0
        seen_glance = False
        with _patch("planner.IDLE_GLANCE_PROBABILITY", 1.0), \
             _patch("planner.IDLE_GLANCE_MIN_INTERVAL_SEC", 0.0), \
             _patch("planner.JITTER_FRACTION", 0.0):
            for i in range(30):
                p = planner2.plan(intent, clear, i, t)
                t += 1.0
                if p.reason.startswith("idle:glance"):
                    seen_glance = True
                    break
                if p.action == "stop" and i > 0:
                    # hold without glance shouldn't persist past next cycle at p=1.0
                    pass
        self.assertTrue(seen_glance)


class TestRecoveryTimeBased(unittest.TestCase):
    def test_cooldown_is_time_based(self):
        rec = RecoveryController()
        rec.escalation_level = 0
        rec.low_motion_streak = 99
        telemetry = {"obstacle_left_ratio": 0.0, "obstacle_right_ratio": 0.0}
        trigger = rec.observe(tracking_forward=True, motion_score=1.0, telemetry=telemetry, now_ts=10.0)
        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.level, "nudge")
        self.assertAlmostEqual(rec.cooldown_cycles, 0.5)

        # Immediately again (still in cooldown): no new trigger despite huge streak
        trigger2 = rec.observe(tracking_forward=True, motion_score=1.0, telemetry=telemetry, now_ts=10.1)
        self.assertIsNone(trigger2)

        # After cooldown expires (10.5), next observe escalates to escape.
        # The decay from the slow 1.9 s gap is far smaller than the streak.
        trigger3 = rec.observe(tracking_forward=True, motion_score=1.0, telemetry=telemetry, now_ts=12.0)
        self.assertIsNotNone(trigger3)
        self.assertEqual(trigger3.level, "escape")
        self.assertEqual(rec.escalation_level, 2)

    def test_streak_decays_over_slow_cycles(self):
        rec = RecoveryController()
        rec.low_motion_streak = 4.0
        rec._last_observe_ts = 10.0
        rec.observe(tracking_forward=False, motion_score=-1.0,
                    telemetry={}, now_ts=12.0)  # 2s gap vs 0.4 expected
        # decay of excess (1.6s / 0.4) = 4 plus non-forward decay of 1 → clamped at 0
        self.assertLessEqual(rec.low_motion_streak, 4.0 - 1.0)



class TestDifferentialDrive(unittest.IsolatedAsyncioTestCase):
    def _mock_client(self):
        client = MagicMock(spec=WheelsonClient)
        ok = lambda **kw: {"ok": True, "busy": False, "safety": False}
        client.set_speed = AsyncMock(side_effect=lambda *a, **k: ok())
        client.stop = AsyncMock(side_effect=lambda *a, **k: ok())
        client.move_timed = AsyncMock(side_effect=lambda action, duration_ms, **k: {
            "ok": True, "busy": False, "safety": False,
            "action": action, "duration_ms": duration_ms})
        client.move_indefinitely = AsyncMock(side_effect=lambda direction, **k: {
            "ok": True, "busy": False, "safety": False, "action": direction})
        client.drive = AsyncMock(side_effect=lambda left, right, **k: {
            "ok": True, "busy": False, "safety": False})
        return client

    async def test_drive_client_posts_command(self):
        mock_http = MagicMock()
        mock_http.post = AsyncMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True}
        mock_http.post.return_value = resp

        wc = WheelsonClient(mock_http)
        await wc.drive(-40, 100, command_id="c1", source="test", mode="explore")
        called_json = mock_http.post.call_args[1]["json"]
        self.assertEqual(called_json["command"], "drive")
        self.assertEqual(called_json["left"], -40)
        self.assertEqual(called_json["right"], 100)

    async def test_apply_plan_with_steer_uses_drive(self):
        sup = MotionSupervisor(self._mock_client())
        plan = MotionPlan("forward", 0, "medium", "explore", "test", steer=0.2)
        await sup.apply_plan(plan, source="test")
        sup.client.drive.assert_awaited_once()
        kwargs = sup.client.drive.call_args[1]
        payload_left = sup.client.drive.call_args[0][0]
        payload_right = sup.client.drive.call_args[0][1]
        base = 40  # SPEED_BASE_INTENSITY medium
        self.assertEqual(payload_left, int(base * 0.8))
        self.assertEqual(payload_right, int(base * 1.2))
        # Heartbeat snapshot must carry the drive targets
        gen, action, mode, source, drive = await sup._hb_snapshot()
        self.assertEqual(action, "drive")
        self.assertEqual(drive, (int(base * 0.8), int(base * 1.2)))

    async def test_recovery_releases_lock_between_steps(self):
        """The supervisor lock must be acquirable while recovery sleeps between steps."""
        sup = MotionSupervisor(self._mock_client())
        task = asyncio.create_task(sup.execute_recovery(
            level="hard_escape", turn_direction="left",
            speed_level="slow", source="reactive",
        ))
        await asyncio.sleep(0.05)  # let it enter the sequence (sleeps ~2.6s total)
        try:
            acquired = await asyncio.wait_for(sup._lock.acquire(), timeout=0.2)
            self.assertTrue(acquired, "lock held across recovery sleep — heartbeat would starve")
            sup._lock.release()
        except asyncio.TimeoutError:
            self.fail("lock held across recovery sleep — heartbeat would starve")
        finally:
            task.cancel()
            with contextlib_suppress():
                await task

    async def test_remote_move_sets_continuous_action_and_timestamp(self):
        sup = MotionSupervisor(self._mock_client())
        before = asyncio.get_running_loop().time()
        await sup.remote_move("forward")
        self.assertEqual(sup.active_continuous_action, "forward")
        self.assertGreaterEqual(sup._last_remote_command_ts, before)
        gen, action, _, source, _ = await sup._hb_snapshot()
        self.assertEqual(action, "forward")
        self.assertEqual(source, "remote")

    async def test_heartbeat_skips_stale_remote_renewal(self):
        from config import REMOTE_RENEWAL_WINDOW_SEC
        sup = MotionSupervisor(self._mock_client())
        await sup.remote_move("forward")
        sup.client.move_indefinitely.reset_mock()
        # Simulate stale commands: push timestamp beyond the renewal window.
        sup._last_remote_command_ts -= (REMOTE_RENEWAL_WINDOW_SEC + 1.0)
        await sup._heartbeat_once()
        sup.client.move_indefinitely.assert_not_called()
        gen, action, _, _, drive = await sup._hb_snapshot()
        self.assertEqual(action, "stop")
        self.assertIsNone(drive)

    async def test_heartbeat_renews_fresh_remote_motion(self):
        from config import REMOTE_RENEWAL_WINDOW_SEC
        sup = MotionSupervisor(self._mock_client())
        await sup.remote_move("forward")
        sup.client.move_indefinitely.reset_mock()
        await sup._heartbeat_once()
        sup.client.move_indefinitely.assert_awaited_once()
        args = sup.client.move_indefinitely.call_args
        self.assertEqual(args[0][0], "forward")


import contextlib


def contextlib_suppress():
    return contextlib.suppress(asyncio.CancelledError)


if __name__ == "__main__":
    unittest.main()
