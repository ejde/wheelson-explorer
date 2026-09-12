"""Transition and failure regressions, using no robot and no model API."""
import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image
from executive import SkillExecutive
from motion import MotionPlan, MotionSupervisor, WheelsonClient
from perception import CameraStream
from planner import IntentPlanner
from skills import observation_is_current, validate_strategy


def client_mock():
    client = MagicMock(spec=WheelsonClient)
    for method in ("set_speed", "set_light", "stop", "drive", "move_timed", "renew"):
        setattr(client, method, AsyncMock(return_value={"ok": True}))
    client.claim = AsyncMock()
    return client


def intent(**kw):
    return SimpleNamespace(**dict(dict(hazard="none", frontier="forward", traversability="high",
        novelty="low", source="vlm"), **kw))


def strategy(skill="explore", direction="forward", duration=2.0):
    return {"objective": "Explore the open floor", "steps": [
        {"skill": skill, "direction": direction, "duration_s": duration}]}


class MotionTransitions(unittest.IsolatedAsyncioTestCase):
    async def test_straight_to_arc_preserves_average_intensity(self):
        sup = MotionSupervisor(client_mock())
        await sup.apply_plan(MotionPlan("forward", 0, "medium", "explore", "straight"), source="planner")
        straight = sup.client.drive.call_args.args
        await sup.apply_plan(MotionPlan("forward", 0, "medium", "explore", "arc", steer=.3), source="planner")
        arc = sup.client.drive.call_args.args
        self.assertEqual(straight, (20, 20))
        self.assertEqual(sum(straight), sum(arc))

    async def test_continuous_to_timed_renewal_keeps_new_identity(self):
        sup = MotionSupervisor(client_mock())
        await sup.apply_plan(MotionPlan("forward", 0, "medium", "explore", "forward"), source="planner")
        previous = sup._motion_command_id
        await sup.apply_plan(MotionPlan("left", 600, "medium", "turn", "turn", maneuver_id="turn-1"), source="planner")
        await sup._heartbeat_once()
        renewed = sup.client.renew.call_args.args[0]
        self.assertNotEqual(renewed, previous)
        self.assertEqual(renewed, sup._motion_command_id)
        sup.client.move_indefinitely.assert_not_called()

    async def test_quantization_preserves_cruise_speed_at_every_level(self):
        for speed, base in (("slow", 12), ("medium", 20), ("fast", 35)):
            sup = MotionSupervisor(client_mock())
            for steer in (0., .06, .15, .3, .6, -.3):
                await sup.apply_plan(MotionPlan("forward", 0, speed, "explore", "arc", steer=steer), source="planner")
                self.assertEqual(sum(sup._hb_drive), 2 * base)

    async def test_same_target_keeps_identity_and_uses_renewal_only(self):
        sup = MotionSupervisor(client_mock())
        plan = MotionPlan("forward", 0, "medium", "explore", "same")
        first = await sup.apply_plan(plan, source="planner")
        second = await sup.apply_plan(plan, source="planner")
        self.assertEqual(first.command_id, second.command_id)
        sup.client.drive.assert_awaited_once()
        await sup._heartbeat_once()
        sup.client.renew.assert_awaited_once()

    async def test_same_timed_maneuver_is_not_restarted_even_after_completion(self):
        sup = MotionSupervisor(client_mock())
        plan = MotionPlan("left", 600, "slow", "turn", "turn", maneuver_id="one")
        with patch.object(sup, "_now", return_value=10.0):
            await sup.apply_plan(plan, source="planner")
        with patch.object(sup, "_now", return_value=10.3):
            await sup.apply_plan(plan, source="planner")
        with patch.object(sup, "_now", return_value=10.7):
            result = await sup.apply_plan(plan, source="planner")
        self.assertEqual(result.action, "stop")
        sup.client.move_timed.assert_awaited_once()
        plan.maneuver_id = "two"
        await sup.apply_plan(plan, source="planner")
        self.assertEqual(sup.client.move_timed.await_count, 2)

    async def test_renewal_and_new_target_are_serialized_without_rollback(self):
        sup = MotionSupervisor(client_mock())
        await sup.apply_plan(MotionPlan("forward", 0, "medium", "explore", "one"), source="planner")
        entered, release = asyncio.Event(), asyncio.Event()
        async def renew(*args, **kw):
            entered.set()
            await release.wait()
            return {"ok": True}
        sup.client.renew.side_effect = renew
        heartbeat = asyncio.create_task(sup._heartbeat_once())
        await entered.wait()
        newer = asyncio.create_task(sup.apply_plan(
            MotionPlan("forward", 0, "medium", "explore", "two", steer=.3), source="planner"))
        await asyncio.sleep(0)
        self.assertFalse(newer.done())
        release.set()
        await asyncio.gather(heartbeat, newer)
        self.assertEqual(sup._hb_drive, (14, 26))
        sup.client.stop.assert_not_called()

    async def test_remote_ownership_blocks_stale_autonomy_and_recovery(self):
        sup = MotionSupervisor(client_mock())
        await sup.execute_recovery(level="escape", turn_direction="left", speed_level="slow", source="reactive")
        await sup.remote_move("right")
        count = sup.client.drive.await_count
        await sup.apply_plan(MotionPlan("forward", 0, "medium", "explore", "old"), source="planner")
        await sup.advance_recovery()
        self.assertFalse(sup.recovery_active)
        self.assertEqual(sup.client.drive.await_count, count)
        self.assertEqual(sup.active_source, "remote")

    async def test_autonomy_freshness_expiry_stops_instead_of_renewing(self):
        sup = MotionSupervisor(client_mock())
        await sup.apply_plan(MotionPlan("forward", 0, "medium", "explore", "old"), source="planner")
        sup._last_autonomy_command_ts -= 2.0
        await sup._heartbeat_once()
        sup.client.renew.assert_not_called()
        sup.client.stop.assert_awaited_once()
        self.assertEqual(sup.active_continuous_action, "stop")

    async def test_rejected_renewal_cannot_restart_motion(self):
        sup = MotionSupervisor(client_mock())
        await sup.remote_move("forward")
        sup.client.renew.return_value = {"ok": False, "busy": True}
        await sup._heartbeat_once()
        await sup._heartbeat_once()
        sup.client.renew.assert_awaited_once()
        sup.client.stop.assert_awaited_once()

    async def test_release_check_cannot_stop_a_new_remote_input(self):
        sup = MotionSupervisor(client_mock())
        await sup.remote_move("forward")
        await sup.release_remote(only_if_inactive=True)
        sup.client.stop.assert_not_called()
        self.assertTrue(sup.remote_owned)

    async def test_recovery_advances_once_per_deadline(self):
        sup = MotionSupervisor(client_mock())
        with patch.object(sup, "_now", return_value=10):
            await sup.execute_recovery(level="escape", turn_direction="left", speed_level="slow", source="reactive")
            deadline = sup._recovery_until
            await sup.advance_recovery()
        self.assertEqual(sup.client.move_timed.await_count, 1)
        with patch.object(sup, "_now", return_value=deadline + .01):
            await sup.advance_recovery()
        self.assertEqual(sup.client.move_timed.await_count, 2)
        self.assertEqual(sup.client.move_timed.call_args.args[0], "left")

    async def test_invalid_acknowledgement_is_not_assumed_success(self):
        http = MagicMock()
        response = MagicMock(status_code=200)
        response.json.return_value = []
        http.post = AsyncMock(return_value=response)
        with self.assertRaises(RuntimeError):
            await WheelsonClient(http).drive(20, 20, command_id="x", source="test", mode="explore")

    async def test_invalid_telemetry_cannot_become_clear_floor(self):
        http = MagicMock()
        response = MagicMock(headers={"x-obstacle-left-ratio":"nan"})
        http.get = AsyncMock(return_value=response)
        with self.assertRaises(RuntimeError):
            await WheelsonClient(http).fetch_frame()


class PlannerTransitions(unittest.TestCase):
    def test_steering_stays_live_during_latch_and_speed_cap_updates(self):
        planner = IntentPlanner(rng_seed=3)
        planner.set_persona("sir_david")
        telemetry = {"obstacle_left_ratio": 0., "obstacle_right_ratio": .3, "semantic_confidence": .9}
        plans = [planner.plan(intent(), telemetry, 100, 10 + n*.1) for n in range(4)]
        self.assertTrue(all(p.steer > 0 for p in plans))
        self.assertEqual(len({p.maneuver_id for p in plans}), 1)
        reversed_scene = dict(telemetry, obstacle_left_ratio=.3, obstacle_right_ratio=0., brightness="Dark")
        plan = planner.plan(intent(), reversed_scene, 100, 10.4)
        self.assertLess(plan.steer, plans[-1].steer)
        self.assertEqual(plan.speed_level, "slow")

    def test_inference_timeout_does_not_force_stop_or_scan(self):
        planner = IntentPlanner(rng_seed=3)
        telemetry = {"semantic_confidence": .9}
        first = planner.plan(intent(), telemetry, 100, 10)
        second = planner.plan(intent(), dict(telemetry, strategy_timeout_event=True), 101, 10.1)
        self.assertEqual(second.action, "forward")
        self.assertEqual(first.maneuver_id, second.maneuver_id)


class SkillContract(unittest.TestCase):
    def test_rejects_unbounded_or_unknown_instructions(self):
        for candidate in (strategy("drive"), strategy(duration=100), strategy(duration=float("nan")),
                          strategy("scan", duration=2), strategy(direction="behind")):
            self.assertIsNone(validate_strategy(candidate))
        self.assertIsNotNone(validate_strategy(strategy()))

    def test_capture_age_and_generation_gate(self):
        self.assertTrue(observation_is_current({"captured_at":10., "generation":2}, 2, 12, 18))
        self.assertFalse(observation_is_current({"captured_at":10., "generation":1}, 2, 12, 18))
        self.assertFalse(observation_is_current({"captured_at":10., "generation":2}, 2, 28, 18))
        self.assertFalse(observation_is_current({"captured_at":30., "generation":2}, 2, 28, 18))

    def test_revisions_wait_for_boundary_and_completed_plan_is_not_replayed(self):
        ex = SkillExecutive()
        ex.submit(1, strategy(), 30, 10)
        first = ex.tick(intent(), {}, 10, "slow")
        ex.submit(2, strategy(direction="left"), 30, 11)
        self.assertEqual(first.maneuver_id, ex.tick(intent(), {}, 11, "slow").maneuver_id)
        next_plan = ex.tick(intent(), {}, 12, "slow")
        self.assertGreater(next_plan.steer, 0)
        self.assertIsNone(ex.tick(intent(), {}, 14, "slow"))
        ex.submit(2, strategy(direction="left"), 30, 15)
        self.assertIsNone(ex.tick(intent(), {}, 15, "slow"))
        self.assertEqual(ex.history[-1]["outcome"], "budget_complete")

    def test_hazard_boundary_loss_and_cancellation(self):
        for signal, telemetry, expected in ((intent(hazard="hard"), {}, "blocked"),
                (intent(), {"obstacle_left_ratio":0}, "boundary_lost")):
            ex = SkillExecutive()
            ex.submit(1, strategy("follow_boundary", "left"), 30, 10)
            self.assertIsNone(ex.tick(signal, telemetry, 10, "slow"))
            self.assertEqual(ex.status, expected)
        ex = SkillExecutive()
        ex.submit(1, strategy(), 30, 10)
        ex.cancel("human_takeover")
        self.assertIsNone(ex.tick(intent(), {}, 11, "slow"))


class PerceptionAndLoop(unittest.IsolatedAsyncioTestCase):
    async def test_expired_frame_is_never_relabelled_as_fresh(self):
        stream = CameraStream(None)
        stream.latest = (b"jpeg", {"captured_at":10., "frame_id":1})
        self.assertEqual(stream.snapshot(10.1)[1]["captured_at"], 10)
        with self.assertRaises(RuntimeError):
            stream.snapshot(11.)

    async def test_loop_runs_skills_and_closes_without_robot_or_llm(self):
        import loop as explorer
        from state import state, AppState
        saved = dict(state.__dict__)
        state.__dict__.update(AppState().__dict__)
        jpeg = io.BytesIO()
        Image.new("RGB", (32, 24), "gray").save(jpeg, format="JPEG")
        client = client_mock()
        client.fetch_frame = AsyncMock(return_value=(jpeg.getvalue(), {
            "distance_cm":999., "visual_obstacle":False, "brightness":"Normal",
            "obstacle_left_ratio":0., "obstacle_right_ratio":0., "nav_state":"HOLD"}))
        events = []
        def broadcast(event):
            events.append(event)
            if len(events) >= 5:
                state.is_running = False
        response = {"observation":"Clear floor", "commentary":"Exploring", "headlight":False,
                    "frontier":"forward", "traversability":"high", "novelty":"low", "hazard":"none",
                    "strategy": strategy()}
        try:
            with patch.object(explorer, "WheelsonClient", return_value=client), \
                 patch.object(explorer, "ask_vlm", AsyncMock(return_value=response)), \
                 patch.object(explorer, "broadcast", side_effect=broadcast), \
                 patch.object(explorer, "LOOP_INTERVAL", 0.05), \
                 patch.object(explorer.signal, "signal"):
                await asyncio.wait_for(explorer.explorer_loop(), timeout=3)
            client.claim.assert_awaited_once()
            self.assertTrue(any(e.get("executive", {}).get("status") == "running" for e in events))
            client.drive.assert_awaited()
            client.stop.assert_awaited()
        finally:
            state.__dict__.clear()
            state.__dict__.update(saved)
