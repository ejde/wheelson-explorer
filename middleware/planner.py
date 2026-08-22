"""
wheelson-explorer · middleware · planner.py
Intent planner (tactical planning, all _plan_* methods).
"""

import random
from typing import Literal

from motion import MotionPlan
from scene import StrategyIntent, _safe_int
from config import (
    PERSONA_POLICIES,
    TURN_DURATION_MS,
    TURN_SIDE_MARGIN,
    FORWARD_BURST_DEFAULT_MS,
    PLAN_LATCH_FORWARD_SEC,
    PLAN_LATCH_TURN_SEC,
    TIMEOUT_SCAN_STEPS,
    TIMEOUT_SCAN_TURN_MS,
    ESCAPE_BACKWARD_MS,
    KLAUS_ARC_INTERVAL,
    KLAUS_ARC_TURN_MS,
    STARTUP_SLOW_CYCLES,
    STEER_GAIN,
    STEER_DEADBAND,
    STEER_MAX,
    JITTER_FRACTION,
    IDLE_GLANCE_PROBABILITY,
    IDLE_GLANCE_MIN_INTERVAL_SEC,
    IDLE_GLANCE_TURN_MS,
)


class IntentPlanner:
    def __init__(self, rng_seed=None):
        self._rng = random.Random(rng_seed)
        self.persona = "benson"
        self.policy = PERSONA_POLICIES["benson"]
        self.last_selected_action = "stop"
        self.same_turn_streak = 0
        self.turn_chain_streak = 0
        self.stop_streak = 0
        self.forward_chain_streak = 0
        self._last_probe_turn: Literal["left", "right"] = "left"
        self.timeout_scan_budget = 0
        self.klaus_wall_seek_streak = 0
        self.david_turn_bias: Literal["left", "right"] = "left"
        self.zog_freeze_cooldown = 0
        self._zog_frozen = False
        # Naturalness state
        self._episode_samples: dict[str, float] = {}
        self._last_glance_ts = -1e9
        self._glance_side: Literal["left", "right"] = "right"
        # Hysteresis latches
        self._zog_bright_streak = 0
        self._zog_clear_streak = 0
        self._klaus_wall_acquired = False
        self._zog_edge_acquired = False
        self._latched_plan: MotionPlan | None = None
        self._latched_until_ts = 0.0
        self._forward_start_ts = 0.0

    def set_persona(self, persona: str) -> None:
        self.persona = persona
        self.policy = PERSONA_POLICIES.get(persona, PERSONA_POLICIES["benson"])
        self.last_selected_action = "stop"
        self.same_turn_streak = 0
        self.turn_chain_streak = 0
        self.stop_streak = 0
        self.forward_chain_streak = 0
        self._last_probe_turn = "left"
        self.timeout_scan_budget = 0
        self.klaus_wall_seek_streak = 0
        self.david_turn_bias = "left"
        self.zog_freeze_cooldown = 0
        self._zog_frozen = False
        self._latched_plan = None
        self._latched_until_ts = 0.0
        self._forward_start_ts = 0.0
        self._episode_samples = {}
        self._last_glance_ts = -1e9
        self._zog_bright_streak = 0
        self._zog_clear_streak = 0
        self._klaus_wall_acquired = False
        self._zog_edge_acquired = False

    def _base_speed(self) -> str:
        return str(self.policy.get("base_speed", "medium"))

    def _choose_probe_turn(self, telemetry: dict) -> Literal["left", "right"]:
        side_left = telemetry.get("obstacle_left_ratio", 0.0)
        side_right = telemetry.get("obstacle_right_ratio", 0.0)
        if side_left + TURN_SIDE_MARGIN < side_right:
            self._last_probe_turn = "left"
            return "left"
        if side_right + TURN_SIDE_MARGIN < side_left:
            self._last_probe_turn = "right"
            return "right"
        self._last_probe_turn = "right" if self._last_probe_turn == "left" else "left"
        return self._last_probe_turn

    def _register_action(self, action: str, now_ts: float) -> None:
        if action in {"left", "right"}:
            if action == self.last_selected_action:
                self.same_turn_streak += 1
            else:
                self.same_turn_streak = 1
            self.turn_chain_streak += 1
        else:
            self.same_turn_streak = 0
            self.turn_chain_streak = 0

        if action == "stop":
            self.stop_streak += 1
        else:
            self.stop_streak = 0

        if action == "forward":
            if self.last_selected_action != "forward":
                self._forward_start_ts = now_ts
            self.forward_chain_streak += 1
        else:
            self.forward_chain_streak = 0
            self._forward_start_ts = 0.0

        self.last_selected_action = action

    def _forward_duration(self, now_ts: float) -> float:
        if self.last_selected_action == "forward" and self._forward_start_ts > 0.0:
            return now_ts - self._forward_start_ts
        return 0.0

    # -- Naturalness helpers -------------------------------------------------
    def _sample(self, key: str, base: float) -> float:
        """Per-forward-episode jittered threshold (de-quantizes persona pacing)."""
        if key not in self._episode_samples:
            frac = self._rng.uniform(-JITTER_FRACTION, JITTER_FRACTION)
            self._episode_samples[key] = base * (1.0 + frac)
        return self._episode_samples[key]

    def _begin_episode_if_new(self) -> None:
        if self.last_selected_action != "forward":
            self._episode_samples = {}

    def _compute_steer(self, telemetry: dict) -> float:
        """Positive steer veers left, away from a higher right-side obstacle ratio."""
        left_r = float(telemetry.get("obstacle_left_ratio", 0.0))
        right_r = float(telemetry.get("obstacle_right_ratio", 0.0))
        raw = STEER_GAIN * (right_r - left_r)
        raw = max(-STEER_MAX, min(STEER_MAX, raw))
        if abs(raw) < STEER_DEADBAND:
            return 0.0
        return raw

    def _maybe_idle_glance(self, plan: MotionPlan, now_ts: float) -> MotionPlan:
        """Replace some long holds with brief orienting glances so stillness
        reads as attentive rather than frozen."""
        if plan.action != "stop" or plan.mode != "hold":
            return plan
        if (now_ts - self._last_glance_ts) < IDLE_GLANCE_MIN_INTERVAL_SEC:
            return plan
        if self._rng.random() >= IDLE_GLANCE_PROBABILITY:
            return plan
        self._last_glance_ts = now_ts
        self._glance_side = "left" if self._glance_side == "right" else "right"
        return MotionPlan(
            action=self._glance_side,
            duration_ms=IDLE_GLANCE_TURN_MS,
            speed_level="slow",
            mode="turn",
            reason="idle:glance",
        )


    def _can_interrupt_latch(self, intent: StrategyIntent, telemetry: dict) -> bool:
        if bool(telemetry.get("strategy_timeout_event", False)):
            return True
        if bool(telemetry.get("visual_obstacle", False)):
            return True
        if intent.hazard == "hard" or intent.frontier == "blocked":
            return True
        return False

    def _latch_plan(self, plan: MotionPlan, now_ts: float) -> None:
        hold_for = 0.0
        if plan.action == "forward" and plan.duration_ms == 0 and plan.mode == "explore":
            base_ms = int(self.policy.get("forward_burst_ms", FORWARD_BURST_DEFAULT_MS))
            persona_hold = max(1.0, min(3.0, base_ms / 500.0))
            hold_for = max(PLAN_LATCH_FORWARD_SEC, persona_hold)
        elif plan.action in {"left", "right"} and plan.duration_ms > 0:
            hold_for = PLAN_LATCH_TURN_SEC

        if hold_for <= 0.0:
            self._latched_plan = None
            self._latched_until_ts = 0.0
            return

        self._latched_plan = MotionPlan(
            action=plan.action,
            duration_ms=plan.duration_ms,
            speed_level=plan.speed_level,
            mode=plan.mode,
            reason=plan.reason,
        )
        self._latched_until_ts = now_ts + hold_for

    def _frontier_turn(self, intent: StrategyIntent, telemetry: dict) -> Literal["left", "right"]:
        if intent.frontier == "left":
            return "left"
        if intent.frontier == "right":
            return "right"
        return self._choose_probe_turn(telemetry)

    def _open_turn(self, telemetry: dict) -> Literal["left", "right"]:
        left_ratio = telemetry.get("obstacle_left_ratio", 0.0)
        right_ratio = telemetry.get("obstacle_right_ratio", 0.0)
        if left_ratio + TURN_SIDE_MARGIN < right_ratio:
            return "left"
        if right_ratio + TURN_SIDE_MARGIN < left_ratio:
            return "right"
        return self._choose_probe_turn(telemetry)

    def _strategy_timeout_plan(self, speed_level: str, telemetry: dict) -> MotionPlan | None:
        if bool(telemetry.get("strategy_timeout_event", False)):
            requested_steps = _safe_int(telemetry.get("timeout_scan_steps", TIMEOUT_SCAN_STEPS), TIMEOUT_SCAN_STEPS)
            self.timeout_scan_budget = max(self.timeout_scan_budget, max(1, requested_steps))

        if self.timeout_scan_budget <= 0:
            return None

        if self.timeout_scan_budget == max(1, _safe_int(telemetry.get("timeout_scan_steps", TIMEOUT_SCAN_STEPS), TIMEOUT_SCAN_STEPS)):
            plan = MotionPlan(
                action="stop",
                duration_ms=0,
                speed_level="slow",
                mode="hold",
                reason="gate:strategist_timeout_pause",
            )
        else:
            plan = MotionPlan(
                action=self._choose_probe_turn(telemetry),
                duration_ms=max(TIMEOUT_SCAN_TURN_MS, TURN_DURATION_MS.get(self.persona, 550)),
                speed_level=speed_level,
                mode="turn",
                reason="gate:strategist_timeout_scan",
            )

        self.timeout_scan_budget = max(0, self.timeout_scan_budget - 1)
        return plan

    def _scene_speed(self, base: str, intent: StrategyIntent, telemetry: dict, cycle_count: int) -> tuple[str, str]:
        rank = {"slow": 0, "medium": 1, "fast": 2}
        cap = "fast"
        reasons: list[str] = []

        if intent.hazard == "hard":
            cap = "slow"
            reasons.append("hazard:hard")
        elif intent.hazard == "soft":
            cap = "medium"
            reasons.append("hazard:soft")

        if intent.traversability == "low":
            cap = "slow"
            reasons.append("traversability:low")
        elif intent.traversability == "medium" and cap == "fast":
            cap = "medium"
            reasons.append("traversability:medium")

        if telemetry.get("brightness") == "Dark":
            cap = "slow"
            reasons.append("dark")

        if cycle_count < STARTUP_SLOW_CYCLES:
            cap = "slow"
            reasons.append("startup")

        degraded_level = _safe_int(telemetry.get("strategy_degraded_level", 0), 0)
        if degraded_level >= 2:
            cap = "slow"
            reasons.append("local_only")
        elif degraded_level >= 1 and cap == "fast":
            cap = "medium"
            reasons.append("degraded")

        if float(telemetry.get("semantic_confidence", 0.5)) < 0.55:
            cap = "slow"
            reasons.append("low_conf")

        effective_rank = min(rank.get(base, 1), rank.get(cap, 1))
        speed = "slow"
        for key, value in rank.items():
            if value == effective_rank:
                speed = key
                break

        return speed, ",".join(reasons) if reasons else "none"

    def _plan_benson(self, intent: StrategyIntent, telemetry: dict, speed_level: str, now_ts: float) -> MotionPlan:
        turn_ms = min(TURN_DURATION_MS.get(self.persona, 550), 650)
        forward_dur = self._forward_duration(now_ts)

        if intent.hazard == "hard" or intent.frontier == "blocked":
            if bool(telemetry.get("visual_obstacle", False)):
                return MotionPlan("stop", 0, "slow", "hold", "benson:hard_hold")
            return MotionPlan("backward", ESCAPE_BACKWARD_MS, "slow", "reverse", "benson:hard_reset")

        if intent.hazard == "soft" or intent.traversability == "low":
            return MotionPlan(self._open_turn(telemetry), turn_ms, "slow", "turn", "benson:risk_turn")

        if forward_dur >= self._sample("pause_limit", 9.6):
            return MotionPlan("stop", 0, "slow", "hold", "benson:safety_pause")

        if bool(telemetry.get("strategy_degraded", False)) and forward_dur >= self._sample("verify_pause", 2.4):
            return MotionPlan("stop", 0, "slow", "hold", "benson:verify_pause")

        if intent.frontier in {"left", "right"}:
            return MotionPlan(intent.frontier, turn_ms, speed_level, "turn", "benson:frontier_turn")

        return MotionPlan("forward", 0, speed_level, "explore", "benson:controlled_advance")

    def _next_david_turn(self, hint: Literal["left", "right"] | None = None) -> Literal["left", "right"]:
        if hint in {"left", "right"}:
            self.david_turn_bias = hint
        else:
            self.david_turn_bias = "right" if self.david_turn_bias == "left" else "left"
        return self.david_turn_bias

    def _plan_sir_david(self, intent: StrategyIntent, telemetry: dict, speed_level: str, now_ts: float) -> MotionPlan:
        turn_ms = max(TURN_DURATION_MS.get(self.persona, 500), 520)
        forward_dur = self._forward_duration(now_ts)

        if intent.hazard == "hard":
            return MotionPlan(self._open_turn(telemetry), turn_ms, "slow", "turn", "sirdavid:careful_reroute")

        if bool(telemetry.get("strategy_degraded", False)) and forward_dur >= 1.2:
            return MotionPlan(self._next_david_turn(), turn_ms, "slow", "turn", "sirdavid:listen_scan")

        if intent.novelty == "high" and intent.hazard == "none":
            if forward_dur >= self._sample("reframe", 2.4):
                return MotionPlan(self._next_david_turn(), turn_ms, speed_level, "turn", "sirdavid:reframe_subject")
            return MotionPlan("forward", 0, speed_level, "explore", "sirdavid:approach_novelty")

        if forward_dur >= self._sample("wander_arc", 3.6):
            hint = intent.frontier if intent.frontier in {"left", "right"} else None
            return MotionPlan(self._next_david_turn(hint), turn_ms, speed_level, "turn", "sirdavid:wander_arc")

        if intent.frontier in {"left", "right"} and intent.hazard != "soft":
            return MotionPlan(intent.frontier, turn_ms, speed_level, "turn", "sirdavid:follow_frontier")

        return MotionPlan("forward", 0, speed_level, "explore", "sirdavid:gentle_explore")

    def _plan_klaus(self, intent: StrategyIntent, telemetry: dict, speed_level: str, cycle_count: int, now_ts: float) -> MotionPlan:
        preferred = str(self.policy.get("preferred_wall", "left"))
        if preferred not in {"left", "right"}:
            preferred = "left"
        opposite: Literal["left", "right"] = "right" if preferred == "left" else "left"
        preferred_turn: Literal["left", "right"] = "left" if preferred == "left" else "right"
        wall_signal = telemetry.get("obstacle_left_ratio", 0.0) if preferred == "left" else telemetry.get("obstacle_right_ratio", 0.0)
        wide_turn_ms = max(TURN_DURATION_MS.get(self.persona, 700), KLAUS_ARC_TURN_MS)
        forward_dur = self._forward_duration(now_ts)

        if intent.hazard == "hard" or intent.frontier == "blocked":
            if bool(telemetry.get("visual_obstacle", False)):
                return MotionPlan("backward", ESCAPE_BACKWARD_MS, "slow", "reverse", "klaus:clearance_reset")
            return MotionPlan(opposite, wide_turn_ms, "slow", "turn", "klaus:detour_arc")

        # Wall-following hysteresis: once acquired, keep following until the
        # wall signal drops well below the acquisition threshold.
        if not self._klaus_wall_acquired and wall_signal < 0.08:
            self.klaus_wall_seek_streak += 1
            if self.klaus_wall_seek_streak % 2 == 0:
                return MotionPlan("forward", 0, speed_level, "explore", "klaus:probe_perimeter")
            return MotionPlan(preferred_turn, wide_turn_ms, speed_level, "turn", "klaus:acquire_perimeter")
        if wall_signal >= 0.12:
            self._klaus_wall_acquired = True
            self.klaus_wall_seek_streak = 0
        elif wall_signal < 0.06:
            self._klaus_wall_acquired = False
            self.klaus_wall_seek_streak = 0

        if wall_signal > 0.72:
            return MotionPlan(opposite, 620, speed_level, "turn", "klaus:clearance_correction")

        if forward_dur >= self._sample("arc_interval", KLAUS_ARC_INTERVAL * 1.2):
            arc_turn: Literal["left", "right"] = preferred_turn
            if cycle_count % 5 == 0:
                arc_turn = opposite
            return MotionPlan(arc_turn, wide_turn_ms, speed_level, "turn", "klaus:signature_arc")

        if bool(telemetry.get("strategy_degraded", False)) and forward_dur >= 1.2:
            return MotionPlan(preferred_turn, wide_turn_ms, "slow", "turn", "klaus:scan_perimeter")

        return MotionPlan("forward", 0, speed_level, "explore", "klaus:perimeter_sweep")

    def _plan_zog7(self, intent: StrategyIntent, telemetry: dict, speed_level: str, now_ts: float) -> MotionPlan:
        preferred = str(self.policy.get("preferred_wall", "right"))
        if preferred not in {"left", "right"}:
            preferred = "right"
        preferred_turn: Literal["left", "right"] = "right" if preferred == "right" else "left"
        opposite: Literal["left", "right"] = "left" if preferred == "right" else "right"
        wall_signal = telemetry.get("obstacle_right_ratio", 0.0) if preferred == "right" else telemetry.get("obstacle_left_ratio", 0.0)
        light = str(telemetry.get("brightness", "Normal"))
        forward_dur = self._forward_duration(now_ts)

        # Brightness freeze with hysteresis: enter on any bright frame, only
        # resume after two consecutive non-bright frames.
        if light == "Bright" and intent.hazard == "none":
            self._zog_bright_streak += 1
            self._zog_clear_streak = 0
            self._zog_frozen = True
            return MotionPlan("stop", 0, "slow", "hold", "zog7:freeze_in_light")

        self._zog_clear_streak += 1
        self._zog_bright_streak = 0

        if self._zog_frozen:
            if self._zog_clear_streak < 2:
                return MotionPlan("stop", 0, "slow", "hold", "zog7:concealment_pause")
            self._zog_frozen = False

        if intent.hazard == "hard" or intent.frontier == "blocked":
            return MotionPlan("backward", ESCAPE_BACKWARD_MS, "slow", "reverse", "zog7:tactical_reset")

        # Edge-following hysteresis.
        if not self._zog_edge_acquired and wall_signal < 0.06:
            self._zog_edge_acquired = False
            return MotionPlan(preferred_turn, 620, speed_level, "turn", "zog7:seek_edge")
        if wall_signal >= 0.10:
            self._zog_edge_acquired = True
        elif wall_signal < 0.04:
            self._zog_edge_acquired = False

        if wall_signal > 0.70:
            return MotionPlan(opposite, 520, speed_level, "turn", "zog7:standoff_adjust")

        if forward_dur >= self._sample("listen_pause", 3.6):
            return MotionPlan("stop", 0, "slow", "hold", "zog7:listen_pause")

        return MotionPlan("forward", 0, speed_level, "explore", "zog7:shadow_run")

    def plan(self, intent: StrategyIntent, telemetry: dict, cycle_count: int, now_ts: float) -> MotionPlan:
        self._begin_episode_if_new()

        if (
            self._latched_plan
            and now_ts < self._latched_until_ts
            and not self._can_interrupt_latch(intent, telemetry)
        ):
            self._register_action(self._latched_plan.action, now_ts)
            return MotionPlan(
                action=self._latched_plan.action,
                duration_ms=self._latched_plan.duration_ms,
                speed_level=self._latched_plan.speed_level,
                mode=self._latched_plan.mode,
                steer=self._latched_plan.steer,
                reason=f"{self._latched_plan.reason}|latched",
            )

        base_speed = self._base_speed()
        speed_level, speed_reason = self._scene_speed(base_speed, intent, telemetry, cycle_count)
        timeout_plan = self._strategy_timeout_plan(speed_level, telemetry)
        if timeout_plan:
            self._latched_plan = None
            self._latched_until_ts = 0.0
            self._register_action(timeout_plan.action, now_ts)
            timeout_plan.reason = f"{timeout_plan.reason}|{speed_reason}"
            return timeout_plan

        if self.persona == "benson":
            plan = self._plan_benson(intent, telemetry, speed_level, now_ts)
        elif self.persona == "sir_david":
            plan = self._plan_sir_david(intent, telemetry, speed_level, now_ts)
        elif self.persona == "klaus":
            plan = self._plan_klaus(intent, telemetry, speed_level, cycle_count, now_ts)
        elif self.persona == "zog7":
            plan = self._plan_zog7(intent, telemetry, speed_level, now_ts)
        else:
            plan = MotionPlan("forward", 0, speed_level, "explore", "default:advance")

        # Continuous proportional steering: shape blind forward bursts into
        # gentle arcs using the per-cycle side obstacle ratios.
        if plan.action == "forward" and plan.mode == "explore":
            plan.steer = self._compute_steer(telemetry)
            if abs(plan.steer) >= 0.01:
                plan.reason = f"{plan.reason}|steer{plan.steer:+.2f}"

        plan = self._maybe_idle_glance(plan, now_ts)
        self._register_action(plan.action, now_ts)

        self._latch_plan(plan, now_ts)
        plan.reason = f"{plan.reason}|{speed_reason}"
        return plan
