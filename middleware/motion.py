"""
wheelson-explorer · middleware · motion.py
Motion supervisor + wheelson client (single writer to /move).
"""

import asyncio
import contextlib
import logging
import math
import uuid
from dataclasses import dataclass

import httpx

from state import state
from scene import _as_bool, _safe_int
import config
from config import (
    WHEELSON_IP,
    LEASE_HEARTBEAT_SEC,
    REMOTE_RENEWAL_WINDOW_SEC,
    SIDE_STEER_TURN_MS,
    ESCAPE_BACKWARD_MS,
    ESCAPE_TURN_MS,
    HARD_ESCAPE_BACKUP_MS,
    HARD_ESCAPE_TURN_MS,
)

log = logging.getLogger("wheelson")

# Base wheel intensity per speed level (mirrors firmware SPEED_* constants).
SPEED_BASE_INTENSITY = {"slow": 25, "medium": 40, "fast": 70}
FORWARD_SPEED_SCALE = 0.5


def _clamp_int8(value: float) -> int:
    return max(-127, min(127, int(value)))


@dataclass
class MotionResult:
    action: str
    duration_ms: int
    safety: bool
    busy: bool
    command_id: str
    source: str
    mode: str


@dataclass
class MotionPlan:
    action: str
    duration_ms: int
    speed_level: str
    mode: str
    reason: str
    steer: float = 0.0  # -1..1, >0 veers left; used with continuous forward plans
    maneuver_id: str = ""


class WheelsonClient:
    def __init__(self, http_client: httpx.AsyncClient):
        self.http = http_client
        self.session_id = uuid.uuid4().hex
        self.sequence = 0

    async def claim(self) -> None:
        payload = await self._post_move({"command": "claim"})
        if payload.get("protocol") != 2:
            raise RuntimeError("Flash protocol 2 firmware before starting this middleware")

    async def renew(self, target_id: str, *, command_id: str, source: str, mode: str) -> dict:
        return await self._post_move({"command": "renew", "target_id": target_id,
                                     "command_id": command_id, "source": source, "mode": mode})

    async def fetch_frame(self) -> tuple[bytes, dict]:
        resp = await self.http.get(f"http://{WHEELSON_IP}/camera", timeout=10.0)
        resp.raise_for_status()

        telemetry = {
            "distance_cm": 999.0,
            "visual_obstacle": False,
            "brightness": "Normal",
            "dominant_color": "#000000",
            "obstacle_left_ratio": 0.0,
            "obstacle_right_ratio": 0.0,
            "nav_state": "UNKNOWN",
            "active_command_id": "",
            "active_command_source": "",
            "active_command_mode": "",
            "safety_latched": False,
            "safety_reason": "",
        }

        try:
            # Ultrasonic sensor is physically broken; always treat distance as unavailable.
            telemetry["distance_cm"] = 999.0
            telemetry["visual_obstacle"] = (
                resp.headers.get("x-visual-obstacle", "false").lower() == "true"
            )
            telemetry["brightness"] = resp.headers.get("x-brightness", "Normal")
            telemetry["dominant_color"] = resp.headers.get("x-dominant-color", "#000000")
            telemetry["obstacle_left_ratio"] = float(resp.headers.get("x-obstacle-left-ratio", "0"))
            telemetry["obstacle_right_ratio"] = float(resp.headers.get("x-obstacle-right-ratio", "0"))
            telemetry["nav_state"] = resp.headers.get("x-nav-state", "UNKNOWN").upper()
            telemetry["active_command_id"] = resp.headers.get("x-active-command-id", "")
            telemetry["active_command_source"] = resp.headers.get("x-active-command-source", "")
            telemetry["active_command_mode"] = resp.headers.get("x-active-command-mode", "")
            telemetry["safety_latched"] = resp.headers.get("x-safety-latched", "false").lower() == "true"
            telemetry["safety_reason"] = resp.headers.get("x-safety-reason", "")
            telemetry["analysis_age_s"] = float(resp.headers.get("x-analysis-age-ms", "999999")) / 1000.0
            telemetry["wheel_output"] = [_safe_int(resp.headers.get(f"x-wheel-output-{side}", 0))
                                         for side in ("left", "right")]
            telemetry["wheel_target"] = [_safe_int(resp.headers.get(f"x-wheel-target-{side}", 0))
                                         for side in ("left", "right")]
            for key in ("obstacle_left_ratio", "obstacle_right_ratio"):
                if not math.isfinite(telemetry[key]) or not 0 <= telemetry[key] <= 1:
                    raise ValueError(f"Invalid {key}")
            if not math.isfinite(telemetry["analysis_age_s"]) or telemetry["analysis_age_s"] < 0:
                raise ValueError("Invalid obstacle analysis timestamp")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Invalid robot telemetry") from exc

        return resp.content, telemetry

    async def _post_move(self, payload: dict, timeout: float = 10.0) -> dict:
        payload = dict(payload)
        self.sequence += 1
        payload.update(session_id=self.session_id, sequence=self.sequence)
        payload["visual_safety"] = "true" if config.FIRMWARE_VISUAL_SAFETY else "false"
        deadline = min(timeout, config.MOTION_REQUEST_TIMEOUT_SEC)
        async with asyncio.timeout(deadline):
            resp = await self.http.post(f"http://{WHEELSON_IP}/move", json=payload, timeout=deadline)
        if resp.status_code == 409:
            return {
                "ok": False,
                "busy": True,
                "safety": False,
                "action": "stop",
                "duration_ms": 0,
                "command_id": payload.get("command_id", ""),
                "source": payload.get("source", ""),
                "mode": payload.get("mode", ""),
            }
        resp.raise_for_status()
        try:
            result = resp.json()
        except ValueError as exc:
            raise RuntimeError("Invalid motor acknowledgement") from exc
        if not isinstance(result, dict) or "ok" not in result:
            raise RuntimeError("Missing motor acknowledgement")
        return result

    async def set_speed(self, level: str, *, command_id: str, source: str, mode: str) -> dict:
        return await self._post_move(
            {
                "command": "set_speed",
                "level": level,
                "command_id": command_id,
                "source": source,
                "mode": mode,
            },
            timeout=5.0,
        )

    async def set_light(self, level: int, *, command_id: str, source: str, mode: str) -> dict:
        return await self._post_move(
            {
                "command": "set_light",
                "level": str(level),
                "command_id": command_id,
                "source": source,
                "mode": mode,
            },
            timeout=5.0,
        )

    async def move_indefinitely(
        self,
        direction: str,
        *,
        command_id: str,
        source: str,
        mode: str,
    ) -> dict:
        return await self._post_move(
            {
                "command": "move_indefinitely",
                "direction": direction,
                "command_id": command_id,
                "source": source,
                "mode": mode,
            },
            timeout=10.0,
        )

    async def drive(
        self,
        left: int,
        right: int,
        *,
        command_id: str,
        source: str,
        mode: str,
    ) -> dict:
        """Differential drive: signed wheel intensities, lease-renewed like move_indefinitely."""
        return await self._post_move(
            {
                "command": "drive",
                "left": _clamp_int8(left),
                "right": _clamp_int8(right),
                "command_id": command_id,
                "source": source,
                "mode": mode,
            },
            timeout=10.0,
        )

    async def move_timed(
        self,
        action: str,
        duration_ms: int,
        *,
        command_id: str,
        source: str,
        mode: str,
    ) -> dict:
        return await self._post_move(
            {
                "action": action,
                "duration_ms": duration_ms,
                "command_id": command_id,
                "source": source,
                "mode": mode,
            },
            timeout=10.0,
        )

    async def stop(self, *, command_id: str, source: str, mode: str) -> dict:
        return await self._post_move(
            {
                "command": "stop",
                "command_id": command_id,
                "source": source,
                "mode": mode,
            },
            timeout=5.0,
        )


class MotionSupervisor:
    """One ordered actuator writer, including renewals and ownership changes.

    Network calls have bounded deadlines. Firmware rejects out-of-order requests
    and renewals for superseded targets, even if an HTTP response was lost.
    """
    def __init__(self, client: WheelsonClient):
        self.client = client
        self._lock = asyncio.Lock()
        self._current_speed = None
        self._active_continuous_action = "stop"
        self._active_mode = "idle"
        self._active_source = "bootstrap"
        self._last_command_id = "boot"
        self._motion_command_id = ""
        self._seq = 0
        self._session = uuid.uuid4().hex[:12]
        self._gen = 0
        self._hb_drive = None
        self._running = False
        self._heartbeat_task = None
        self._heartbeat_safety_event = False
        self._last_remote_command_ts = -float("inf")
        self._last_autonomy_command_ts = -float("inf")
        self._timed_key = None
        self._timed_until = 0.0
        self._last_result = MotionResult("stop", 0, False, False, "boot", "bootstrap", "idle")
        self._recovery_steps = []
        self._recovery_index = 0
        self._recovery_until = 0.0
        self._recovery_speed = "slow"
        self._recovery_level = ""
        self._recovery_id = 0

    @property
    def active_continuous_action(self):
        return self._active_continuous_action

    @property
    def active_mode(self):
        return self._active_mode

    @property
    def active_source(self):
        return self._active_source

    @property
    def last_command_id(self):
        return self._last_command_id

    @property
    def last_result(self):
        return self._last_result

    @property
    def tracking_forward(self):
        return self._hb_drive is not None and all(v > 0 for v in self._hb_drive)

    @property
    def remote_owned(self):
        return self._now() - self._last_remote_command_ts < config.CONTROL_INACTIVITY_SEC

    @property
    def recovery_active(self):
        return bool(self._recovery_steps)

    @staticmethod
    def _now():
        return asyncio.get_running_loop().time()

    def consume_heartbeat_safety(self):
        fired = self._heartbeat_safety_event
        self._heartbeat_safety_event = False
        return fired

    def _next_command_id(self, source):
        self._seq += 1
        return f"{self._session}-{self._seq}"

    async def _hb_snapshot(self):
        async with self._lock:
            return (self._gen, self._active_continuous_action, self._active_mode,
                    self._active_source, self._hb_drive)

    async def start(self):
        await self.client.claim()
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def close(self):
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        with contextlib.suppress(Exception):
            await self.stop_now(source="shutdown", mode="hold")

    def _clear_motion(self):
        self._gen += 1
        self._active_continuous_action = "stop"
        self._active_mode = "idle"
        self._hb_drive = None
        self._motion_command_id = ""
        self._timed_key = None
        self._timed_until = 0.0

    def _accept(self, payload, action, duration, command_id, source, mode):
        result = MotionResult(action, duration, _as_bool(payload.get("safety", False)),
                              _as_bool(payload.get("busy", False)), command_id, source, mode)
        if payload.get("ok") is False or result.busy or result.safety:
            result.action = "stop"
            self._clear_motion()
            self._recovery_steps = []
        self._last_command_id = command_id
        self._last_result = result
        return result

    async def _ensure_speed_locked(self, level, *, source, mode):
        if self._current_speed != level:
            payload = await self.client.set_speed(level, command_id=self._next_command_id(source),
                                                  source=source, mode=mode)
            if payload.get("ok") is False or payload.get("busy"):
                raise RuntimeError("Firmware rejected speed command")
            self._current_speed = level

    async def _post_stop_locked(self, *, source, mode):
        self._clear_motion()  # Invalidate renewal even if the stop request fails.
        self._recovery_steps = []
        self._active_source = source
        command_id = self._next_command_id(source)
        payload = await self.client.stop(command_id=command_id, source=source, mode=mode)
        return self._accept(payload, "stop", 0, command_id, source, mode)

    async def _apply_locked(self, plan, source):
        now = self._now()
        if plan.action == "stop":
            return await self._post_stop_locked(source=source, mode=plan.mode)
        key = plan.maneuver_id or (source, plan.action, plan.duration_ms, plan.reason.split("|")[0])
        if plan.duration_ms > 0 and key == self._timed_key:
            # Observe the same maneuver without restarting its firmware deadline.
            if now >= self._timed_until:
                return MotionResult("stop", 0, False, False, self._last_command_id, source, plan.mode)
            return self._last_result
        if plan.duration_ms > 0:
            self._clear_motion()
            command_id = self._next_command_id(source)
            await self._ensure_speed_locked(plan.speed_level, source=source, mode=plan.mode)
            payload = await self.client.move_timed(plan.action, plan.duration_ms,
                                                   command_id=command_id, source=source, mode=plan.mode)
            result = self._accept(payload, plan.action, plan.duration_ms, command_id, source, plan.mode)
            if result.action != "stop":
                self._timed_key = key
                self._timed_until = self._now() + plan.duration_ms / 1000.0
        else:
            base = SPEED_BASE_INTENSITY.get(plan.speed_level, 40)
            if plan.action == "forward":
                base = int(base * FORWARD_SPEED_SCALE)
                left = round(base * (1.0 - plan.steer))
                right = 2 * base - left  # Preserve mean speed through integer quantization.
            elif plan.action == "backward":
                left = right = -base
            elif plan.action == "left":
                left, right = -base, base
            elif plan.action == "right":
                left, right = base, -base
            else:
                raise ValueError(f"Unknown action: {plan.action}")
            left, right = _clamp_int8(left), _clamp_int8(right)
            if (self._hb_drive == (left, right) and self._active_source == source
                    and self._active_mode == plan.mode and self._motion_command_id):
                return self._last_result
            self._clear_motion()
            command_id = self._next_command_id(source)
            payload = await self.client.drive(left, right, command_id=command_id, source=source, mode=plan.mode)
            result = self._accept(payload, plan.action, 0, command_id, source, plan.mode)
            if result.action != "stop":
                self._active_continuous_action = "drive"
                self._hb_drive = (left, right)
        if result.action != "stop":
            self._motion_command_id = command_id
            self._active_mode = plan.mode
            self._active_source = source
        return result

    async def apply_plan(self, plan, *, source):
        async with self._lock:
            if self.remote_owned:
                return self._last_result
            self._last_autonomy_command_ts = self._now()
            return await self._apply_locked(plan, source)

    async def execute_recovery(self, *, level, turn_direction, speed_level, source):
        """Start a sequence and return immediately; advance_recovery ticks it."""
        async with self._lock:
            if self.remote_owned:
                return self._last_result
            self._recovery_id += 1
            if level == "nudge":
                steps = [(turn_direction, SIDE_STEER_TURN_MS)]
            elif level == "escape":
                steps = [("backward", ESCAPE_BACKWARD_MS), (turn_direction, ESCAPE_TURN_MS)]
            else:
                other = "right" if turn_direction == "left" else "left"
                steps = [("backward", HARD_ESCAPE_BACKUP_MS), (other, HARD_ESCAPE_TURN_MS)]
            self._recovery_steps = steps
            self._recovery_index = 0
            self._recovery_until = 0.0
            self._recovery_level = level
            self._recovery_speed = speed_level
            return await self._advance_recovery_locked()

    async def _advance_recovery_locked(self):
        self._last_autonomy_command_ts = self._now()
        if self.remote_owned or not self._recovery_steps:
            return self._last_result
        if self._now() < self._recovery_until:
            return self._last_result
        if self._recovery_until:
            self._recovery_index += 1
        if self._recovery_index >= len(self._recovery_steps):
            self._recovery_steps = []
            return await self._post_stop_locked(source="reactive", mode="hold")
        action, duration = self._recovery_steps[self._recovery_index]
        plan = MotionPlan(action, duration, self._recovery_speed, self._recovery_level,
                          "recovery", maneuver_id=f"recovery-{self._recovery_id}-{self._recovery_index}")
        result = await self._apply_locked(plan, "reactive")
        self._recovery_until = self._now() + duration / 1000.0
        return result

    async def advance_recovery(self):
        async with self._lock:
            return await self._advance_recovery_locked()

    async def sync_headlight(self, on, *, source):
        async with self._lock:
            await self.client.set_light(255 if on else 0, command_id=self._next_command_id(source),
                                        source=source, mode="light")

    async def stop_now(self, *, source, mode):
        async with self._lock:
            return await self._post_stop_locked(source=source, mode=mode)

    async def release_remote(self, *, only_if_inactive=False):
        async with self._lock:
            if only_if_inactive and self.remote_owned:
                return self._last_result
            self._last_remote_command_ts = -float("inf")
            return await self._post_stop_locked(source="remote_release", mode="manual")

    async def remote_move(self, direction, duration_ms=600):
        async with self._lock:
            self._last_remote_command_ts = self._now()
            self._recovery_steps = []
            speed = "slow" if state.last_scene_hazard == "hard" else "medium"
            return await self._apply_locked(MotionPlan(direction, 0, speed, "manual", "remote"), "remote")

    async def _heartbeat_loop(self):
        while self._running:
            await asyncio.sleep(LEASE_HEARTBEAT_SEC)
            await self._heartbeat_once()

    async def _heartbeat_once(self):
        async with self._lock:
            if not self._motion_command_id:
                return
            remote = self._active_source == "remote"
            age = self._now() - (self._last_remote_command_ts if remote else self._last_autonomy_command_ts)
            limit = REMOTE_RENEWAL_WINDOW_SEC if remote else config.AUTONOMY_RENEWAL_WINDOW_SEC
            try:
                if age > limit:
                    await self._post_stop_locked(source="input_expired", mode="hold")
                    return
                if self._timed_key and self._now() >= self._timed_until:
                    # Timed motions stop locally; retain their identity for deduplication.
                    self._motion_command_id = ""
                    return
                payload = await self.client.renew(self._motion_command_id,
                    command_id=self._next_command_id("lease"), source="lease", mode=self._active_mode)
                if payload.get("ok") is False or payload.get("busy") or payload.get("safety"):
                    self._heartbeat_safety_event = bool(payload.get("safety"))
                    await self._post_stop_locked(source="lease_rejected", mode="hold")
            except Exception as exc:
                # Never keep renewing an uncertain command. Firmware's watchdog is independent.
                self._clear_motion()
                self._recovery_steps = []
                self._heartbeat_safety_event = True
                log.warning("Motion renewal failed: %s", exc)
                with contextlib.suppress(Exception):
                    await self._post_stop_locked(source="failsafe", mode="hold")
