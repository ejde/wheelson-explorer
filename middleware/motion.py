"""
wheelson-explorer · middleware · motion.py
Motion supervisor + wheelson client (single writer to /move).
"""

import asyncio
import contextlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime

import httpx

from state import state
from scene import _as_bool, _safe_int
from config import (
    WHEELSON_IP,
    MOVE_ACTIONS,
    ALL_ACTIONS,
    LEASE_HEARTBEAT_SEC,
    LEASE_MISSED_LIMIT,
    SIDE_STEER_TURN_MS,
    ESCAPE_BACKWARD_MS,
    ESCAPE_TURN_MS,
    HARD_ESCAPE_BACKUP_MS,
    HARD_ESCAPE_TURN_MS,
)

log = logging.getLogger("wheelson")


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


class WheelsonClient:
    def __init__(self, http_client: httpx.AsyncClient):
        self.http = http_client

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
        except Exception:
            pass

        return resp.content, telemetry

    async def _post_move(self, payload: dict, timeout: float = 10.0) -> dict:
        resp = await self.http.post(f"http://{WHEELSON_IP}/move", json=payload, timeout=timeout)
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
            return resp.json()
        except ValueError:
            return {
                "ok": True,
                "busy": False,
                "safety": False,
                "action": payload.get("action", payload.get("direction", "stop")),
                "duration_ms": payload.get("duration_ms", 0),
                "command_id": payload.get("command_id", ""),
                "source": payload.get("source", ""),
                "mode": payload.get("mode", ""),
            }

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
    def __init__(self, client: WheelsonClient):
        self.client = client
        self._lock = asyncio.Lock()
        self._current_speed: str | None = None
        self._active_continuous_action: str = "stop"
        self._active_mode: str = "idle"
        self._active_source: str = "bootstrap"
        self._last_command_id: str = "boot"

        self._seq = 0
        self._session = datetime.now().strftime("%H%M%S")

        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._missed_lease_count = 0
        self._heartbeat_safety_event = False

    @property
    def active_continuous_action(self) -> str:
        return self._active_continuous_action

    @property
    def active_mode(self) -> str:
        return self._active_mode

    @property
    def active_source(self) -> str:
        return self._active_source

    @property
    def last_command_id(self) -> str:
        return self._last_command_id

    def consume_heartbeat_safety(self) -> bool:
        fired = self._heartbeat_safety_event
        self._heartbeat_safety_event = False
        return fired

    def _next_command_id(self, source: str) -> str:
        self._seq += 1
        src = re.sub(r"[^a-z0-9]+", "", source.lower())[:8] or "cmd"
        return f"{self._session}-{src}-{self._seq:05d}"

    async def start(self) -> None:
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def close(self) -> None:
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
        with contextlib.suppress(Exception):
            await self.stop_now(source="shutdown", mode="hold")

    async def _ensure_speed_locked(self, level: str, *, source: str, mode: str) -> None:
        if self._current_speed == level:
            return
        command_id = self._next_command_id(source)
        payload = await self.client.set_speed(level, command_id=command_id, source=source, mode=mode)
        self._current_speed = level
        self._last_command_id = str(payload.get("command_id", command_id))

    def _result_from_payload(
        self,
        payload: dict,
        *,
        fallback_action: str,
        fallback_duration_ms: int,
        fallback_command_id: str,
        source: str,
        mode: str,
    ) -> MotionResult:
        action = str(payload.get("action", fallback_action)).lower()
        if action not in ALL_ACTIONS:
            action = fallback_action
        duration_ms = _safe_int(payload.get("duration_ms", fallback_duration_ms), fallback_duration_ms)
        busy = _as_bool(payload.get("busy", False))
        safety = _as_bool(payload.get("safety", False))
        command_id = str(payload.get("command_id", fallback_command_id) or fallback_command_id)
        source_out = str(payload.get("source", source) or source)
        mode_out = str(payload.get("mode", mode) or mode)
        self._last_command_id = command_id
        return MotionResult(
            action=action,
            duration_ms=duration_ms,
            safety=safety,
            busy=busy,
            command_id=command_id,
            source=source_out,
            mode=mode_out,
        )

    async def _post_stop_locked(self, *, source: str, mode: str) -> MotionResult:
        command_id = self._next_command_id(source)
        payload = await self.client.stop(command_id=command_id, source=source, mode=mode)
        self._active_continuous_action = "stop"
        self._active_mode = "idle"
        self._active_source = source
        return self._result_from_payload(
            payload,
            fallback_action="stop",
            fallback_duration_ms=0,
            fallback_command_id=command_id,
            source=source,
            mode=mode,
        )

    async def _post_timed_locked(self, action: str, duration_ms: int, *, source: str, mode: str) -> MotionResult:
        command_id = self._next_command_id(source)
        payload = await self.client.move_timed(
            action,
            duration_ms,
            command_id=command_id,
            source=source,
            mode=mode,
        )
        self._active_continuous_action = "stop"
        self._active_mode = "idle"
        self._active_source = source
        return self._result_from_payload(
            payload,
            fallback_action=action,
            fallback_duration_ms=duration_ms,
            fallback_command_id=command_id,
            source=source,
            mode=mode,
        )

    async def _post_continuous_locked(self, action: str, *, source: str, mode: str) -> MotionResult:
        command_id = self._next_command_id(source)
        payload = await self.client.move_indefinitely(
            action,
            command_id=command_id,
            source=source,
            mode=mode,
        )
        result = self._result_from_payload(
            payload,
            fallback_action=action,
            fallback_duration_ms=0,
            fallback_command_id=command_id,
            source=source,
            mode=mode,
        )
        if not result.busy and not result.safety and result.action in MOVE_ACTIONS:
            self._active_continuous_action = result.action
            self._active_mode = mode
            self._active_source = source
        else:
            self._active_continuous_action = "stop"
            self._active_mode = "idle"
            self._active_source = source
        return result

    async def apply_plan(self, plan: MotionPlan, *, source: str) -> MotionResult:
        async with self._lock:
            await self._ensure_speed_locked(plan.speed_level, source=source, mode=plan.mode)
            if plan.action == "stop":
                return await self._post_stop_locked(source=source, mode=plan.mode)
            if plan.duration_ms > 0:
                return await self._post_timed_locked(
                    plan.action,
                    plan.duration_ms,
                    source=source,
                    mode=plan.mode,
                )
            return await self._post_continuous_locked(plan.action, source=source, mode=plan.mode)

    async def execute_recovery(
        self,
        *,
        level: str,
        turn_direction: str,
        speed_level: str,
        source: str,
    ) -> MotionResult:
        async with self._lock:
            await self._ensure_speed_locked(speed_level, source=source, mode=level)
            await self._post_stop_locked(source=source, mode=level)

            steps: list[tuple[str, int]]
            if level == "nudge":
                steps = [(turn_direction, SIDE_STEER_TURN_MS)]
            elif level == "escape":
                steps = [("backward", ESCAPE_BACKWARD_MS), (turn_direction, ESCAPE_TURN_MS)]
            else:
                hard_turn = "right" if turn_direction == "left" else "left"
                steps = [("backward", HARD_ESCAPE_BACKUP_MS), (hard_turn, HARD_ESCAPE_TURN_MS)]

            result = MotionResult(
                action="stop",
                duration_ms=0,
                safety=False,
                busy=False,
                command_id=self._last_command_id,
                source=source,
                mode=level,
            )
            for action, duration in steps:
                result = await self._post_timed_locked(action, duration, source=source, mode=level)
                if result.safety:
                    break
            return result

    async def sync_headlight(self, on: bool, *, source: str) -> None:
        async with self._lock:
            command_id = self._next_command_id(source)
            payload = await self.client.set_light(
                255 if on else 0,
                command_id=command_id,
                source=source,
                mode="light",
            )
            self._last_command_id = str(payload.get("command_id", command_id))

    async def stop_now(self, *, source: str, mode: str) -> MotionResult:
        async with self._lock:
            return await self._post_stop_locked(source=source, mode=mode)

    async def remote_move(self, direction: str, duration_ms: int = 600) -> MotionResult:
        """Execute a single remote-control command. Called directly by HTTP handler."""
        # Cap speed based on current scene hazard
        speed = "medium"
        if state.last_scene_hazard == "hard" or state.last_scene_traversability == "low":
            speed = "slow"
        async with self._lock:
            await self._ensure_speed_locked(level=speed, source="remote", mode="manual")
            if direction == "stop":
                return await self._post_stop_locked(source="remote", mode="manual")
            return await self._post_timed_locked(direction, duration_ms, source="remote", mode="manual")

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(LEASE_HEARTBEAT_SEC)
            async with self._lock:
                if self._active_continuous_action not in MOVE_ACTIONS:
                    self._missed_lease_count = 0
                    continue

                command_id = self._next_command_id("lease")
                try:
                    payload = await self.client.move_indefinitely(
                        self._active_continuous_action,
                        command_id=command_id,
                        source="lease",
                        mode=self._active_mode,
                    )
                    self._last_command_id = str(payload.get("command_id", command_id))
                    self._missed_lease_count = 0

                    if _as_bool(payload.get("safety", False)):
                        log.warning(
                            "\U0001f6d1 [LEASE] firmware safety override while renewing %s",
                            self._active_continuous_action,
                        )
                        self._active_continuous_action = "stop"
                        self._active_mode = "idle"
                        self._heartbeat_safety_event = True
                except httpx.RequestError as exc:
                    self._missed_lease_count += 1
                    log.warning(
                        "\U0001f4e1 [LEASE] heartbeat failed (%d/%d) action=%s err=%s",
                        self._missed_lease_count,
                        LEASE_MISSED_LIMIT,
                        self._active_continuous_action,
                        exc,
                    )
                    if self._missed_lease_count >= LEASE_MISSED_LIMIT:
                        log.warning("\U0001f6d1 [LEASE] missed too many heartbeats, forcing stop")
                        try:
                            await self._post_stop_locked(source="failsafe", mode="hold")
                        except Exception as stop_exc:
                            log.warning("\U0001f6d1 [LEASE] failsafe stop command failed: %s", stop_exc)
                        finally:
                            self._active_continuous_action = "stop"
                            self._active_mode = "idle"
                            self._heartbeat_safety_event = True
