"""
wheelson-explorer · middleware · loop.py
The main explorer loop.
"""

import asyncio
import base64
import contextlib
import logging
import signal
from datetime import datetime

import httpx

from state import (
    state,
    broadcast,
    _control_queue,
    _queue_lock,
    _persona_queue,
)
import state as _state_module
from config import (
    PERSONALITIES,
    LOOP_INTERVAL,
    PORT,
    MAX_LOG_SIZE,
    GEMINI_MIN_INTERVAL,
    OLLAMA_STRATEGY_MIN_INTERVAL,
    OLLAMA_VLM_TIMEOUT_SEC,
    OLLAMA_TIMEOUT_BACKOFF_SEC,
    STRATEGY_STALE_MULTIPLIER,
    SEMANTIC_TTL_SEC,
    STRATEGY_TIMEOUT_WINDOW_SEC,
    STRATEGY_TIMEOUT_STAGE2_COUNT,
    STRATEGY_TIMEOUT_STAGE3_COUNT,
    STRATEGY_LOCAL_ONLY_SEC,
    TIMEOUT_SCAN_STEPS,
    CONTROL_BUDGET_SEC,
    CONTROL_INACTIVITY_SEC,
    MOVE_ACTIONS,
)
import config as _config_module
from scene import (
    VLMResponseError,
    _exc_summary,
    _is_vlm_quota_error,
    _is_placeholder_text,
    flip_jpeg,
    _frame_signature,
    _motion_score,
    _persona_reactive_thought,
    _persona_timeout_scan_thought,
    _system_prompt,
    _local_scene_assessment,
    _apply_vlm_semantic_contract,
    build_strategy_intent,
    ask_vlm,
)
from motion import MotionSupervisor, MotionResult, WheelsonClient
from planner import IntentPlanner
from recovery import RecoveryController, RecoveryTrigger

log = logging.getLogger("wheelson")


async def explorer_loop() -> None:
    state.is_running = True

    def _shutdown_handler(signum, frame):
        log.warning("\U0001f6d1 [SHUTDOWN] Signal %d received, stopping motors", signum)
        state.is_running = False

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _shutdown_handler)
        except (OSError, ValueError):
            pass  # Can't set signal handlers in non-main thread
    system = _system_prompt(state.personality_key)
    personality = PERSONALITIES[state.personality_key]

    bootstrap_telemetry = {
        "brightness": "Normal",
        "obstacle_left_ratio": 0.0,
        "obstacle_right_ratio": 0.0,
        "visual_obstacle": False,
        "nav_state": "HOLD",
    }
    bootstrap_scene = _local_scene_assessment(bootstrap_telemetry)
    bootstrap_intent = build_strategy_intent(
        bootstrap_scene,
        bootstrap_telemetry,
        "bootstrap",
        state.personality_key,
    )
    if _is_placeholder_text(state.last_objective):
        state.last_objective = bootstrap_intent.objective
    if _is_placeholder_text(state.last_thought):
        state.last_thought = bootstrap_intent.thought
    state.last_vlm_headlight_pref = bootstrap_intent.headlight
    state.last_scene_frontier = bootstrap_intent.frontier
    state.last_scene_traversability = bootstrap_intent.traversability
    state.last_scene_novelty = bootstrap_intent.novelty
    state.last_scene_hazard = bootstrap_intent.hazard
    state.last_scene_observation = bootstrap_intent.observation
    state.last_vlm_commentary = bootstrap_intent.thought

    loop = asyncio.get_running_loop()
    state.last_strategy_source = "bootstrap"
    state.last_strategy_update_ts = loop.time()
    state.strategy_mode = "normal"
    state.strategy_degraded_level = 0
    state.semantic_confidence = 0.55
    state.semantic_pending_hard = 0
    state.semantic_pending_blocked = 0
    state.vlm_timeout_count = 0
    state.vlm_disabled_until_ts = 0.0

    strategy_refresh_interval = (
        GEMINI_MIN_INTERVAL if _config_module.PROVIDER == "gemini" else OLLAMA_STRATEGY_MIN_INTERVAL
    )
    strategy_stale_interval = strategy_refresh_interval * STRATEGY_STALE_MULTIPLIER

    log.info(
        "\U0001f916 Explorer started personality=%s %s interval=%.1fs",
        personality["emoji"],
        personality["name"],
        LOOP_INTERVAL,
    )
    log.info("\U0001f310 Dashboard -> http://localhost:%d", PORT)

    async with httpx.AsyncClient() as client:
        wheelson = WheelsonClient(client)
        planner = IntentPlanner()
        planner.set_persona(state.personality_key)
        motion = MotionSupervisor(wheelson)
        recovery = RecoveryController()

        await motion.start()

        # Initial firmware health check
        try:
            _, init_telemetry = await wheelson.fetch_frame()
            log.info(
                "\u2705 [STARTUP] Firmware health probe OK: nav=%s brightness=%s",
                init_telemetry.get("nav_state", "UNKNOWN"),
                init_telemetry.get("brightness", "Unknown"),
            )
        except Exception as init_exc:
            log.warning("\u26a0\ufe0f [STARTUP] Initial firmware health probe failed: %s", init_exc)

        _state_module._motion_supervisor = motion

        _remote_was_active = False

        last_vlm_request_ts = 0.0
        vlm_backoff_until_ts = 0.0
        vlm_task: asyncio.Task | None = None
        vlm_timeout_events: list[float] = []

        def _prune_vlm_timeouts(now_ts: float) -> None:
            nonlocal vlm_timeout_events
            vlm_timeout_events = [t for t in vlm_timeout_events if (now_ts - t) <= STRATEGY_TIMEOUT_WINDOW_SEC]

        def _update_strategy_mode(now_ts: float) -> None:
            _prune_vlm_timeouts(now_ts)
            state.vlm_timeout_count = len(vlm_timeout_events)
            if now_ts < state.vlm_disabled_until_ts:
                state.strategy_mode = "local_only"
                state.strategy_degraded_level = 2
                return
            if state.vlm_timeout_count >= STRATEGY_TIMEOUT_STAGE2_COUNT or state.last_strategy_source != "vlm":
                state.strategy_mode = "degraded"
                state.strategy_degraded_level = 1
                return
            state.strategy_mode = "normal"
            state.strategy_degraded_level = 0

        def _register_vlm_timeout(now_ts: float) -> tuple[int, int]:
            vlm_timeout_events.append(now_ts)
            _prune_vlm_timeouts(now_ts)
            timeout_count = len(vlm_timeout_events)

            stage = 1
            if timeout_count >= STRATEGY_TIMEOUT_STAGE3_COUNT:
                stage = 3
                state.vlm_disabled_until_ts = max(state.vlm_disabled_until_ts, now_ts + STRATEGY_LOCAL_ONLY_SEC)
            elif timeout_count >= STRATEGY_TIMEOUT_STAGE2_COUNT:
                stage = 2

            scan_steps = TIMEOUT_SCAN_STEPS + (stage - 1)
            return stage, scan_steps

        async def request_strategy(frame_bytes: bytes, frame_telemetry: dict) -> dict:
            if _config_module.PROVIDER == "ollama":
                return await asyncio.wait_for(
                    ask_vlm(frame_bytes, frame_telemetry, system, client),
                    timeout=OLLAMA_VLM_TIMEOUT_SEC,
                )
            return await ask_vlm(frame_bytes, frame_telemetry, system, client)

        async def stop_with_failsafe(reason: str) -> None:
            nonlocal vlm_task, vlm_timeout_events
            log.warning("\U0001f6d1 [FAILSAFE] %s", reason)
            recovery.reset()
            vlm_timeout_events = []
            state.strategy_mode = "degraded"
            state.strategy_degraded_level = 2
            state.visual_stall_count = 0
            state.visual_stall_samples = 0
            state.no_progress_count = 0
            state.no_progress_samples = 0
            state.recovery_level = 0
            state.recovery_cooldown_cycles = 0
            state.low_motion_streak = 0
            if vlm_task:
                vlm_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await vlm_task
                vlm_task = None
            with contextlib.suppress(Exception):
                await motion.stop_now(source="failsafe", mode="hold")
            # Health probe: verify firmware is still responding after failsafe stop
            try:
                await wheelson.fetch_frame()
                motion._current_speed = None  # Force re-send on next speed command
                log.info("\u2705 [FAILSAFE] Firmware health probe OK, speed state reset")
            except Exception as probe_exc:
                log.warning("\u26a0\ufe0f [FAILSAFE] Firmware health probe failed: %s", probe_exc)

        try:
            while state.is_running:
                cycle_start = loop.time()
                try:
                    jpeg_bytes, telemetry = await wheelson.fetch_frame()
                    jpeg_bytes = flip_jpeg(jpeg_bytes)
                    state.last_frame_b64 = base64.b64encode(jpeg_bytes).decode()

                    state.last_distance_cm = telemetry["distance_cm"]
                    state.last_nav_state = telemetry.get("nav_state", "UNKNOWN")
                    state.last_command_id = telemetry.get("active_command_id", state.last_command_id)
                    state.last_command_source = telemetry.get("active_command_source", state.last_command_source)
                    state.last_command_mode = telemetry.get("active_command_mode", state.last_command_mode)

                    if motion.consume_heartbeat_safety():
                        state.hard_stop_count += 1

                    curr_sig = _frame_signature(jpeg_bytes)
                    motion_score = _motion_score(state.prev_frame_signature, curr_sig)
                    state.prev_frame_signature = curr_sig
                    state.last_motion_score = motion_score
                    telemetry["motion_score"] = motion_score

                    now_ts = loop.time()

                    # -- Persona switch (requested via HTTP) --
                    _new_persona = None
                    while not _persona_queue.empty():
                        try:
                            _new_persona = _persona_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    if _new_persona and _new_persona in PERSONALITIES:
                        state.personality_key = _new_persona
                        planner.set_persona(_new_persona)
                        recovery.reset()
                        system = _system_prompt(_new_persona)
                        personality = PERSONALITIES[_new_persona]
                        log.info(
                            "\U0001f3ad [PERSONA] Switched to %s %s",
                            personality["emoji"],
                            personality["name"],
                        )

                    # -- Control queue advancement --
                    async with _queue_lock:
                        while _control_queue and (now_ts - _control_queue[0].joined_at) > CONTROL_BUDGET_SEC:
                            expired_sess = _control_queue.pop(0)
                            log.info("\U0001f3ae [REMOTE] Session expired for '%s'", expired_sess.name)
                        state.remote_queue_length = len(_control_queue)
                        if _control_queue:
                            _head = _control_queue[0]
                            state.remote_controller_name = _head.name
                            state.remote_control_remaining_s = max(
                                0.0, CONTROL_BUDGET_SEC - (now_ts - _head.joined_at)
                            )
                            state.remote_is_active = (
                                _head.last_command_at > 0
                                and (now_ts - _head.last_command_at) < CONTROL_INACTIVITY_SEC
                            )
                        else:
                            state.remote_controller_name = ""
                            state.remote_control_remaining_s = 0.0
                            state.remote_is_active = False

                    fresh_observation = ""
                    strategy_timeout_event = False
                    timeout_scan_steps = TIMEOUT_SCAN_STEPS

                    # Consume completed VLM task
                    if vlm_task and vlm_task.done():
                        try:
                            vlm_update = vlm_task.result()
                            vetted_scene, semantic_conf, contract_note = _apply_vlm_semantic_contract(vlm_update, telemetry)
                            vlm_backoff_until_ts = 0.0
                            state.last_vlm_headlight_pref = vetted_scene.get("headlight", False)
                            state.last_scene_frontier = str(vetted_scene.get("frontier", "unknown"))
                            state.last_scene_traversability = str(vetted_scene.get("traversability", "medium"))
                            state.last_scene_novelty = str(vetted_scene.get("novelty", "medium"))
                            state.last_scene_hazard = str(vetted_scene.get("hazard", "none"))
                            state.last_scene_observation = str(vetted_scene.get("observation", ""))
                            incoming_commentary = str(vetted_scene.get("commentary", "")).strip()
                            if incoming_commentary:
                                state.last_vlm_commentary = incoming_commentary
                            state.semantic_confidence = semantic_conf
                            state.last_strategy_source = "vlm"
                            state.last_strategy_update_ts = now_ts
                            fresh_observation = state.last_scene_observation
                            vlm_timeout_events = []
                            log.info(
                                "\U0001f9e0 [VLM_SCENE] frontier=%s trav=%s novelty=%s hazard=%s",
                                state.last_scene_frontier,
                                state.last_scene_traversability,
                                state.last_scene_novelty,
                                state.last_scene_hazard,
                            )
                            if contract_note != "ok":
                                log.info("\U0001f9e0 [SEMANTIC_CONTRACT] %s (conf=%.2f)", contract_note, semantic_conf)
                        except asyncio.TimeoutError:
                            if _config_module.PROVIDER == "ollama":
                                log.warning(
                                    "\U0001f916 [VLM] ollama refresh timed out after %.1fs; keeping previous strategy",
                                    OLLAMA_VLM_TIMEOUT_SEC,
                                )
                                stage, timeout_scan_steps = _register_vlm_timeout(now_ts)
                                if stage >= 3:
                                    log.warning(
                                        "\U0001f916 [VLM] timeout escalation -> local_only for %.0fs (count=%d)",
                                        STRATEGY_LOCAL_ONLY_SEC,
                                        len(vlm_timeout_events),
                                    )
                                vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + OLLAMA_TIMEOUT_BACKOFF_SEC)
                                strategy_timeout_event = True
                            else:
                                log.warning("\U0001f916 [VLM] gemini request timed out; switching to local scene fallback")
                                vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + strategy_refresh_interval)
                            state.last_strategy_update_ts = now_ts - max(strategy_stale_interval, SEMANTIC_TTL_SEC)
                        except VLMResponseError as exc:
                            log.warning(
                                "\U0001f916 [VLM] %s parse failure: %s; switching to local scene fallback",
                                _config_module.PROVIDER,
                                _exc_summary(exc),
                            )
                            state.semantic_confidence = min(state.semantic_confidence, 0.45)
                            vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + strategy_refresh_interval)
                            state.last_strategy_update_ts = now_ts - max(strategy_stale_interval, SEMANTIC_TTL_SEC)
                        except Exception as exc:
                            if _is_vlm_quota_error(exc):
                                log.warning(
                                    "\U0001f916 [VLM] %s quota/rate limited (%s); switching to local scene fallback",
                                    _config_module.PROVIDER,
                                    _exc_summary(exc),
                                )
                                vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + 60.0)
                            else:
                                log.warning(
                                    "\U0001f916 [VLM] %s strategy task error: %s; keeping previous strategy",
                                    _config_module.PROVIDER,
                                    _exc_summary(exc),
                                )
                                vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + strategy_refresh_interval)
                            state.semantic_confidence = min(state.semantic_confidence, 0.45)
                            state.last_strategy_update_ts = now_ts - max(strategy_stale_interval, SEMANTIC_TTL_SEC)
                        finally:
                            vlm_task = None

                    _update_strategy_mode(now_ts)

                    should_refresh_vlm = (
                        vlm_task is None
                        and now_ts >= vlm_backoff_until_ts
                        and now_ts >= state.vlm_disabled_until_ts
                        and (
                            state.last_strategy_update_ts <= 0.0
                            or (now_ts - last_vlm_request_ts) >= strategy_refresh_interval
                        )
                    )
                    if should_refresh_vlm:
                        vlm_task = asyncio.create_task(request_strategy(bytes(jpeg_bytes), dict(telemetry)))
                        last_vlm_request_ts = now_ts

                    strategy_age = (
                        now_ts - state.last_strategy_update_ts
                        if state.last_strategy_update_ts > 0.0
                        else float("inf")
                    )
                    strategy_semantic_expired = (
                        state.last_strategy_update_ts <= 0.0 or strategy_age >= SEMANTIC_TTL_SEC
                    )
                    if strategy_semantic_expired:
                        local_scene = _local_scene_assessment(telemetry)
                        prev_source = state.last_strategy_source
                        state.last_vlm_headlight_pref = local_scene["headlight"]
                        state.last_scene_frontier = str(local_scene["frontier"])
                        state.last_scene_traversability = str(local_scene["traversability"])
                        state.last_scene_novelty = str(local_scene["novelty"])
                        state.last_scene_hazard = str(local_scene["hazard"])
                        if local_scene.get("observation"):
                            state.last_scene_observation = str(local_scene["observation"])
                        state.last_strategy_source = "local"
                        state.last_strategy_update_ts = now_ts
                        state.semantic_confidence = 0.55
                        state.semantic_pending_hard = max(0, state.semantic_pending_hard - 1)
                        state.semantic_pending_blocked = max(0, state.semantic_pending_blocked - 1)
                        if prev_source != "local":
                            log.info(
                                "\U0001f9e0 [LOCAL_SCENE] nav=%s light=%s side(L=%.2f R=%.2f) -> frontier=%s hazard=%s",
                                state.last_nav_state,
                                telemetry.get("brightness", "Unknown"),
                                telemetry.get("obstacle_left_ratio", 0.0),
                                telemetry.get("obstacle_right_ratio", 0.0),
                                state.last_scene_frontier,
                                state.last_scene_hazard,
                            )

                    _update_strategy_mode(now_ts)

                    telemetry["strategy_timeout_event"] = strategy_timeout_event
                    telemetry["timeout_scan_steps"] = timeout_scan_steps
                    telemetry["strategy_source"] = state.last_strategy_source
                    telemetry["strategy_age_sec"] = strategy_age if strategy_age != float("inf") else 999.0
                    telemetry["strategy_mode"] = state.strategy_mode
                    telemetry["strategy_degraded"] = state.strategy_degraded_level > 0
                    telemetry["strategy_degraded_level"] = state.strategy_degraded_level
                    telemetry["semantic_confidence"] = state.semantic_confidence

                    vlm = {
                        "observation": fresh_observation or state.last_scene_observation,
                        "commentary": state.last_vlm_commentary,
                        "headlight": state.last_vlm_headlight_pref,
                        "frontier": state.last_scene_frontier,
                        "traversability": state.last_scene_traversability,
                        "novelty": state.last_scene_novelty,
                        "hazard": state.last_scene_hazard,
                    }

                    intent = build_strategy_intent(vlm, telemetry, state.last_strategy_source, state.personality_key)
                    state.last_scene_frontier = intent.frontier
                    state.last_scene_traversability = intent.traversability
                    state.last_scene_novelty = intent.novelty
                    state.last_scene_hazard = intent.hazard
                    if intent.observation:
                        state.last_scene_observation = intent.observation
                    state.last_objective = intent.objective

                    # Headlight sync via single motion authority
                    desired_headlight = intent.headlight or telemetry.get("brightness") == "Dark"
                    if desired_headlight != state.last_headlight:
                        await motion.sync_headlight(desired_headlight, source="planner")
                        state.last_headlight = desired_headlight

                    # -- S5: Human Override --
                    if state.remote_is_active:
                        if not _remote_was_active:
                            # Transition: halt any ongoing autonomous motion
                            await motion.stop_now(source="remote", mode="manual")
                            recovery.reset()
                            _remote_was_active = True
                            log.info("\U0001f3ae [REMOTE] %s took control", state.remote_controller_name)
                        # Keep displaying the last autonomous thought while human drives
                        result = MotionResult(
                            action=state.last_action,
                            duration_ms=0,
                            safety=False,
                            busy=False,
                            command_id=state.last_command_id,
                            source="remote",
                            mode="manual",
                        )
                        actual_action = state.last_action
                        actual_duration = 0
                        plan_speed = "remote"
                        plan_reason = "remote_override"
                    else:
                        # -- S2-S3: Recovery FSM / Autonomous Planner --
                        if _remote_was_active:
                            _remote_was_active = False
                            log.info("\U0001f916 [REMOTE] Autonomy resumed after '%s'", state.remote_controller_name)

                        trigger: RecoveryTrigger | None = None
                        if not strategy_timeout_event:
                            tracking_forward = (
                                state.last_action == "forward"
                                or motion.active_continuous_action == "forward"
                            )
                            trigger = recovery.observe(
                                tracking_forward=tracking_forward,
                                motion_score=motion_score,
                                telemetry=telemetry,
                            )

                        plan_speed = "medium"
                        plan_reason = "recovery"

                        if trigger:
                            reactive_thought = _persona_reactive_thought(state.personality_key, trigger.level)
                            result = await motion.execute_recovery(
                                level=trigger.level,
                                turn_direction=trigger.turn_direction,
                                speed_level="medium",
                                source="reactive",
                            )
                            state.last_thought = reactive_thought
                            state.last_strategy_source = "reactive"
                            state.last_strategy_update_ts = loop.time()
                            actual_action = result.action
                            actual_duration = result.duration_ms
                            plan_speed = "medium"
                            plan_reason = f"recovery:{trigger.level}"
                        else:
                            plan = planner.plan(intent, telemetry, state.cycle_count, now_ts)
                            fallback_plan = plan.reason.startswith("gate:strategist_timeout")
                            source = "strategist_fallback" if fallback_plan else intent.source
                            result = await motion.apply_plan(plan, source=source)
                            if fallback_plan:
                                state.last_thought = _persona_timeout_scan_thought(state.personality_key)
                                state.last_strategy_source = "fallback"
                                state.last_strategy_update_ts = loop.time()
                                state.semantic_confidence = min(state.semantic_confidence, 0.5)
                            else:
                                state.last_thought = intent.thought
                            actual_action = result.action
                            actual_duration = result.duration_ms
                            plan_speed = plan.speed_level
                            plan_reason = plan.reason

                    _update_strategy_mode(loop.time())

                    if result.safety:
                        state.hard_stop_count += 1
                        recovery.reset()

                    if actual_action == state.last_action:
                        state.action_streak += 1
                    else:
                        state.action_streak = 1

                    state.last_action = actual_action
                    state.last_duration_ms = actual_duration
                    state.cycle_count += 1

                    state.last_command_id = result.command_id or state.last_command_id
                    state.last_command_source = result.source or state.last_command_source
                    state.last_command_mode = result.mode or state.last_command_mode

                    state.visual_stall_count = recovery.low_motion_streak
                    state.visual_stall_samples = recovery.low_motion_streak
                    state.no_progress_count = recovery.low_motion_streak
                    state.no_progress_samples = recovery.low_motion_streak
                    state.recovery_level = recovery.escalation_level
                    state.recovery_cooldown_cycles = recovery.cooldown_cycles
                    state.low_motion_streak = recovery.low_motion_streak
                    state.speed_cap_level = plan_speed
                    state.speed_cap_reason = plan_reason

                    if intent.observation:
                        mem_entry = {
                            "ts": datetime.now().strftime("%H:%M:%S"),
                            "action": actual_action,
                            "duration_ms": actual_duration,
                            "distance_cm": round(state.last_distance_cm, 1),
                            "observation": intent.observation,
                        }
                        state.exploration_memory.append(mem_entry)
                        if len(state.exploration_memory) > 10:
                            state.exploration_memory.pop(0)

                    entry = {
                        "ts": datetime.now().strftime("%H:%M:%S"),
                        "thought": state.last_thought,
                        "action": actual_action,
                        "duration_ms": actual_duration,
                        "distance_cm": round(state.last_distance_cm, 1),
                        "safety": result.safety,
                        "nav_state": state.last_nav_state,
                        "strategy_source": state.last_strategy_source,
                        "strategy_mode": state.strategy_mode,
                        "strategy_degraded_level": state.strategy_degraded_level,
                        "semantic_confidence": round(state.semantic_confidence, 2),
                        "scene_frontier": intent.frontier,
                        "scene_traversability": intent.traversability,
                        "scene_novelty": intent.novelty,
                        "scene_hazard": intent.hazard,
                        "command_id": state.last_command_id,
                        "command_source": state.last_command_source,
                        "command_mode": state.last_command_mode,
                        "recovery_level": state.recovery_level,
                        "low_motion_streak": state.low_motion_streak,
                        "cooldown_cycles": state.recovery_cooldown_cycles,
                        "speed_cap": state.speed_cap_level,
                        "speed_cap_reason": state.speed_cap_reason,
                    }
                    state.event_log.append(entry)
                    if len(state.event_log) > MAX_LOG_SIZE:
                        state.event_log.pop(0)

                    broadcast(
                        {
                            "frame_b64": state.last_frame_b64,
                            "thought": state.last_thought,
                            "action": actual_action,
                            "duration_ms": actual_duration,
                            "distance_cm": round(state.last_distance_cm, 1),
                            "personality": state.personality_key,
                            "cycle": state.cycle_count,
                            "safety": result.safety,
                            "nav_state": state.last_nav_state,
                            "strategy_mode": state.strategy_mode,
                            "strategy_degraded_level": state.strategy_degraded_level,
                            "semantic_confidence": round(state.semantic_confidence, 2),
                            "command_id": state.last_command_id,
                            "command_source": state.last_command_source,
                            "command_mode": state.last_command_mode,
                            "scene_frontier": intent.frontier,
                            "scene_traversability": intent.traversability,
                            "scene_novelty": intent.novelty,
                            "scene_hazard": intent.hazard,
                            "remote_controller": state.remote_controller_name,
                            "remote_queue_length": state.remote_queue_length,
                            "remote_remaining_s": round(state.remote_control_remaining_s, 1),
                            "remote_is_active": state.remote_is_active,
                            "ts": entry["ts"],
                        }
                    )

                    log.info(
                        "\U0001f9ed cycle=%d nav=%s action=%s dur=%dms lease=%s dist=%.1fcm motion=%.2f streak=%d rec_lvl=%d rec_cd=%d spd=%s src=%s mode=%s lvl=%d conf=%.2f cmd=%s/%s/%s scene(f=%s t=%s n=%s h=%s) visual=%s side(L=%.2f R=%.2f) light=%s safety=%s objective='%.64s'",
                        state.cycle_count,
                        state.last_nav_state,
                        actual_action,
                        actual_duration,
                        motion.active_continuous_action if motion.active_continuous_action in MOVE_ACTIONS else "off",
                        state.last_distance_cm,
                        motion_score,
                        state.low_motion_streak,
                        state.recovery_level,
                        state.recovery_cooldown_cycles,
                        state.speed_cap_level,
                        state.last_strategy_source,
                        state.strategy_mode,
                        state.strategy_degraded_level,
                        state.semantic_confidence,
                        state.last_command_id,
                        state.last_command_source,
                        state.last_command_mode,
                        intent.frontier,
                        intent.traversability,
                        intent.novelty,
                        intent.hazard,
                        telemetry.get("visual_obstacle", False),
                        telemetry.get("obstacle_left_ratio", 0.0),
                        telemetry.get("obstacle_right_ratio", 0.0),
                        telemetry.get("brightness", "Unknown"),
                        result.safety,
                        state.last_objective,
                    )

                except httpx.RequestError as exc:
                    log.error("\U0001f4e1 [NETWORK] Cannot reach Wheelson at %s: %s", _config_module.WHEELSON_IP, exc)
                    await stop_with_failsafe("network error while robot may be moving")
                except VLMResponseError as exc:
                    log.error("\U0001f916 [VLM] %s raw='%.200s'", exc, exc.raw.replace("\n", " "))
                    await stop_with_failsafe("invalid VLM response")
                except Exception as exc:
                    log.error("\U0001f4a5 [LOOP] Unexpected error: %s", exc, exc_info=True)
                    await stop_with_failsafe("unexpected middleware loop error")

                elapsed = loop.time() - cycle_start
                await asyncio.sleep(max(0.0, LOOP_INTERVAL - elapsed))
        finally:
            if vlm_task:
                vlm_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await vlm_task
            await motion.close()
