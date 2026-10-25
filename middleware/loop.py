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
    FAILSAFE_BACKOFF_SEC,
    GEMINI_MIN_INTERVAL,
    OPENROUTER_STRATEGY_MIN_INTERVAL,
    OPENROUTER_VLM_TIMEOUT_SEC,
    OPENROUTER_TIMEOUT_BACKOFF_SEC,
    OLLAMA_STRATEGY_MIN_INTERVAL,
    OLLAMA_VLM_TIMEOUT_SEC,
    OLLAMA_TIMEOUT_BACKOFF_SEC,
    STRATEGY_STALE_MULTIPLIER,
    SEMANTIC_TTL_SEC,
    VLM_OBSERVATION_MAX_AGE_SEC,
    STRATEGY_TIMEOUT_WINDOW_SEC,
    STRATEGY_TIMEOUT_STAGE2_COUNT,
    STRATEGY_TIMEOUT_STAGE3_COUNT,
    STRATEGY_LOCAL_ONLY_SEC,
    CONTROL_BUDGET_SEC,
    CONTROL_INACTIVITY_SEC,
    MOVE_ACTIONS,
    CAMERA_OUTAGE_RESTART_AFTER,
    CAMERA_OUTAGE_LOG_INTERVAL_SEC,
)
import config as _config_module
from scene import (
    VLMResponseError,
    _exc_summary,
    _is_vlm_quota_error,
    _is_placeholder_text,
    _frame_signature,
    _motion_score,
    _persona_reactive_thought,
    _system_prompt,
    _local_scene_assessment,
    _apply_vlm_semantic_contract,
    build_strategy_intent,
    ask_vlm,
)
from motion import MotionSupervisor, WheelsonClient
from planner import IntentPlanner
from perception import CameraStream, CameraOutage
from executive import SkillExecutive
from skills import vlm_discard_reason
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

    strategy_refresh_interval = {
        "gemini": GEMINI_MIN_INTERVAL,
        "openrouter": OPENROUTER_STRATEGY_MIN_INTERVAL,
        "ollama": OLLAMA_STRATEGY_MIN_INTERVAL,
    }[_config_module.PROVIDER]
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
        executive = SkillExecutive()
        camera = CameraStream(wheelson)
        outage = CameraOutage(
            restart_after=CAMERA_OUTAGE_RESTART_AFTER,
            log_interval_sec=CAMERA_OUTAGE_LOG_INTERVAL_SEC,
        )
        last_frame_id = 0
        planning_generation = 0
        # VLM validity generation: advances ONLY on safety-critical regime
        # changes (persona switch, human takeover/resume, safety event).
        # Recovery triggers and transient firmware-busy rejections bump
        # planning_generation but NOT this: they fire repeatedly inside one
        # ~20s VLM round trip, which discarded every result (generation churn).
        vlm_generation = 0
        started_at = asyncio.get_running_loop().time()

        await motion.start()

        try:
            await camera.start()
        except BaseException:
            await camera.close()
            await motion.close()
            raise

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

        def _register_vlm_timeout(now_ts: float) -> int:
            vlm_timeout_events.append(now_ts)
            _prune_vlm_timeouts(now_ts)
            timeout_count = len(vlm_timeout_events)

            stage = 1
            if timeout_count >= STRATEGY_TIMEOUT_STAGE3_COUNT:
                stage = 3
                state.vlm_disabled_until_ts = max(state.vlm_disabled_until_ts, now_ts + STRATEGY_LOCAL_ONLY_SEC)
            elif timeout_count >= STRATEGY_TIMEOUT_STAGE2_COUNT:
                stage = 2

            return stage

        async def request_strategy(frame_bytes, frame_telemetry, prompt, generation):
            request = ask_vlm(frame_bytes, frame_telemetry, prompt, client)
            provider_timeout = {
                "openrouter": OPENROUTER_VLM_TIMEOUT_SEC,
                "ollama": OLLAMA_VLM_TIMEOUT_SEC,
            }.get(_config_module.PROVIDER)
            result = await asyncio.wait_for(request, timeout=provider_timeout) if provider_timeout else await request
            return dict(result, captured_at=frame_telemetry["captured_at"],
                        frame_id=frame_telemetry["frame_id"], generation=generation)

        async def stop_with_failsafe(reason: str) -> None:
            nonlocal vlm_task, vlm_timeout_events
            log.warning("\U0001f6d1 [FAILSAFE] %s", reason)
            recovery.reset()
            executive.cancel("failsafe")
            planner.reset_maneuver()
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

        try:
            while state.is_running:
                cycle_start = loop.time()
                try:
                    jpeg_bytes, telemetry = camera.snapshot(loop.time())
                    if outage.note_recovered():
                        log.info("\U0001f4f7 [CAMERA] Feed recovered after outage; resuming")
                    state.camera_feed = camera.status()
                    new_frame = telemetry["frame_id"] != last_frame_id
                    telemetry["new_frame"] = new_frame
                    # Protocol 2 firmware supplies upright frames to every consumer.
                    state.last_frame_b64 = (await asyncio.to_thread(base64.b64encode, jpeg_bytes)).decode()

                    state.last_distance_cm = telemetry["distance_cm"]
                    state.last_nav_state = telemetry.get("nav_state", "UNKNOWN")
                    state.last_command_id = telemetry.get("active_command_id", state.last_command_id)
                    state.last_command_source = telemetry.get("active_command_source", state.last_command_source)
                    state.last_command_mode = telemetry.get("active_command_mode", state.last_command_mode)

                    if motion.consume_heartbeat_safety():
                        state.hard_stop_count += 1

                    motion_score = state.last_motion_score
                    if new_frame:
                        curr_sig = await asyncio.to_thread(_frame_signature, jpeg_bytes)
                        motion_score = _motion_score(state.prev_frame_signature, curr_sig)
                        state.prev_frame_signature = curr_sig
                        state.last_motion_score = motion_score
                        last_frame_id = telemetry["frame_id"]
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
                        planning_generation += 1
                        vlm_generation += 1  # persona change invalidates in-flight VLM plans
                        executive.cancel("persona_changed")
                        state.last_vlm_strategy = None
                        state.last_strategy_update_ts = 0.0
                        state.semantic_pending_hard = state.semantic_pending_blocked = 0
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
                            await motion.release_remote()
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

                    # Consume completed VLM task
                    if vlm_task and vlm_task.done():
                        try:
                            vlm_update = vlm_task.result()
                            discard_reason = vlm_discard_reason(
                                vlm_update, vlm_generation, now_ts, VLM_OBSERVATION_MAX_AGE_SEC
                            )
                            if discard_reason is not None:
                                raise VLMResponseError(f"Discarded stale observation ({discard_reason})", "")
                            state.last_vlm_strategy = None if motion.remote_owned else vlm_update.get("strategy")
                            state.last_vlm_frame_id = vlm_update["frame_id"]
                            state.last_vlm_capture_ts = vlm_update["captured_at"]
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
                            state.last_strategy_update_ts = vlm_update["captured_at"]
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
                            if _config_module.PROVIDER in {"ollama", "openrouter"}:
                                timeout_sec = (
                                    OPENROUTER_VLM_TIMEOUT_SEC
                                    if _config_module.PROVIDER == "openrouter"
                                    else OLLAMA_VLM_TIMEOUT_SEC
                                )
                                timeout_backoff = (
                                    OPENROUTER_TIMEOUT_BACKOFF_SEC
                                    if _config_module.PROVIDER == "openrouter"
                                    else OLLAMA_TIMEOUT_BACKOFF_SEC
                                )
                                log.warning(
                                    "\U0001f916 [VLM] %s refresh timed out after %.1fs; keeping previous strategy",
                                    _config_module.PROVIDER,
                                    timeout_sec,
                                )
                                stage = _register_vlm_timeout(now_ts)
                                if stage >= 3:
                                    log.warning(
                                        "\U0001f916 [VLM] timeout escalation -> local_only for %.0fs (count=%d)",
                                        STRATEGY_LOCAL_ONLY_SEC,
                                        len(vlm_timeout_events),
                                    )
                                vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + timeout_backoff)
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
                        vlm_task = asyncio.create_task(request_strategy(bytes(jpeg_bytes), dict(telemetry), system, vlm_generation))
                        last_vlm_request_ts = now_ts

                    strategy_age = (
                        now_ts - state.last_strategy_update_ts
                        if state.last_strategy_update_ts > 0.0
                        else float("inf")
                    )
                    strategy_semantic_expired = (
                        state.last_strategy_update_ts <= 0.0 or strategy_age >= SEMANTIC_TTL_SEC
                    )
                    if strategy_semantic_expired or state.last_strategy_source != "vlm":
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

                    if state.last_vlm_strategy is not None:
                        executive.submit(state.last_vlm_frame_id, state.last_vlm_strategy,
                            state.last_vlm_capture_ts + SEMANTIC_TTL_SEC, now_ts)

                    # -- S5: Human Override --
                    if state.remote_is_active or motion.remote_owned:
                        if not _remote_was_active:
                            # remote_move already atomically took ownership; do not stop its first input.
                            planner.reset_maneuver()
                            executive.cancel("human_takeover")
                            planning_generation += 1
                            vlm_generation += 1  # human takeover invalidates in-flight VLM plans
                            state.last_vlm_strategy = None
                            state.last_strategy_update_ts = 0.0
                            recovery.reset()
                            _remote_was_active = True
                            log.info("\U0001f3ae [REMOTE] %s took control", state.remote_controller_name)
                        # Keep displaying the last autonomous thought while human drives
                        result = motion.last_result
                        actual_action = result.action
                        actual_duration = 0
                        plan_speed = "remote"
                        plan_reason = "remote_override"
                    else:
                        # -- S2-S3: Recovery FSM / Autonomous Planner --
                        if _remote_was_active:
                            _remote_was_active = False
                            # Halt any residual remote motion before autonomy resumes.
                            await motion.release_remote(only_if_inactive=True)
                            planner.reset_maneuver()
                            planning_generation += 1
                            vlm_generation += 1  # autonomy resumes after human control
                            state.last_vlm_strategy = None
                            state.last_strategy_update_ts = 0.0
                            log.info("\U0001f916 [REMOTE] Autonomy resumed after '%s'", state.remote_controller_name)

                        trigger: RecoveryTrigger | None = None
                        if new_frame and not motion.recovery_active:
                            tracking_forward = motion.tracking_forward
                            trigger = recovery.observe(
                                tracking_forward=tracking_forward,
                                motion_score=motion_score,
                                telemetry=telemetry,
                                now_ts=now_ts,
                            )

                        plan_speed = "medium"
                        plan_reason = "recovery"

                        if motion.recovery_active:
                            recovery.pause_tracking()
                            result = await motion.advance_recovery()
                            actual_action, actual_duration = result.action, result.duration_ms
                            plan_reason = "recovery:running"
                        elif trigger:
                            executive.cancel("recovery")
                            planning_generation += 1
                            # NOTE: recovery does NOT bump vlm_generation: recovery
                            # nudges fire inside a VLM round trip, and a room-level
                            # VLM plan stays usable after a short autonomous recovery.
                            state.last_vlm_strategy = None
                            planner.reset_maneuver()
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
                            # Preserve startup timing despite the faster controller cadence.
                            planning_cycle = int((now_ts - started_at) / 0.4)
                            speed, _ = planner._scene_speed(planner._base_speed(), intent, telemetry, planning_cycle)
                            plan = executive.tick(intent, telemetry, now_ts, speed)
                            if plan is None:
                                if motion.active_source == "skill":
                                    planner.reset_maneuver()
                                plan = planner.plan(intent, telemetry, planning_cycle, now_ts)
                            else:
                                plan = planner.shape_plan(plan, telemetry, now_ts)
                            source = "skill" if plan.reason.startswith("skill:") else intent.source
                            result = await motion.apply_plan(plan, source=source)
                            state.last_thought = intent.thought
                            actual_action = result.action
                            actual_duration = result.duration_ms
                            plan_speed = plan.speed_level
                            plan_reason = plan.reason

                    state.executive = executive.snapshot()
                    if state.executive["objective"]:
                        state.last_objective = state.executive["objective"]
                    _update_strategy_mode(loop.time())

                    if result.safety:
                        state.hard_stop_count += 1
                        recovery.reset()
                    if result.safety or result.busy:
                        executive.cancel("firmware_rejected")
                        planner.reset_maneuver()
                        planning_generation += 1
                        if result.safety:
                            # A safety event changes the situation: in-flight VLM
                            # plans must not execute. Transient busy rejections
                            # do not invalidate them.
                            vlm_generation += 1
                        state.last_vlm_strategy = None
                        state.executive = executive.snapshot()

                    # Streak counts distinct motion commands, not latch repeats.
                    new_command_id = result.command_id or ""
                    if actual_action == state.last_action and new_command_id != state.last_command_id:
                        state.action_streak += 1
                    elif actual_action != state.last_action:
                        state.action_streak = 1

                    state.last_action = actual_action
                    state.last_duration_ms = actual_duration
                    state.cycle_count += 1

                    state.last_command_id = result.command_id or state.last_command_id
                    state.last_command_source = result.source or state.last_command_source
                    state.last_command_mode = result.mode or state.last_command_mode

                    _streak_int = int(round(recovery.low_motion_streak))
                    state.visual_stall_count = _streak_int
                    state.visual_stall_samples = _streak_int
                    state.no_progress_count = _streak_int
                    state.no_progress_samples = _streak_int
                    state.recovery_level = recovery.escalation_level
                    state.recovery_cooldown_cycles = int(round(recovery.cooldown_cycles))
                    state.low_motion_streak = _streak_int
                    state.speed_cap_level = plan_speed
                    state.speed_cap_reason = plan_reason

                    if intent.observation and (fresh_observation or not state.exploration_memory or
                            intent.observation != state.exploration_memory[-1]["observation"]):
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
                            "executive": state.executive,
                            "perception_age_s": round(now_ts - telemetry["captured_at"], 3),
                            "wheel_output": telemetry.get("wheel_output", []),
                            "wheel_target": telemetry.get("wheel_target", []),
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
                        motion.active_continuous_action
                        if motion.active_continuous_action in MOVE_ACTIONS
                        or motion.active_continuous_action == "drive"
                        else "off",
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
                    await asyncio.sleep(FAILSAFE_BACKOFF_SEC)
                except VLMResponseError as exc:
                    log.error("\U0001f916 [VLM] %s raw='%.200s'", exc, exc.raw.replace("\n", " "))
                    await stop_with_failsafe("invalid VLM response")
                    await asyncio.sleep(FAILSAFE_BACKOFF_SEC)
                except Exception as exc:
                    if "expired" in str(exc):
                        # Dead camera feed: hold the robot, say so once,
                        # restart the producer once, then stay quiet until
                        # frames return. No traceback spam, no driving on
                        # stale vision. The lease watchdog covers motion:
                        # no renewals are issued while snapshots fail.
                        action = outage.note_expired(loop.time())
                        state.camera_feed = camera.status()
                        if action == "log":
                            log.warning("\U0001f4f7 [CAMERA] Feed lost (%s); holding", exc)
                            await stop_with_failsafe("camera feed lost")
                        elif action == "restart":
                            log.warning(
                                "\U0001f4f7 [CAMERA] Feed still lost after %d expired snapshots; "
                                "restarting camera producer",
                                outage.count,
                            )
                            await stop_with_failsafe("camera feed lost; restarting producer")
                            with contextlib.suppress(Exception):
                                await camera.restart()
                        await asyncio.sleep(FAILSAFE_BACKOFF_SEC)
                    else:
                        log.error("\U0001f4a5 [LOOP] Unexpected error: %s", exc, exc_info=True)
                        await stop_with_failsafe("unexpected middleware loop error")
                        await asyncio.sleep(FAILSAFE_BACKOFF_SEC)

                elapsed = loop.time() - cycle_start
                await asyncio.sleep(max(0.0, LOOP_INTERVAL - elapsed))
        finally:
            if vlm_task:
                vlm_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await vlm_task
            await camera.close()
            await motion.close()
            _state_module._motion_supervisor = None
