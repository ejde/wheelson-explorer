"""
wheelson-explorer · middleware · routes.py
Core FastAPI routes (dashboard, SSE stream, snapshot, health).
"""

import asyncio
import base64
import json
import os
from datetime import datetime
from typing import AsyncGenerator

from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from state import state, _sse_subscribers
from config import PERSONALITIES, WHEELSON_IP


async def dashboard() -> str:
    path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(path) as f:
        return f.read()


async def sse_stream() -> StreamingResponse:
    async def generator() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _sse_subscribers.add(q)
        initial = {
            "frame_b64": state.last_frame_b64,
            "thought": state.last_thought,
            "action": state.last_action,
            "duration_ms": state.last_duration_ms,
            "distance_cm": state.last_distance_cm,
            "personality": state.personality_key,
            "cycle": state.cycle_count,
            "safety": False,
            "ts": datetime.now().strftime("%H:%M:%S"),
        }
        yield f"data: {json.dumps(initial)}\n\n"
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            _sse_subscribers.discard(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def snapshot() -> Response:
    if not state.last_frame_b64:
        return Response(status_code=503, content="No frame captured yet")
    return Response(content=base64.b64decode(state.last_frame_b64), media_type="image/jpeg")


async def health() -> JSONResponse:
    p = PERSONALITIES[state.personality_key]
    now_ts = asyncio.get_running_loop().time()
    strategy_age_sec = (
        now_ts - state.last_strategy_update_ts if state.last_strategy_update_ts > 0.0 else -1.0
    )
    return JSONResponse(
        {
            "status": "ok",
            "personality": state.personality_key,
            "personality_name": p["name"],
            "cycle_count": state.cycle_count,
            "hard_stops": state.hard_stop_count,
            "last_action": state.last_action,
            "last_thought": state.last_thought[:80],
            "last_distance_cm": state.last_distance_cm,
            "last_nav_state": state.last_nav_state,
            "visual_stall_count": state.visual_stall_count,
            "visual_stall_samples": state.visual_stall_samples,
            "no_progress_count": state.no_progress_count,
            "no_progress_samples": state.no_progress_samples,
            "recovery_level": state.recovery_level,
            "recovery_cooldown_cycles": state.recovery_cooldown_cycles,
            "low_motion_streak": state.low_motion_streak,
            "speed_cap_level": state.speed_cap_level,
            "speed_cap_reason": state.speed_cap_reason,
            "strategy_source": state.last_strategy_source,
            "strategy_mode": state.strategy_mode,
            "strategy_degraded_level": state.strategy_degraded_level,
            "strategy_age_sec": round(strategy_age_sec, 2) if strategy_age_sec >= 0 else -1,
            "semantic_confidence": round(state.semantic_confidence, 2),
            "semantic_pending_hard": state.semantic_pending_hard,
            "semantic_pending_blocked": state.semantic_pending_blocked,
            "vlm_timeout_count": state.vlm_timeout_count,
            "vlm_disabled_remaining_sec": round(max(0.0, state.vlm_disabled_until_ts - now_ts), 2),
            "last_motion_score": state.last_motion_score,
            "scene_frontier": state.last_scene_frontier,
            "scene_traversability": state.last_scene_traversability,
            "scene_novelty": state.last_scene_novelty,
            "scene_hazard": state.last_scene_hazard,
            "command_id": state.last_command_id,
            "command_source": state.last_command_source,
            "command_mode": state.last_command_mode,
            "is_running": state.is_running,
            "wheelson_ip": WHEELSON_IP,
        }
    )
