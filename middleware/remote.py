"""
wheelson-explorer · middleware · remote.py
Remote control + chat HTTP endpoints.
"""

import asyncio
import logging

import google.genai as genai
from google.genai import types as genai_types
import httpx
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from state import (
    state,
    broadcast,
    _control_queue,
    _queue_lock,
    _motion_supervisor,
    _persona_queue,
    _chat_timestamps,
    CHAT_RATE_LIMIT,
    CHAT_RATE_WINDOW,
    ControlSession,
)
from config import (
    PERSONALITIES,
    ALL_ACTIONS,
    CONTROL_BUDGET_SEC,
    GEMINI_API_KEY,
    OLLAMA_BASE_URL,
)
from scene import _exc_summary
import config as _config_module
import state as _state_module

from datetime import datetime
import secrets

log = logging.getLogger("wheelson")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class JoinQueueRequest(BaseModel):
    name: str

class LeaveQueueRequest(BaseModel):
    token: str

class RemoteControlRequest(BaseModel):
    token: str
    direction: str

class PersonaRequest(BaseModel):
    persona: str

class ChatRequest(BaseModel):
    message: str
    name: str = "visitor"


# ---------------------------------------------------------------------------
# Chat helper
# ---------------------------------------------------------------------------
async def _chat_response(message: str, name: str) -> str:
    """Generate a short in-character reply grounded in the current scene."""
    p = PERSONALITIES[state.personality_key]
    system = p["system_prompt"]

    action_desc = state.last_action if state.last_action != "stop" else "stopped"
    scene = state.last_scene_observation or "an unclear scene"
    dist = f"{state.last_distance_cm:.0f} cm" if state.last_distance_cm < 900 else "open space"

    prompt = (
        f"You are {p['name']} \u2014 a robot currently {action_desc}ing.\n"
        f"What you can see right now: {scene}\n"
        f"Distance ahead: {dist}.\n"
        f"Your last thought: {state.last_thought}\n\n"
        f"{name} says to you: \"{message}\"\n\n"
        f"Reply in character as {p['name']} in at most 3 sentences. "
        f"Be aware of your surroundings. Do not output JSON or motor commands."
    )

    if _config_module.PROVIDER == "gemini":
        loop = asyncio.get_running_loop()
        gc = genai.Client(api_key=GEMINI_API_KEY)
        resp = await loop.run_in_executor(
            None,
            lambda: gc.models.generate_content(
                model=_config_module.ACTIVE_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.8,
                    max_output_tokens=180,
                ),
            ),
        )
        return resp.text.strip()
    else:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": _config_module.ACTIVE_MODEL,
                    "system": system,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 120, "temperature": 0.8},
                },
                timeout=20.0,
            )
            return resp.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# Route functions
# ---------------------------------------------------------------------------
async def queue_join(req: JoinQueueRequest) -> JSONResponse:
    name = req.name.strip()[:32]
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    token = secrets.token_urlsafe(16)
    async with _queue_lock:
        now = asyncio.get_event_loop().time()
        # Don't allow the same name to join twice
        if any(s.name == name for s in _control_queue):
            return JSONResponse({"error": "already_in_queue"}, status_code=409)
        session = ControlSession(token=token, name=name, joined_at=now)
        _control_queue.append(session)
        position = len(_control_queue)
    log.info("\U0001f3ae [REMOTE] '%s' joined queue at position %d", name, position)
    return JSONResponse({
        "token": token,
        "position": position,
        "budget_sec": CONTROL_BUDGET_SEC,
    })


async def queue_leave(req: LeaveQueueRequest) -> JSONResponse:
    async with _queue_lock:
        before = len(_control_queue)
        _control_queue[:] = [s for s in _control_queue if s.token != req.token]
        removed = before - len(_control_queue)
    return JSONResponse({"ok": True, "removed": removed})


async def remote_control(req: RemoteControlRequest) -> JSONResponse:
    if not _control_queue or _control_queue[0].token != req.token:
        return JSONResponse({"error": "not_your_turn"}, status_code=403)
    session = _control_queue[0]
    now = asyncio.get_event_loop().time()
    if (now - session.joined_at) > CONTROL_BUDGET_SEC:
        return JSONResponse({"error": "session_expired"}, status_code=403)
    direction = req.direction
    if direction not in ALL_ACTIONS:
        return JSONResponse({"error": "invalid_direction"}, status_code=400)
    session.last_command_at = now
    motion_sup = _state_module._motion_supervisor
    if motion_sup is None:
        return JSONResponse({"error": "not_ready"}, status_code=503)
    await motion_sup.remote_move(direction)
    return JSONResponse({"ok": True})


async def switch_persona(req: PersonaRequest) -> JSONResponse:
    if req.persona not in PERSONALITIES:
        return JSONResponse({"error": "invalid_persona", "valid": list(PERSONALITIES.keys())}, status_code=400)
    try:
        _persona_queue.put_nowait(req.persona)
    except asyncio.QueueFull:
        return JSONResponse({"error": "too_many_pending_switches"}, status_code=429)
    p = PERSONALITIES[req.persona]
    log.info("\U0001f3ad [PERSONA] Switch requested -> %s %s", p["emoji"], p["name"])
    return JSONResponse({"ok": True, "persona": req.persona})


async def queue_status() -> JSONResponse:
    async with _queue_lock:
        now = asyncio.get_event_loop().time()
        entries = [
            {
                "name": s.name,
                "position": i + 1,
                "remaining_s": round(max(0.0, CONTROL_BUDGET_SEC - (now - s.joined_at)), 1),
            }
            for i, s in enumerate(_control_queue)
        ]
    return JSONResponse({"queue": entries, "budget_sec": CONTROL_BUDGET_SEC})


async def chat_endpoint(req: ChatRequest) -> JSONResponse:
    now = asyncio.get_event_loop().time()
    _chat_timestamps[:] = [t for t in _chat_timestamps if (now - t) < CHAT_RATE_WINDOW]
    if len(_chat_timestamps) >= CHAT_RATE_LIMIT:
        return JSONResponse({"error": "rate_limited", "retry_after_sec": round(CHAT_RATE_WINDOW - (now - _chat_timestamps[0]), 1)}, status_code=429)
    _chat_timestamps.append(now)
    message = req.message.strip()[:400]
    name = req.name.strip()[:32] or "visitor"
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)
    try:
        response = await _chat_response(message, name)
    except Exception as exc:
        log.warning("\U0001f4ac [CHAT] Error generating response: %s", _exc_summary(exc))
        return JSONResponse({"error": "chat unavailable"}, status_code=503)
    p = PERSONALITIES[state.personality_key]
    log.info("\U0001f4ac [CHAT] %s: '%s' \u2192 %.80s", name, message, response)
    broadcast({
        "type": "chat",
        "from": name,
        "message": message,
        "response": response,
        "personality": state.personality_key,
        "p_emoji": p["emoji"],
        "p_name": p["name"],
        "ts": datetime.now().strftime("%H:%M:%S"),
    })
    return JSONResponse({"response": response, "personality": state.personality_key})
