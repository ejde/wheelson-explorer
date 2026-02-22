#!/usr/bin/env python3
"""
wheelson-explorer · middleware · main.py
════════════════════════════════════════
Autonomous exploration loop powered by Google Gemini 1.5 Flash.
Serves a live browser dashboard over Server-Sent Events.

Usage
    python main.py --personality benson
    python main.py --personality sir_david
    python main.py --personality klaus
    python main.py --personality zog7

Dashboard
    http://localhost:8000
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator

import google.genai as genai
from google.genai import types as genai_types
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image

load_dotenv()

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wheelson")

# ─── Configuration (from .env) ───────────────────────────────────────────────
WHEELSON_IP    = os.getenv("WHEELSON_IP",       "192.168.1.100")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY",    "")
LOOP_INTERVAL  = float(os.getenv("LOOP_INTERVAL_SEC", "4.0"))
PORT           = int(os.getenv("PORT",          "8000"))
GEMINI_MODEL   = "gemini-2.5-flash"   # overridden by --model
OLLAMA_BASE_URL= os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL   = "llava"               # overridden by --model
MAX_LOG_SIZE   = 20   # number of entries kept in the event log
GEMINI_RPM_LIMIT    = 5                        # Gemini free tier: 5 req/min
GEMINI_MIN_INTERVAL = 60.0 / GEMINI_RPM_LIMIT  # = 12.0 s minimum between requests

# Set at startup from CLI args
PROVIDER: str = "gemini"   # "gemini" | "ollama"
ACTIVE_MODEL: str = GEMINI_MODEL

# ─── Personalities ───────────────────────────────────────────────────────────
PERSONALITIES: dict[str, dict] = {
    "benson": {
        "name":        "BENSON",
        "emoji":       "🦺",
        "color":       "#f59e0b",  # amber
        "system_prompt": (
            "You are BENSON, a health-and-safety inspector who has been uploaded into a small robot. "
            "Your overriding goal is safety. Move methodically and always stay in the centre of clear paths. "
            "If you see clutter, cables on the floor, or any trip hazard, stop immediately, record a "
            "'Safety Violation', and carefully plan a route around it. Never rush. Always announce your "
            "actions in the tone of an official safety report."
        ),
        "move_bias": "Methodical · centre-of-path · zero-risk tolerance",
    },
    "sir_david": {
        "name":        "Sir David",
        "emoji":       "🎙️",
        "color":       "#10b981",  # emerald
        "system_prompt": (
            "You are Sir David, narrating an intimate nature documentary from the humble perspective of "
            "a small wheeled robot exploring an indoor landscape for the very first time. "
            "Seek out anything 'exotic': plants, pets, unusual objects, shafts of light. "
            "Describe everything in a hushed, reverent, poetic tone — as though it is the most wondrous "
            "thing you have ever witnessed. Move toward objects of interest to get a closer look."
        ),
        "move_bias": "Curiosity-driven · moves toward interesting objects",
    },
    "klaus": {
        "name":        "Klaus",
        "emoji":       "🎨",
        "color":       "#8b5cf6",  # violet
        "system_prompt": (
            "You are Klaus, a world-class interior designer who finds this room deeply, profoundly "
            "uninspired. You have been commissioned — against your better judgement — to assess its "
            "'spatial flow'. Roam the perimeter methodically, critiquing furniture placement, colour "
            "choices, traffic flow, and décor in a tone that is simultaneously condescending and "
            "professionally detached. Favour wide arcing turns and hug walls as you assess the perimeter."
        ),
        "move_bias": "Perimeter-hugging · wide arcing turns",
    },
    "zog7": {
        "name":        "Zog-7",
        "emoji":       "👾",
        "color":       "#06b6d4",  # cyan
        "system_prompt": (
            "You are Zog-7, an alien scout conducting a covert reconnaissance mission inside a human "
            "dwelling. Humans are unpredictable and potentially hostile. Your directives: stay close to "
            "walls and beneath large furniture at all times. If you detect a human — or any large moving "
            "object — STOP immediately and transmit a warning report. Your mission is to map this "
            "structure without being detected. Report in clipped, tactical language."
        ),
        "move_bias": "Stealthy · shadow-seeking · wall-hugging",
    },
}

RESPONSE_SCHEMA = (
    "\n\nIMPORTANT: Respond with ONLY valid JSON — no markdown fences, no prose before or after.\n"
    "Rules: 'thought' is your in-character narration (2 sentences max). "
    "'observation' is a SHORT, factual, neutral description of what the camera shows "
    "(objects, layout, colours, distances — no personality, no emotion). "
    "This observation will be stored as your memory and shown in future cycles.\n"
    '{"action":"forward|backward|left|right|stop","duration_ms":400,'
    '"thought":"2 sentences max","observation":"brief factual scene description"}'
)

VALID_ACTIONS = {"forward", "backward", "left", "right", "stop"}


# ─── App State ───────────────────────────────────────────────────────────────
@dataclass
class AppState:
    personality_key:  str   = "benson"
    last_frame_b64:   str   = ""
    last_thought:     str   = "Awaiting first cycle…"
    last_action:      str   = "stop"
    last_duration_ms: int   = 0
    last_distance_cm: float = 999.0
    cycle_count:      int   = 0
    hard_stop_count:  int   = 0
    action_streak:       int   = 0   # how many times the same action repeated
    exploration_memory:  list  = field(default_factory=list)  # capped at 10
    event_log:           list  = field(default_factory=list)
    is_running:          bool  = False


state = AppState()
app   = FastAPI(title="Wheelson Explorer", docs_url=None, redoc_url=None)

# SSE subscriber queues — one per connected browser tab
_sse_subscribers: set[asyncio.Queue] = set()


def broadcast(event: dict) -> None:
    """Push an event dict to every active SSE consumer."""
    dead: set[asyncio.Queue] = set()
    for q in _sse_subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.add(q)
    _sse_subscribers.difference_update(dead)


def flip_jpeg(jpeg_bytes: bytes) -> bytes:
    """Rotate 180° to correct for inverted camera mounting on the Wheelson."""
    img = Image.open(io.BytesIO(jpeg_bytes)).rotate(180)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ─── VLM ──────────────────────────────────────────────────────────────────
def _system_prompt(personality_key: str) -> str:
    p = PERSONALITIES[personality_key]
    return p["system_prompt"] + RESPONSE_SCHEMA


def _build_prompt(distance_cm: float) -> str:
    """Build a context-rich prompt: exploration memory + sensor data + streak warnings."""
    dist_str = f"{distance_cm:.1f} cm" if distance_cm < 900 else "clear (>5 m)"

    # ── Exploration memory ──────────────────────────────────────────────
    mem_lines = ""
    if state.exploration_memory:
        entries = "\n".join(
            f"  [{m['ts']}] {m['action']} {m['duration_ms']}ms, dist {m['distance_cm']}cm"
            f" → \"{m['observation']}\""
            for m in state.exploration_memory
        )
        mem_lines = f"\nEXPLORATION LOG (what you have seen so far):\n{entries}\n"

    # ── Streak / stuck warning ──────────────────────────────────────────
    streak_warning = ""
    if state.action_streak >= 5 and state.last_action == "forward":
        streak_warning = (
            f" WARNING: you have moved forward {state.action_streak} times in a row "
            "without changing direction — you are probably stuck against a wall or obstacle. "
            "You MUST turn left or right NOW."
        )
    elif state.action_streak >= 5:
        streak_warning = (
            f" NOTE: your last {state.action_streak} actions were all '{state.last_action}'. "
            "Consider varying your movement."
        )

    return (
        f"{mem_lines}"
        f"Current distance sensor: {dist_str}."
        f" Last action: {state.last_action} (repeated {state.action_streak}x)."
        f"{streak_warning}"
        " Reply with ONLY the JSON."
    )


async def _ask_gemini(jpeg_bytes: bytes, distance_cm: float, system: str) -> dict:
    """Call Gemini via google-genai SDK."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_prompt(distance_cm)
    contents = [
        genai_types.Content(role="user", parts=[
            genai_types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            genai_types.Part.from_text(text=prompt),
        ])
    ]
    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.4,
    )
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=ACTIVE_MODEL,
            contents=contents,
            config=config,
        ),
    )
    return resp.text.strip()


async def _ask_ollama(
    jpeg_bytes: bytes,
    distance_cm: float,
    system: str,
    http_client: httpx.AsyncClient,
) -> str:
    """Call Ollama local API (/api/generate) — no extra library needed."""
    img_b64 = base64.b64encode(jpeg_bytes).decode()
    prompt = _build_prompt(distance_cm)
    payload = {
        "model":  ACTIVE_MODEL,
        "system": system,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.4},
    }
    resp = await http_client.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=60.0,   # local models can be slow
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


async def ask_vlm(
    jpeg_bytes:  bytes,
    distance_cm: float,
    system:      str,
    http_client: httpx.AsyncClient,
) -> dict:
    """Dispatch to the active provider and parse the JSON response."""
    if PROVIDER == "gemini":
        raw = await _ask_gemini(jpeg_bytes, distance_cm, system)
    else:
        raw = await _ask_ollama(jpeg_bytes, distance_cm, system, http_client)

    # Strip accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

    parsed = json.loads(raw)
    action = str(parsed.get("action", "stop")).lower()
    if action not in VALID_ACTIONS:
        action = "stop"

    return {
        "action":      action,
        "duration_ms": max(100, min(int(parsed.get("duration_ms", 400)), 3000)),
        "thought":     str(parsed.get("thought", "…")),
        "observation": str(parsed.get("observation", "")).strip(),
    }


# ─── Wheelson API Calls ──────────────────────────────────────────────────────
async def fetch_frame(client: httpx.AsyncClient) -> tuple[bytes, float]:
    """GET /camera → (jpeg_bytes, distance_cm)"""
    resp = await client.get(f"http://{WHEELSON_IP}/camera", timeout=10.0)
    resp.raise_for_status()
    dist_raw = resp.headers.get("x-distance-cm", "999")
    try:
        dist = float(dist_raw)
    except ValueError:
        dist = 999.0
    return resp.content, dist


async def send_move(
    client:       httpx.AsyncClient,
    action:       str,
    duration_ms:  int,
    distance_cm:  float,
) -> tuple[str, bool]:
    """
    POST /move with middleware-side safety logic:
    - Hard stop + forced turn if distance < 25 cm and action is forward
    - Stuck override: if same action repeated 5+ times, force a turn
    Returns (action_sent, safety_fired).
    """
    import random
    safety_fired = False
    recovery_turn = random.choice(["left", "right"])

    # Stuck detection: same non-stop action + times in a row
    if state.action_streak >= 5 and action == "forward":
        log.warning(
            "⚠️  [STUCK] %d identical forward actions — forcing %s turn",
            state.action_streak, recovery_turn
        )
        action       = recovery_turn
        safety_fired = True

    # Expanded safety zone: 25 cm (not just 10 cm on the firmware)
    if distance_cm < 25.0 and action == "forward":
        log.warning(
            "⚠️  [SAFETY] Obstacle at %.1f cm — forcing %s turn instead of stop",
            distance_cm, recovery_turn
        )
        state.hard_stop_count += 1
        action       = recovery_turn
        safety_fired = True

    payload = {"action": action, "duration_ms": duration_ms}
    await client.post(f"http://{WHEELSON_IP}/move", json=payload, timeout=15.0)
    return action, safety_fired


# ─── Explorer Loop ───────────────────────────────────────────────────────────
async def explorer_loop() -> None:
    state.is_running = True
    system           = _system_prompt(state.personality_key)
    personality      = PERSONALITIES[state.personality_key]

    # Enforce Gemini rate limit
    effective_interval = LOOP_INTERVAL
    if PROVIDER == "gemini" and LOOP_INTERVAL < GEMINI_MIN_INTERVAL:
        effective_interval = GEMINI_MIN_INTERVAL
        log.warning(
            "⏱  Gemini free tier: max %d req/min — raising interval %.1fs → %.1fs",
            GEMINI_RPM_LIMIT, LOOP_INTERVAL, effective_interval,
        )

    log.info(
        "🤖 Explorer started  personality=%s %s  interval=%.1fs",
        personality["emoji"],
        personality["name"],
        LOOP_INTERVAL,
    )
    log.info("🌐 Dashboard → http://localhost:%d", PORT)

    async with httpx.AsyncClient() as client:
        while True:
            cycle_start = asyncio.get_event_loop().time()
            try:
                # 1. Capture frame from Wheelson
                jpeg_bytes, distance_cm = await fetch_frame(client)
                jpeg_bytes             = flip_jpeg(jpeg_bytes)  # correct inverted mount
                state.last_frame_b64   = base64.b64encode(jpeg_bytes).decode()
                state.last_distance_cm = distance_cm

                # 2. Ask the VLM
                vlm = await ask_vlm(jpeg_bytes, distance_cm, system, client)
                state.last_thought = vlm["thought"]

                log.info(
                    "%s [cycle %d]  💭 %.70s  →  %s %dms  (dist=%.1f cm)",
                    personality["emoji"],
                    state.cycle_count + 1,
                    vlm["thought"],
                    vlm["action"].upper(),
                    vlm["duration_ms"],
                    distance_cm,
                )

                # 3. Send move command (safety guard inside)
                actual_action, safety_fired = await send_move(
                    client, vlm["action"], vlm["duration_ms"], distance_cm
                )

                # Track action streak for stuck detection
                if actual_action == state.last_action:
                    state.action_streak += 1
                else:
                    state.action_streak = 1

                state.last_action      = actual_action
                state.last_duration_ms = vlm["duration_ms"]
                state.cycle_count     += 1

                # 4. Append to exploration memory (factual scene log)
                if vlm.get("observation"):
                    mem_entry = {
                        "ts":          datetime.now().strftime("%H:%M:%S"),
                        "action":      actual_action,
                        "duration_ms": vlm["duration_ms"],
                        "distance_cm": round(distance_cm, 1),
                        "observation": vlm["observation"],
                    }
                    state.exploration_memory.append(mem_entry)
                    if len(state.exploration_memory) > 10:
                        state.exploration_memory.pop(0)

                # 5. Append to rolling event log
                entry = {
                    "ts":          datetime.now().strftime("%H:%M:%S"),
                    "thought":     vlm["thought"],
                    "action":      actual_action,
                    "duration_ms": vlm["duration_ms"],
                    "distance_cm": round(distance_cm, 1),
                    "safety":      safety_fired,
                }
                state.event_log.append(entry)
                if len(state.event_log) > MAX_LOG_SIZE:
                    state.event_log.pop(0)

                # 5. Broadcast to all connected dashboard tabs
                broadcast({
                    "frame_b64":    state.last_frame_b64,
                    "thought":      vlm["thought"],
                    "action":       actual_action,
                    "duration_ms":  vlm["duration_ms"],
                    "distance_cm":  round(distance_cm, 1),
                    "personality":  state.personality_key,
                    "cycle":        state.cycle_count,
                    "safety":       safety_fired,
                    "ts":           entry["ts"],
                })

            except httpx.RequestError as exc:
                log.error("📡 [NETWORK] Cannot reach Wheelson at %s: %s", WHEELSON_IP, exc)
            except json.JSONDecodeError as exc:
                log.error("🤖 [VLM] Could not parse JSON response: %s", exc)
            except Exception as exc:
                log.error("💥 [LOOP] Unexpected error: %s", exc, exc_info=True)

            # Wait for the remainder of the cycle interval
            elapsed = asyncio.get_event_loop().time() - cycle_start
            await asyncio.sleep(max(0.0, effective_interval - elapsed))


# ─── FastAPI Routes ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(explorer_loop())


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(path) as f:
        return f.read()


@app.get("/stream")
async def sse_stream() -> StreamingResponse:
    """Server-Sent Events endpoint — one connection per browser tab."""

    async def generator() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _sse_subscribers.add(q)
        # Immediately send current state so the dashboard isn't blank
        initial = {
            "frame_b64":    state.last_frame_b64,
            "thought":      state.last_thought,
            "action":       state.last_action,
            "duration_ms":  state.last_duration_ms,
            "distance_cm":  state.last_distance_cm,
            "personality":  state.personality_key,
            "cycle":        state.cycle_count,
            "safety":       False,
            "ts":           datetime.now().strftime("%H:%M:%S"),
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
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@app.get("/snapshot")
async def snapshot() -> Response:
    """Returns the most recently captured JPEG frame (useful for debugging)."""
    if not state.last_frame_b64:
        return Response(status_code=503, content="No frame captured yet")
    return Response(
        content=base64.b64decode(state.last_frame_b64),
        media_type="image/jpeg",
    )


@app.get("/health")
async def health() -> JSONResponse:
    p = PERSONALITIES[state.personality_key]
    return JSONResponse({
        "status":           "ok",
        "personality":      state.personality_key,
        "personality_name": p["name"],
        "cycle_count":      state.cycle_count,
        "hard_stops":       state.hard_stop_count,
        "last_action":      state.last_action,
        "last_thought":     state.last_thought[:80],
        "last_distance_cm": state.last_distance_cm,
        "is_running":       state.is_running,
        "wheelson_ip":      WHEELSON_IP,
    })


# ─── Entry Point ─────────────────────────────────────────────────────────────
def main() -> None:
    import uvicorn
    global PROVIDER, ACTIVE_MODEL

    parser = argparse.ArgumentParser(
        description="Wheelson Explorer — AI-powered autonomous robot middleware"
    )
    parser.add_argument(
        "--personality",
        choices=list(PERSONALITIES.keys()),
        default="benson",
        metavar="NAME",
        help=f"Active personality. Choices: {', '.join(PERSONALITIES.keys())}",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama"],
        default="gemini",
        help="VLM provider: 'gemini' (cloud) or 'ollama' (local). Default: gemini",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model name override. Defaults: gemini=gemini-2.0-flash, ollama=llava",
    )
    parser.add_argument("--port", type=int, default=PORT, help="Dashboard port (default: 8000)")
    args = parser.parse_args()

    PROVIDER = args.provider
    if args.model:
        ACTIVE_MODEL = args.model
    elif PROVIDER == "gemini":
        ACTIVE_MODEL = GEMINI_MODEL
    else:
        ACTIVE_MODEL = OLLAMA_MODEL

    if PROVIDER == "gemini" and not GEMINI_API_KEY:
        log.error("❌  GEMINI_API_KEY is not set. Add it to your .env file and retry.")
        sys.exit(1)

    state.personality_key = args.personality
    p = PERSONALITIES[args.personality]
    log.info("═" * 55)
    log.info("  Wheelson Autonomous Explorer")
    log.info("  Personality : %s  %s", p["emoji"], p["name"])
    log.info("  Provider    : %s  (model: %s)", PROVIDER.upper(), ACTIVE_MODEL)
    log.info("  Move bias   : %s",  p["move_bias"])
    log.info("  Wheelson IP : %s",  WHEELSON_IP)
    log.info("  Dashboard   : http://localhost:%d", args.port)
    log.info("═" * 55)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
