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

import google.generativeai as genai
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
MAX_LOG_SIZE   = 20   # number of entries kept in the event log

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
    "\n\nIMPORTANT: Respond with ONLY valid JSON — no markdown fences, no prose before or after:\n"
    '{"action":"forward|backward|left|right|stop","duration_ms":400,"thought":"your narration here"}'
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
    event_log:        list  = field(default_factory=list)
    is_running:       bool  = False


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


# ─── Gemini ──────────────────────────────────────────────────────────────────
def _build_model(personality_key: str) -> genai.GenerativeModel:
    p      = PERSONALITIES[personality_key]
    system = p["system_prompt"] + RESPONSE_SCHEMA
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system,
    )


async def ask_vlm(
    model:        genai.GenerativeModel,
    jpeg_bytes:   bytes,
    distance_cm:  float,
) -> dict:
    """
    Send the current frame + sensor reading to Gemini.
    Returns {"action": str, "duration_ms": int, "thought": str}.
    """
    img    = Image.open(io.BytesIO(jpeg_bytes))
    prompt = (
        f"Distance sensor reading: {distance_cm:.1f} cm. "
        "Reply with ONLY the JSON as specified."
    )
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: model.generate_content([img, prompt]),
    )

    raw = resp.text.strip()

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
    POST /move with a middleware-side safety guard.
    Returns (action_sent, safety_fired).
    """
    safety_fired = False
    if distance_cm < 10.0 and action == "forward":
        log.warning(
            "⚠️  [SAFETY] Hard stop triggered! distance=%.1f cm (threshold=10 cm)", distance_cm
        )
        state.hard_stop_count += 1
        action       = "stop"
        safety_fired = True

    payload = {"action": action, "duration_ms": duration_ms}
    await client.post(f"http://{WHEELSON_IP}/move", json=payload, timeout=15.0)
    return action, safety_fired


# ─── Explorer Loop ───────────────────────────────────────────────────────────
async def explorer_loop() -> None:
    state.is_running = True
    genai.configure(api_key=GEMINI_API_KEY)
    model       = _build_model(state.personality_key)
    personality = PERSONALITIES[state.personality_key]

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
                state.last_frame_b64   = base64.b64encode(jpeg_bytes).decode()
                state.last_distance_cm = distance_cm

                # 2. Ask the VLM
                vlm = await ask_vlm(model, jpeg_bytes, distance_cm)
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
                state.last_action      = actual_action
                state.last_duration_ms = vlm["duration_ms"]
                state.cycle_count     += 1

                # 4. Append to rolling event log
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
            await asyncio.sleep(max(0.0, LOOP_INTERVAL - elapsed))


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
    parser.add_argument("--port", type=int, default=PORT, help="Dashboard port (default: 8000)")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        log.error("❌  GEMINI_API_KEY is not set. Add it to your .env file and retry.")
        sys.exit(1)

    state.personality_key = args.personality
    p = PERSONALITIES[args.personality]
    log.info("═" * 55)
    log.info("  Wheelson Autonomous Explorer")
    log.info("  Personality : %s  %s", p["emoji"], p["name"])
    log.info("  Move bias   : %s",  p["move_bias"])
    log.info("  Wheelson IP : %s",  WHEELSON_IP)
    log.info("  Dashboard   : http://localhost:%d", args.port)
    log.info("═" * 55)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
