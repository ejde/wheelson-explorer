#!/usr/bin/env python3
"""
wheelson-explorer · middleware · main.py
════════════════════════════════════════
Autonomous exploration loop powered by Google Gemini / Ollama.
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
import contextlib
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Literal

import google.genai as genai
from google.genai import types as genai_types
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wheelson")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WHEELSON_IP = os.getenv("WHEELSON_IP", "192.168.1.100")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LOOP_INTERVAL = float(os.getenv("LOOP_INTERVAL_SEC", "4.0"))
PORT = int(os.getenv("PORT", "8000"))
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llava"
MAX_LOG_SIZE = 20

GEMINI_RPM_LIMIT = 5
GEMINI_MIN_INTERVAL = 60.0 / GEMINI_RPM_LIMIT
OLLAMA_STRATEGY_MIN_INTERVAL = float(os.getenv("OLLAMA_STRATEGY_MIN_INTERVAL_SEC", "8.0"))
OLLAMA_VLM_TIMEOUT_SEC = float(os.getenv("OLLAMA_VLM_TIMEOUT_SEC", "14.0"))
OLLAMA_TIMEOUT_BACKOFF_SEC = float(os.getenv("OLLAMA_TIMEOUT_BACKOFF_SEC", "24.0"))
STRATEGY_STALE_MULTIPLIER = 2.0

LEASE_HEARTBEAT_SEC = 0.4
LEASE_MISSED_LIMIT = 3

VISUAL_STALL_SCORE_MAX = 3.0
MOTION_RECOVERY_SCORE = 6.0
RECOVERY_NUDGE_STREAK = 2
RECOVERY_ESCAPE_STREAK = 4
RECOVERY_HARD_STREAK = 6
RECOVERY_COOLDOWN_NUDGE = 1
RECOVERY_COOLDOWN_ESCAPE = 2
RECOVERY_COOLDOWN_HARD = 3
FORWARD_BURST_DEFAULT_MS = int(os.getenv("FORWARD_BURST_DEFAULT_MS", "850"))
KLAUS_ARC_INTERVAL = int(os.getenv("KLAUS_ARC_INTERVAL", "3"))
KLAUS_ARC_TURN_MS = int(os.getenv("KLAUS_ARC_TURN_MS", "760"))
TIMEOUT_SCAN_STEPS = int(os.getenv("TIMEOUT_SCAN_STEPS", "2"))
TIMEOUT_SCAN_TURN_MS = int(os.getenv("TIMEOUT_SCAN_TURN_MS", "700"))

TURN_SIDE_MARGIN = 0.08
SIDE_STEER_RATIO_THRESHOLD = 0.22
SIDE_STEER_MARGIN = 0.10
SIDE_STEER_TURN_MS = 420
SCENE_HARD_CLEAR_THRESHOLD = 0.30
SCENE_SOFT_CLEAR_THRESHOLD = 0.22
SCENE_FORWARD_CLEAR_THRESHOLD = 0.16
SCENE_HOLD_RESET_THRESHOLD = 0.12

TURN_DURATION_MS = {
    "benson": 600,
    "sir_david": 500,
    "klaus": 700,
    "zog7": 420,
}

ESCAPE_BACKWARD_MS = 700
ESCAPE_TURN_MS = 1000
HARD_ESCAPE_BACKUP_MS = 1400
HARD_ESCAPE_TURN_MS = 1200
STARTUP_SLOW_CYCLES = 12

MOVE_ACTIONS = {"forward", "backward", "left", "right"}
ALL_ACTIONS = MOVE_ACTIONS | {"stop"}

# Set at startup from CLI args
PROVIDER: str = "gemini"  # "gemini" | "ollama"
ACTIVE_MODEL: str = GEMINI_MODEL

# ---------------------------------------------------------------------------
# Personalities
# ---------------------------------------------------------------------------
PERSONALITIES: dict[str, dict] = {
    "benson": {
        "name": "BENSON",
        "emoji": "🦺",
        "color": "#f59e0b",
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
        "name": "Sir David",
        "emoji": "🎙️",
        "color": "#10b981",
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
        "name": "Klaus",
        "emoji": "🎨",
        "color": "#8b5cf6",
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
        "name": "Zog-7",
        "emoji": "👾",
        "color": "#06b6d4",
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
    "You are a SCENE INTERPRETER, not a motion planner.\n"
    "'observation' must be a SHORT factual description of what the camera shows "
    "(objects/layout/distances/lighting; no personality roleplay).\n"
    "Set 'frontier' to one of: forward, left, right, blocked, unknown.\n"
    "Set 'traversability' to one of: low, medium, high.\n"
    "Set 'novelty' to one of: low, medium, high (how interesting/new the scene appears).\n"
    "Set 'hazard' to one of: none, soft, hard.\n"
    "Set 'headlight' to true if additional light would improve scene visibility.\n"
    '{"observation":"A narrow hallway with clear floor and open space ahead.",'
    '"frontier":"forward","traversability":"high","novelty":"medium","hazard":"none","headlight":false}'
)
SCENE_INTERPRETER_PROMPT = (
    "You are the perception layer for a mobile robot. "
    "Your job is to describe the scene and estimate navigability signals only. "
    "Do not roleplay a persona and do not output motor actions."
)

PERSONA_POLICIES: dict[str, dict] = {
    "benson": {
        "base_speed": "slow",
        "safety_weight": 1.7,
        "curiosity_weight": 0.5,
        "perimeter_weight": 0.4,
        "preferred_wall": "none",
        "forward_burst_ms": 700,
        "hold_bias": 0.8,
    },
    "sir_david": {
        "base_speed": "medium",
        "safety_weight": 1.0,
        "curiosity_weight": 1.6,
        "perimeter_weight": 0.2,
        "preferred_wall": "none",
        "forward_burst_ms": 900,
        "hold_bias": 0.2,
    },
    "klaus": {
        "base_speed": "medium",
        "safety_weight": 1.2,
        "curiosity_weight": 0.8,
        "perimeter_weight": 1.5,
        "preferred_wall": "left",
        "forward_burst_ms": 800,
        "hold_bias": 0.4,
    },
    "zog7": {
        "base_speed": "fast",
        "safety_weight": 1.1,
        "curiosity_weight": 1.2,
        "perimeter_weight": 1.7,
        "preferred_wall": "right",
        "forward_burst_ms": 700,
        "hold_bias": 0.1,
    },
}


class VLMResponseError(Exception):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def _exc_summary(exc: Exception, max_len: int = 180) -> str:
    text = str(exc).replace("\n", " ").strip()
    if not text:
        text = exc.__class__.__name__
    return text[:max_len]


def _is_vlm_quota_error(exc: Exception) -> bool:
    text = _exc_summary(exc, max_len=500).lower()
    tokens = (
        "429",
        "resource_exhausted",
        "rate limit",
        "ratelimit",
        "quota",
        "too many requests",
    )
    return any(token in text for token in tokens)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    personality_key: str = "benson"
    last_frame_b64: str = ""
    last_thought: str = "Awaiting first cycle…"
    last_objective: str = "Awaiting first cycle…"
    last_action: str = "stop"
    last_duration_ms: int = 0
    last_distance_cm: float = 999.0
    last_headlight: bool = False
    cycle_count: int = 0
    hard_stop_count: int = 0
    action_streak: int = 0
    last_nav_state: str = "HOLD"

    # Motion/progress tracking
    last_motion_score: float = -1.0
    prev_frame_signature: bytes = b""
    visual_stall_count: int = 0
    visual_stall_samples: int = 0
    no_progress_count: int = 0
    no_progress_samples: int = 0

    # Strategy tracking
    last_strategy_source: str = "bootstrap"
    last_strategy_update_ts: float = 0.0
    last_vlm_headlight_pref: bool = False
    last_scene_frontier: str = "unknown"
    last_scene_traversability: str = "medium"
    last_scene_novelty: str = "medium"
    last_scene_hazard: str = "none"
    last_scene_observation: str = ""

    # Command authority telemetry
    last_command_id: str = "boot"
    last_command_source: str = "bootstrap"
    last_command_mode: str = "idle"

    # Recovery telemetry
    recovery_level: int = 0
    recovery_cooldown_cycles: int = 0
    low_motion_streak: int = 0

    speed_cap_level: str = "none"
    speed_cap_reason: str = ""

    exploration_memory: list = field(default_factory=list)
    event_log: list = field(default_factory=list)
    is_running: bool = False


state = AppState()
app = FastAPI(title="Wheelson Explorer", docs_url=None, redoc_url=None)
_sse_subscribers: set[asyncio.Queue] = set()


def broadcast(event: dict) -> None:
    dead: set[asyncio.Queue] = set()
    for q in _sse_subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.add(q)
    _sse_subscribers.difference_update(dead)


def flip_jpeg(jpeg_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(jpeg_bytes)).rotate(180)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _frame_signature(jpeg_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("L").resize((32, 24))
    return bytes(img.tobytes())


def _motion_score(prev_sig: bytes, curr_sig: bytes) -> float:
    if not prev_sig or len(prev_sig) != len(curr_sig):
        return -1.0
    prev_mean = sum(prev_sig) / len(prev_sig)
    curr_mean = sum(curr_sig) / len(curr_sig)
    return sum(abs((a - prev_mean) - (b - curr_mean)) for a, b in zip(prev_sig, curr_sig)) / len(curr_sig)


def _is_placeholder_text(value: str) -> bool:
    v = value.strip().lower()
    if not v:
        return True
    placeholders = {
        "...",
        "…",
        "n/a",
        "none",
        "1 sentence max",
        "brief factual description",
        "short factual description",
        "awaiting first cycle…",
        "awaiting first cycle...",
    }
    return v in placeholders

def _persona_reactive_thought(personality_key: str, mode: str) -> str:
    if personality_key == "benson":
        if mode == "nudge":
            return "Safety intervention: issuing a corrective turn to break emerging stagnation."
        if mode == "escape":
            return "Safety intervention: controlled reverse-and-turn recovery in progress."
        return "Safety intervention: full hard-escape maneuver initiated."
    if personality_key == "sir_david":
        if mode == "nudge":
            return "A subtle course correction to restore graceful forward momentum."
        if mode == "escape":
            return "Progress remains constrained, so I withdraw and arc toward a clearer route."
        return "The path is fully compromised; executing a decisive retreat and redirect."
    if personality_key == "klaus":
        if mode == "nudge":
            return "Minor trajectory correction to recover acceptable spatial flow."
        if mode == "escape":
            return "This lane is unusable; reversing and reframing the perimeter line."
        return "Severe circulation failure detected; executing hard reposition."
    if personality_key == "zog7":
        if mode == "nudge":
            return "Micro-adjustment applied to preserve covert mobility."
        if mode == "escape":
            return "Mobility degraded. Executing tactical reverse-turn recovery."
        return "Critical mobility fault. Hard escape protocol active."
    if mode == "nudge":
        return "Low motion detected; applying corrective turn."
    if mode == "escape":
        return "No progress state detected; executing escape maneuver."
    return "Persistent no-progress state; executing hard escape."


def _persona_timeout_scan_thought(personality_key: str) -> str:
    if personality_key == "benson":
        return "Strategist link degraded; pausing and scanning before the next movement."
    if personality_key == "sir_david":
        return "The signal fades, so I pause and gently pan to reassess the terrain."
    if personality_key == "klaus":
        return "Strategist latency is unacceptable; conducting a deliberate perimeter scan."
    if personality_key == "zog7":
        return "Uplink unstable. Holding and sweeping for a safer vector."
    return "Strategist unavailable; holding and scanning locally."


def _extract_first_json_object(raw: str) -> str | None:
    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return None


def _normalize_vlm_json(raw: str) -> str:
    text = raw.strip()

    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if not candidate:
                continue
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                text = candidate
                break

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    candidate = _extract_first_json_object(text)
    if candidate:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    candidate = _extract_first_json_object(cleaned)
    if candidate:
        return candidate
    return text


# ---------------------------------------------------------------------------
# VLM integration
# ---------------------------------------------------------------------------
def _system_prompt(personality_key: str) -> str:
    _ = personality_key
    return SCENE_INTERPRETER_PROMPT + RESPONSE_SCHEMA


def _build_prompt(telemetry: dict) -> str:
    mem_lines = ""
    if state.exploration_memory:
        entries = "\n".join(
            f"  [{m['ts']}] {m['action']} {m['duration_ms']}ms, dist {m['distance_cm']}cm"
            f" -> \"{m['observation']}\""
            for m in state.exploration_memory
        )
        mem_lines = f"\nEXPLORATION LOG (what you have seen so far):\n{entries}\n"

    streak_warning = ""
    if state.action_streak >= 5 and state.last_action == "forward":
        streak_warning = (
            f" NOTE: robot has attempted forward {state.action_streak} times with poor progress. "
            "Scene interpretation should emphasize viable side frontiers and hazards."
        )

    obs_warn = ""
    if telemetry.get("visual_obstacle", False):
        obs_warn = " PRE-EMPTIVE WARNING: visual obstacle telemetry is true near the forward lane."

    return (
        f"{mem_lines}"
        " Current system telemetry: Ultrasonic distance=unavailable (sensor hardware offline)."
        f" Lighting={telemetry.get('brightness', 'Unknown')}."
        f" Dominant Floor Color={telemetry.get('dominant_color', 'Unknown')}."
        f" Firmware nav state: {telemetry.get('nav_state', 'UNKNOWN')}."
        f" Side obstacle ratios: left={telemetry.get('obstacle_left_ratio', 0.0):.2f}, right={telemetry.get('obstacle_right_ratio', 0.0):.2f}."
        f"{obs_warn}"
        f" Last planner objective: {state.last_objective}."
        f" Last action: {state.last_action} (repeated {state.action_streak}x)."
        f"{streak_warning}"
        " Reply with ONLY the JSON."
    )


async def _ask_gemini(jpeg_bytes: bytes, telemetry: dict, system: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_prompt(telemetry)
    contents = [
        genai_types.Content(
            role="user",
            parts=[
                genai_types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                genai_types.Part.from_text(text=prompt),
            ],
        )
    ]
    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.4,
    )
    loop = asyncio.get_running_loop()
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
    telemetry: dict,
    system: str,
    http_client: httpx.AsyncClient,
) -> str:
    img_b64 = base64.b64encode(jpeg_bytes).decode()
    prompt = _build_prompt(telemetry)
    payload = {
        "model": ACTIVE_MODEL,
        "system": system,
        "prompt": prompt,
        "images": [img_b64],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.4},
    }
    resp = await http_client.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


async def ask_vlm(
    jpeg_bytes: bytes,
    telemetry: dict,
    system: str,
    http_client: httpx.AsyncClient,
) -> dict:
    if PROVIDER == "gemini":
        raw = await _ask_gemini(jpeg_bytes, telemetry, system)
    else:
        raw = await _ask_ollama(jpeg_bytes, telemetry, system, http_client)

    raw = _normalize_vlm_json(raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VLMResponseError(f"Could not parse model JSON: {exc}", raw) from exc

    observation = str(parsed.get("observation", "")).strip()
    if _is_placeholder_text(observation):
        observation = ""

    frontier = str(parsed.get("frontier", "")).strip().lower()
    traversability = str(parsed.get("traversability", "")).strip().lower()
    novelty = str(parsed.get("novelty", "")).strip().lower()
    hazard = str(parsed.get("hazard", "")).strip().lower()

    return {
        "observation": observation,
        "headlight": _as_bool(parsed.get("headlight", False)),
        "frontier": frontier,
        "traversability": traversability,
        "novelty": novelty,
        "hazard": hazard,
    }


# ---------------------------------------------------------------------------
# Scene semantics + strategy intent
# ---------------------------------------------------------------------------
SceneFrontier = Literal["forward", "left", "right", "blocked", "unknown"]
SceneLevel = Literal["low", "medium", "high"]
HazardLevel = Literal["none", "soft", "hard"]


@dataclass
class StrategyIntent:
    objective: str
    thought: str
    observation: str
    headlight: bool
    frontier: SceneFrontier
    traversability: SceneLevel
    novelty: SceneLevel
    hazard: HazardLevel
    source: str


def _normalize_frontier(value: str) -> SceneFrontier | None:
    v = value.strip().lower()
    if v in {"forward", "left", "right", "blocked", "unknown"}:
        return v  # type: ignore[return-value]
    return None


def _normalize_scene_level(value: str) -> SceneLevel | None:
    v = value.strip().lower()
    if v in {"low", "medium", "high"}:
        return v  # type: ignore[return-value]
    return None


def _normalize_hazard(value: str) -> HazardLevel | None:
    v = value.strip().lower()
    if v in {"none", "soft", "hard"}:
        return v  # type: ignore[return-value]
    return None


def _infer_frontier(telemetry: dict) -> SceneFrontier:
    if telemetry.get("visual_obstacle", False):
        left_ratio = telemetry.get("obstacle_left_ratio", 0.0)
        right_ratio = telemetry.get("obstacle_right_ratio", 0.0)
        if left_ratio + TURN_SIDE_MARGIN < right_ratio:
            return "left"
        if right_ratio + TURN_SIDE_MARGIN < left_ratio:
            return "right"
        return "blocked"

    left_ratio = telemetry.get("obstacle_left_ratio", 0.0)
    right_ratio = telemetry.get("obstacle_right_ratio", 0.0)
    if left_ratio + TURN_SIDE_MARGIN < right_ratio:
        return "left"
    if right_ratio + TURN_SIDE_MARGIN < left_ratio:
        return "right"
    return "forward"


def _infer_traversability(telemetry: dict) -> SceneLevel:
    if telemetry.get("visual_obstacle", False):
        return "low"
    side_peak = max(
        telemetry.get("obstacle_left_ratio", 0.0),
        telemetry.get("obstacle_right_ratio", 0.0),
    )
    if side_peak >= 0.36:
        return "medium"
    return "high"


def _infer_novelty(telemetry: dict) -> SceneLevel:
    if telemetry.get("brightness") == "Dark":
        return "medium"
    return "medium"


def _infer_hazard(telemetry: dict) -> HazardLevel:
    if telemetry.get("visual_obstacle", False):
        return "hard"
    side_peak = max(
        telemetry.get("obstacle_left_ratio", 0.0),
        telemetry.get("obstacle_right_ratio", 0.0),
    )
    if side_peak >= 0.40:
        return "soft"
    return "none"


def _reconcile_scene_with_telemetry(
    frontier: SceneFrontier,
    traversability: SceneLevel,
    hazard: HazardLevel,
    telemetry: dict,
) -> tuple[SceneFrontier, SceneLevel, HazardLevel]:
    visual_obstacle = bool(telemetry.get("visual_obstacle", False))
    side_left = telemetry.get("obstacle_left_ratio", 0.0)
    side_right = telemetry.get("obstacle_right_ratio", 0.0)
    side_peak = max(side_left, side_right)
    nav_state = str(telemetry.get("nav_state", "UNKNOWN")).upper()

    # Hard evidence from telemetry should quickly clear stale high-risk states.
    if not visual_obstacle:
        if side_peak < SCENE_HARD_CLEAR_THRESHOLD and hazard == "hard":
            hazard = "soft"
        if side_peak < SCENE_SOFT_CLEAR_THRESHOLD and hazard == "soft":
            hazard = "none"

        if side_peak < SCENE_HARD_CLEAR_THRESHOLD and traversability == "low":
            traversability = "medium"
        if side_peak < SCENE_SOFT_CLEAR_THRESHOLD and traversability == "medium":
            traversability = "high"

        if frontier == "blocked" and side_peak < SCENE_HARD_CLEAR_THRESHOLD:
            frontier = "unknown"
        if frontier in {"left", "right", "unknown"} and side_peak < SCENE_FORWARD_CLEAR_THRESHOLD:
            frontier = "forward"

        # If firmware is already in HOLD and side ratios are near-zero, reset to clear-forward.
        if nav_state == "HOLD" and side_peak < SCENE_HOLD_RESET_THRESHOLD:
            frontier = "forward"
            traversability = "high"
            hazard = "none"

    # Conversely, hard visual obstacle should not keep "none" hazard.
    if visual_obstacle and hazard == "none":
        hazard = "soft"
        if traversability == "high":
            traversability = "medium"
        if frontier == "forward":
            frontier = "blocked"
    if visual_obstacle and side_peak >= 0.35 and hazard == "soft":
        hazard = "hard"

    return frontier, traversability, hazard


def _compose_objective(
    personality_key: str,
    frontier: SceneFrontier,
    traversability: SceneLevel,
    novelty: SceneLevel,
    hazard: HazardLevel,
) -> str:
    direction = "left" if frontier == "left" else "right"

    if personality_key == "benson":
        if hazard == "hard":
            return "Issue a safety hold and re-route around the obstruction"
        if hazard == "soft" or traversability == "low":
            return "Reduce risk and verify a compliant path before advancing"
        if frontier in {"left", "right"}:
            return f"Probe the safer {direction} corridor under strict control"
        return "Maintain a methodical safety sweep down the clear lane"

    if personality_key == "sir_david":
        if hazard == "hard":
            return "Pause the approach and drift toward a clearer vantage"
        if novelty == "high":
            return "Approach the most compelling feature for close observation"
        if frontier in {"left", "right"}:
            return f"Arc {direction} to reveal a fresh documentary viewpoint"
        return "Continue a curious roaming pass through the scene"

    if personality_key == "klaus":
        if hazard == "hard":
            return "Hold, then reshape trajectory to preserve spatial flow"
        if frontier in {"left", "right"}:
            return f"Execute a wide {direction} arc and continue perimeter audit"
        return "Sweep the perimeter with deliberate wide arcs and wall discipline"

    if personality_key == "zog7":
        if hazard == "hard":
            return "Break contact and reroute to a safer concealed approach"
        if frontier in {"left", "right"}:
            return f"Shift {direction} to retain edge cover and low exposure"
        return "Map the edge corridor while maintaining covert spacing"

    if hazard == "hard":
        return "Hold and re-route around immediate obstruction"
    if frontier == "blocked":
        return "Search for a safer side corridor"
    if frontier in {"left", "right"}:
        return f"Probe the clearer {direction} frontier"
    if novelty == "high":
        return "Advance and inspect the newly interesting area"
    return "Continue controlled forward exploration"


def _compose_persona_thought(personality_key: str, observation: str, objective: str) -> str:
    fact = observation if observation else objective
    if personality_key == "benson":
        return f"Safety report: {fact}."
    if personality_key == "sir_david":
        return f"From this vantage, {fact.lower()}."
    if personality_key == "klaus":
        return f"Assessment note: {fact.lower()}."
    if personality_key == "zog7":
        return f"Recon update: {fact.lower()}."
    return f"Scene update: {fact}."


def _local_scene_assessment(telemetry: dict) -> dict:
    return {
        "observation": "",
        "headlight": telemetry.get("brightness") == "Dark",
        "frontier": _infer_frontier(telemetry),
        "traversability": _infer_traversability(telemetry),
        "novelty": _infer_novelty(telemetry),
        "hazard": _infer_hazard(telemetry),
    }


def build_strategy_intent(vlm: dict, telemetry: dict, source: str, personality_key: str) -> StrategyIntent:
    observation = str(vlm.get("observation", "")).strip()
    headlight = _as_bool(vlm.get("headlight", False))

    frontier = _normalize_frontier(str(vlm.get("frontier", "")))
    if frontier is None:
        frontier = _infer_frontier(telemetry)

    traversability = _normalize_scene_level(str(vlm.get("traversability", "")))
    if traversability is None:
        traversability = _infer_traversability(telemetry)

    novelty = _normalize_scene_level(str(vlm.get("novelty", "")))
    if novelty is None:
        novelty = _infer_novelty(telemetry)

    hazard = _normalize_hazard(str(vlm.get("hazard", "")))
    if hazard is None:
        hazard = _infer_hazard(telemetry)

    frontier, traversability, hazard = _reconcile_scene_with_telemetry(
        frontier,
        traversability,
        hazard,
        telemetry,
    )

    objective = _compose_objective(personality_key, frontier, traversability, novelty, hazard)
    thought = _compose_persona_thought(personality_key, observation, objective)

    return StrategyIntent(
        objective=objective,
        thought=thought,
        observation=observation,
        headlight=headlight,
        frontier=frontier,
        traversability=traversability,
        novelty=novelty,
        hazard=hazard,
        source=source,
    )


# ---------------------------------------------------------------------------
# Tactical planner (pure planning; no HTTP side-effects)
# ---------------------------------------------------------------------------
@dataclass
class MotionPlan:
    action: str
    duration_ms: int
    speed_level: str
    mode: str
    reason: str


class IntentPlanner:
    def __init__(self):
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

    def _register_action(self, action: str) -> None:
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
            self.forward_chain_streak += 1
        else:
            self.forward_chain_streak = 0

        self.last_selected_action = action

    def _forward_ms(self, default_ms: int) -> int:
        configured = int(self.policy.get("forward_burst_ms", default_ms))
        return configured if configured > 0 else default_ms

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
            self.timeout_scan_budget = max(self.timeout_scan_budget, TIMEOUT_SCAN_STEPS)

        if self.timeout_scan_budget <= 0:
            return None

        if self.timeout_scan_budget == TIMEOUT_SCAN_STEPS:
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

        effective_rank = min(rank.get(base, 1), rank.get(cap, 1))
        speed = "slow"
        for key, value in rank.items():
            if value == effective_rank:
                speed = key
                break

        return speed, ",".join(reasons) if reasons else "none"

    def _plan_benson(self, intent: StrategyIntent, telemetry: dict, speed_level: str) -> MotionPlan:
        forward_ms = min(self._forward_ms(700), 750)
        turn_ms = min(TURN_DURATION_MS.get(self.persona, 550), 650)

        if intent.hazard == "hard" or intent.frontier == "blocked":
            if bool(telemetry.get("visual_obstacle", False)):
                return MotionPlan("stop", 0, "slow", "hold", "benson:hard_hold")
            return MotionPlan("backward", ESCAPE_BACKWARD_MS, "slow", "reverse", "benson:hard_reset")

        if intent.hazard == "soft" or intent.traversability == "low":
            return MotionPlan(self._open_turn(telemetry), turn_ms, "slow", "turn", "benson:risk_turn")

        if self.forward_chain_streak >= 4:
            return MotionPlan("stop", 0, "slow", "hold", "benson:safety_pause")

        if bool(telemetry.get("strategy_degraded", False)) and self.forward_chain_streak >= 2:
            return MotionPlan("stop", 0, "slow", "hold", "benson:verify_pause")

        if intent.frontier in {"left", "right"}:
            return MotionPlan(intent.frontier, turn_ms, speed_level, "turn", "benson:frontier_turn")

        return MotionPlan("forward", forward_ms, speed_level, "explore", "benson:controlled_advance")

    def _next_david_turn(self, hint: Literal["left", "right"] | None = None) -> Literal["left", "right"]:
        if hint in {"left", "right"}:
            self.david_turn_bias = hint
        else:
            self.david_turn_bias = "right" if self.david_turn_bias == "left" else "left"
        return self.david_turn_bias

    def _plan_sir_david(self, intent: StrategyIntent, telemetry: dict, speed_level: str) -> MotionPlan:
        forward_ms = max(self._forward_ms(900), 900)
        turn_ms = max(TURN_DURATION_MS.get(self.persona, 500), 520)

        if intent.hazard == "hard":
            return MotionPlan(self._open_turn(telemetry), turn_ms, "slow", "turn", "sirdavid:careful_reroute")

        if bool(telemetry.get("strategy_degraded", False)) and self.forward_chain_streak >= 1:
            return MotionPlan(self._next_david_turn(), turn_ms, "slow", "turn", "sirdavid:listen_scan")

        if intent.novelty == "high" and intent.hazard == "none":
            if self.forward_chain_streak >= 2:
                return MotionPlan(self._next_david_turn(), turn_ms, speed_level, "turn", "sirdavid:reframe_subject")
            return MotionPlan("forward", forward_ms + 200, speed_level, "explore", "sirdavid:approach_novelty")

        if self.forward_chain_streak >= 3:
            hint = intent.frontier if intent.frontier in {"left", "right"} else None
            return MotionPlan(self._next_david_turn(hint), turn_ms, speed_level, "turn", "sirdavid:wander_arc")

        if intent.frontier in {"left", "right"} and intent.hazard != "soft":
            return MotionPlan(intent.frontier, turn_ms, speed_level, "turn", "sirdavid:follow_frontier")

        return MotionPlan("forward", forward_ms, speed_level, "explore", "sirdavid:gentle_explore")

    def _plan_klaus(self, intent: StrategyIntent, telemetry: dict, speed_level: str, cycle_count: int) -> MotionPlan:
        preferred = str(self.policy.get("preferred_wall", "left"))
        if preferred not in {"left", "right"}:
            preferred = "left"
        opposite: Literal["left", "right"] = "right" if preferred == "left" else "left"
        preferred_turn: Literal["left", "right"] = "left" if preferred == "left" else "right"
        wall_signal = telemetry.get("obstacle_left_ratio", 0.0) if preferred == "left" else telemetry.get("obstacle_right_ratio", 0.0)
        wide_turn_ms = max(TURN_DURATION_MS.get(self.persona, 700), KLAUS_ARC_TURN_MS)

        if intent.hazard == "hard" or intent.frontier == "blocked":
            if bool(telemetry.get("visual_obstacle", False)):
                return MotionPlan("backward", ESCAPE_BACKWARD_MS, "slow", "reverse", "klaus:clearance_reset")
            return MotionPlan(opposite, wide_turn_ms, "slow", "turn", "klaus:detour_arc")

        if wall_signal < 0.08:
            self.klaus_wall_seek_streak += 1
            if self.klaus_wall_seek_streak % 2 == 0:
                return MotionPlan("forward", 600, speed_level, "explore", "klaus:probe_perimeter")
            return MotionPlan(preferred_turn, wide_turn_ms, speed_level, "turn", "klaus:acquire_perimeter")
        self.klaus_wall_seek_streak = 0

        if wall_signal > 0.72:
            return MotionPlan(opposite, 620, speed_level, "turn", "klaus:clearance_correction")

        if self.forward_chain_streak >= KLAUS_ARC_INTERVAL:
            arc_turn: Literal["left", "right"] = preferred_turn
            if cycle_count % 5 == 0:
                arc_turn = opposite
            return MotionPlan(arc_turn, wide_turn_ms, speed_level, "turn", "klaus:signature_arc")

        if bool(telemetry.get("strategy_degraded", False)) and self.forward_chain_streak >= 1:
            return MotionPlan(preferred_turn, wide_turn_ms, "slow", "turn", "klaus:scan_perimeter")

        return MotionPlan("forward", self._forward_ms(850), speed_level, "explore", "klaus:perimeter_sweep")

    def _plan_zog7(self, intent: StrategyIntent, telemetry: dict, speed_level: str) -> MotionPlan:
        preferred = str(self.policy.get("preferred_wall", "right"))
        if preferred not in {"left", "right"}:
            preferred = "right"
        preferred_turn: Literal["left", "right"] = "right" if preferred == "right" else "left"
        opposite: Literal["left", "right"] = "left" if preferred == "right" else "right"
        wall_signal = telemetry.get("obstacle_right_ratio", 0.0) if preferred == "right" else telemetry.get("obstacle_left_ratio", 0.0)
        light = str(telemetry.get("brightness", "Normal"))

        if light == "Bright" and intent.hazard == "none":
            self.zog_freeze_cooldown = 2
            return MotionPlan("stop", 0, "slow", "hold", "zog7:freeze_in_light")

        if self.zog_freeze_cooldown > 0:
            self.zog_freeze_cooldown -= 1
            if self.stop_streak == 0:
                return MotionPlan("stop", 0, "slow", "hold", "zog7:concealment_pause")

        if intent.hazard == "hard" or intent.frontier == "blocked":
            return MotionPlan("backward", ESCAPE_BACKWARD_MS, "slow", "reverse", "zog7:tactical_reset")

        if wall_signal < 0.06:
            return MotionPlan(preferred_turn, 620, speed_level, "turn", "zog7:seek_edge")

        if wall_signal > 0.70:
            return MotionPlan(opposite, 520, speed_level, "turn", "zog7:standoff_adjust")

        if self.forward_chain_streak >= 3:
            return MotionPlan("stop", 0, "slow", "hold", "zog7:listen_pause")

        return MotionPlan("forward", min(self._forward_ms(650), 720), speed_level, "explore", "zog7:shadow_run")

    def plan(self, intent: StrategyIntent, telemetry: dict, cycle_count: int) -> MotionPlan:
        base_speed = self._base_speed()
        speed_level, speed_reason = self._scene_speed(base_speed, intent, telemetry, cycle_count)
        timeout_plan = self._strategy_timeout_plan(speed_level, telemetry)
        if timeout_plan:
            self._register_action(timeout_plan.action)
            timeout_plan.reason = f"{timeout_plan.reason}|{speed_reason}"
            return timeout_plan

        if self.persona == "benson":
            plan = self._plan_benson(intent, telemetry, speed_level)
        elif self.persona == "sir_david":
            plan = self._plan_sir_david(intent, telemetry, speed_level)
        elif self.persona == "klaus":
            plan = self._plan_klaus(intent, telemetry, speed_level, cycle_count)
        elif self.persona == "zog7":
            plan = self._plan_zog7(intent, telemetry, speed_level)
        else:
            plan = MotionPlan("forward", self._forward_ms(800), speed_level, "explore", "default:advance")

        self._register_action(plan.action)
        plan.reason = f"{plan.reason}|{speed_reason}"
        return plan


# ---------------------------------------------------------------------------
# Wheelson client (HTTP only)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Motion authority (single writer to /move)
# ---------------------------------------------------------------------------
@dataclass
class MotionResult:
    action: str
    duration_ms: int
    safety: bool
    busy: bool
    command_id: str
    source: str
    mode: str


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
                            "🛑 [LEASE] firmware safety override while renewing %s",
                            self._active_continuous_action,
                        )
                        self._active_continuous_action = "stop"
                        self._active_mode = "idle"
                        self._heartbeat_safety_event = True
                except httpx.RequestError as exc:
                    self._missed_lease_count += 1
                    log.warning(
                        "📡 [LEASE] heartbeat failed (%d/%d) action=%s err=%s",
                        self._missed_lease_count,
                        LEASE_MISSED_LIMIT,
                        self._active_continuous_action,
                        exc,
                    )
                    if self._missed_lease_count >= LEASE_MISSED_LIMIT:
                        log.warning("🛑 [LEASE] missed too many heartbeats, forcing stop")
                        try:
                            await self._post_stop_locked(source="failsafe", mode="hold")
                        except Exception as stop_exc:
                            log.warning("🛑 [LEASE] failsafe stop command failed: %s", stop_exc)
                        finally:
                            self._active_continuous_action = "stop"
                            self._active_mode = "idle"
                            self._heartbeat_safety_event = True


# ---------------------------------------------------------------------------
# Deterministic recovery FSM
# ---------------------------------------------------------------------------
@dataclass
class RecoveryTrigger:
    level: Literal["nudge", "escape", "hard_escape"]
    turn_direction: Literal["left", "right"]


class RecoveryController:
    def __init__(self):
        self.low_motion_streak = 0
        self.escalation_level = 0
        self.cooldown_cycles = 0
        self._last_turn: Literal["left", "right"] = "left"

    def reset(self) -> None:
        self.low_motion_streak = 0
        self.escalation_level = 0
        self.cooldown_cycles = 0

    def _choose_turn(self, telemetry: dict) -> Literal["left", "right"]:
        left_ratio = telemetry.get("obstacle_left_ratio", 0.0)
        right_ratio = telemetry.get("obstacle_right_ratio", 0.0)
        if left_ratio + TURN_SIDE_MARGIN < right_ratio:
            self._last_turn = "left"
            return "left"
        if right_ratio + TURN_SIDE_MARGIN < left_ratio:
            self._last_turn = "right"
            return "right"
        self._last_turn = "right" if self._last_turn == "left" else "left"
        return self._last_turn

    def observe(
        self,
        *,
        tracking_forward: bool,
        motion_score: float,
        telemetry: dict,
    ) -> RecoveryTrigger | None:
        if self.cooldown_cycles > 0:
            self.cooldown_cycles -= 1

        if not tracking_forward or motion_score < 0.0:
            if motion_score >= MOTION_RECOVERY_SCORE:
                self.reset()
            else:
                self.low_motion_streak = max(0, self.low_motion_streak - 1)
            return None

        if motion_score >= MOTION_RECOVERY_SCORE:
            self.reset()
            return None

        if motion_score < VISUAL_STALL_SCORE_MAX:
            self.low_motion_streak += 2
        elif motion_score < MOTION_RECOVERY_SCORE:
            self.low_motion_streak += 1
        else:
            return None

        if self.cooldown_cycles > 0:
            return None

        turn_direction = self._choose_turn(telemetry)

        if self.escalation_level == 0 and self.low_motion_streak >= RECOVERY_NUDGE_STREAK:
            self.escalation_level = 1
            self.cooldown_cycles = RECOVERY_COOLDOWN_NUDGE
            return RecoveryTrigger(level="nudge", turn_direction=turn_direction)

        if self.escalation_level == 1 and self.low_motion_streak >= RECOVERY_ESCAPE_STREAK:
            self.escalation_level = 2
            self.cooldown_cycles = RECOVERY_COOLDOWN_ESCAPE
            return RecoveryTrigger(level="escape", turn_direction=turn_direction)

        if self.escalation_level >= 2 and self.low_motion_streak >= RECOVERY_HARD_STREAK:
            self.escalation_level = 0
            self.low_motion_streak = 0
            self.cooldown_cycles = RECOVERY_COOLDOWN_HARD
            return RecoveryTrigger(level="hard_escape", turn_direction=turn_direction)

        return None


# ---------------------------------------------------------------------------
# Explorer loop
# ---------------------------------------------------------------------------
async def explorer_loop() -> None:
    state.is_running = True
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

    loop = asyncio.get_running_loop()
    state.last_strategy_source = "bootstrap"
    state.last_strategy_update_ts = loop.time()

    strategy_refresh_interval = (
        GEMINI_MIN_INTERVAL if PROVIDER == "gemini" else OLLAMA_STRATEGY_MIN_INTERVAL
    )
    strategy_stale_interval = strategy_refresh_interval * STRATEGY_STALE_MULTIPLIER

    log.info(
        "🤖 Explorer started personality=%s %s interval=%.1fs",
        personality["emoji"],
        personality["name"],
        LOOP_INTERVAL,
    )
    log.info("🌐 Dashboard -> http://localhost:%d", PORT)

    async with httpx.AsyncClient() as client:
        wheelson = WheelsonClient(client)
        planner = IntentPlanner()
        planner.set_persona(state.personality_key)
        motion = MotionSupervisor(wheelson)
        recovery = RecoveryController()

        await motion.start()

        last_vlm_request_ts = 0.0
        vlm_backoff_until_ts = 0.0
        vlm_task: asyncio.Task | None = None

        async def request_strategy(frame_bytes: bytes, frame_telemetry: dict) -> dict:
            if PROVIDER == "ollama":
                return await asyncio.wait_for(
                    ask_vlm(frame_bytes, frame_telemetry, system, client),
                    timeout=OLLAMA_VLM_TIMEOUT_SEC,
                )
            return await ask_vlm(frame_bytes, frame_telemetry, system, client)

        async def stop_with_failsafe(reason: str) -> None:
            nonlocal vlm_task
            log.warning("🛑 [FAILSAFE] %s", reason)
            recovery.reset()
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
            while True:
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
                    fresh_observation = ""
                    strategy_timeout_event = False

                    # Consume completed VLM task
                    if vlm_task and vlm_task.done():
                        try:
                            vlm_update = vlm_task.result()
                            vlm_backoff_until_ts = 0.0
                            state.last_vlm_headlight_pref = vlm_update.get("headlight", False)
                            state.last_scene_frontier = str(vlm_update.get("frontier", "unknown"))
                            state.last_scene_traversability = str(vlm_update.get("traversability", "medium"))
                            state.last_scene_novelty = str(vlm_update.get("novelty", "medium"))
                            state.last_scene_hazard = str(vlm_update.get("hazard", "none"))
                            state.last_scene_observation = str(vlm_update.get("observation", ""))
                            state.last_strategy_source = "vlm"
                            state.last_strategy_update_ts = now_ts
                            fresh_observation = state.last_scene_observation
                            log.info(
                                "🧠 [VLM_SCENE] frontier=%s trav=%s novelty=%s hazard=%s",
                                state.last_scene_frontier,
                                state.last_scene_traversability,
                                state.last_scene_novelty,
                                state.last_scene_hazard,
                            )
                        except asyncio.TimeoutError:
                            if PROVIDER == "ollama":
                                log.warning(
                                    "🤖 [VLM] ollama refresh timed out after %.1fs; keeping previous strategy",
                                    OLLAMA_VLM_TIMEOUT_SEC,
                                )
                                vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + OLLAMA_TIMEOUT_BACKOFF_SEC)
                                strategy_timeout_event = True
                            else:
                                log.warning("🤖 [VLM] gemini request timed out; switching to local scene fallback")
                                vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + strategy_refresh_interval)
                            state.last_strategy_update_ts = now_ts - strategy_stale_interval
                        except VLMResponseError as exc:
                            log.warning(
                                "🤖 [VLM] %s parse failure: %s; switching to local scene fallback",
                                PROVIDER,
                                _exc_summary(exc),
                            )
                            vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + strategy_refresh_interval)
                            state.last_strategy_update_ts = now_ts - strategy_stale_interval
                        except Exception as exc:
                            if _is_vlm_quota_error(exc):
                                log.warning(
                                    "🤖 [VLM] %s quota/rate limited (%s); switching to local scene fallback",
                                    PROVIDER,
                                    _exc_summary(exc),
                                )
                                vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + 60.0)
                            else:
                                log.warning(
                                    "🤖 [VLM] %s strategy task error: %s; keeping previous strategy",
                                    PROVIDER,
                                    _exc_summary(exc),
                                )
                                vlm_backoff_until_ts = max(vlm_backoff_until_ts, now_ts + strategy_refresh_interval)
                            state.last_strategy_update_ts = now_ts - strategy_stale_interval
                        finally:
                            vlm_task = None

                    should_refresh_vlm = (
                        vlm_task is None
                        and now_ts >= vlm_backoff_until_ts
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
                    if state.last_strategy_update_ts <= 0.0 or strategy_age >= strategy_stale_interval:
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
                        if prev_source != "local":
                            log.info(
                                "🧠 [LOCAL_SCENE] nav=%s light=%s side(L=%.2f R=%.2f) -> frontier=%s hazard=%s",
                                state.last_nav_state,
                                telemetry.get("brightness", "Unknown"),
                                telemetry.get("obstacle_left_ratio", 0.0),
                                telemetry.get("obstacle_right_ratio", 0.0),
                                state.last_scene_frontier,
                                state.last_scene_hazard,
                            )

                    telemetry["strategy_timeout_event"] = strategy_timeout_event
                    telemetry["strategy_source"] = state.last_strategy_source
                    telemetry["strategy_age_sec"] = strategy_age if strategy_age != float("inf") else 999.0
                    telemetry["strategy_degraded"] = state.last_strategy_source != "vlm"

                    vlm = {
                        "observation": fresh_observation or state.last_scene_observation,
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
                        plan = planner.plan(intent, telemetry, state.cycle_count)
                        fallback_plan = plan.reason.startswith("gate:strategist_timeout")
                        source = "strategist_fallback" if fallback_plan else intent.source
                        result = await motion.apply_plan(plan, source=source)
                        if fallback_plan:
                            state.last_thought = _persona_timeout_scan_thought(state.personality_key)
                            state.last_strategy_source = "fallback"
                            state.last_strategy_update_ts = loop.time()
                        else:
                            state.last_thought = intent.thought
                        actual_action = result.action
                        actual_duration = result.duration_ms
                        plan_speed = plan.speed_level
                        plan_reason = plan.reason

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
                            "command_id": state.last_command_id,
                            "command_source": state.last_command_source,
                            "command_mode": state.last_command_mode,
                            "scene_frontier": intent.frontier,
                            "scene_traversability": intent.traversability,
                            "scene_novelty": intent.novelty,
                            "scene_hazard": intent.hazard,
                            "ts": entry["ts"],
                        }
                    )

                    log.info(
                        "🧭 cycle=%d nav=%s action=%s dur=%dms lease=%s dist=%.1fcm motion=%.2f streak=%d rec_lvl=%d rec_cd=%d spd=%s src=%s cmd=%s/%s/%s scene(f=%s t=%s n=%s h=%s) visual=%s side(L=%.2f R=%.2f) light=%s safety=%s objective='%.64s'",
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
                    log.error("📡 [NETWORK] Cannot reach Wheelson at %s: %s", WHEELSON_IP, exc)
                    await stop_with_failsafe("network error while robot may be moving")
                except VLMResponseError as exc:
                    log.error("🤖 [VLM] %s raw='%.200s'", exc, exc.raw.replace("\n", " "))
                    await stop_with_failsafe("invalid VLM response")
                except Exception as exc:
                    log.error("💥 [LOOP] Unexpected error: %s", exc, exc_info=True)
                    await stop_with_failsafe("unexpected middleware loop error")

                elapsed = loop.time() - cycle_start
                await asyncio.sleep(max(0.0, LOOP_INTERVAL - elapsed))
        finally:
            if vlm_task:
                vlm_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await vlm_task
            await motion.close()


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------
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


@app.get("/snapshot")
async def snapshot() -> Response:
    if not state.last_frame_b64:
        return Response(status_code=503, content="No frame captured yet")
    return Response(content=base64.b64decode(state.last_frame_b64), media_type="image/jpeg")


@app.get("/health")
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
            "strategy_age_sec": round(strategy_age_sec, 2) if strategy_age_sec >= 0 else -1,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
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
        help="Model name override. Defaults: gemini=gemini-2.5-flash, ollama=llava",
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
        log.error("❌ GEMINI_API_KEY is not set. Add it to your .env file and retry.")
        sys.exit(1)

    state.personality_key = args.personality
    p = PERSONALITIES[args.personality]

    log.info("═" * 55)
    log.info("  Wheelson Autonomous Explorer")
    log.info("  Personality : %s  %s", p["emoji"], p["name"])
    log.info("  Provider    : %s  (model: %s)", PROVIDER.upper(), ACTIVE_MODEL)
    log.info("  Move bias   : %s", p["move_bias"])
    log.info("  Wheelson IP : %s", WHEELSON_IP)
    log.info("  Dashboard   : http://localhost:%d", args.port)
    log.info("═" * 55)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
