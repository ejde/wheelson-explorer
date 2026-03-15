"""
wheelson-explorer · middleware · scene.py
Scene semantics, VLM calling, strategy intent, and image helpers.
"""

import asyncio
import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Literal

import google.genai as genai
from google.genai import types as genai_types
import httpx
from PIL import Image

from state import state
from config import (
    GEMINI_API_KEY,
    OLLAMA_BASE_URL,
    RESPONSE_SCHEMA,
    SCENE_INTERPRETER_PROMPT,
    PERSONA_SALIENCE,
    PERSONA_COMMENTARY_STYLE,
    SEMANTIC_HARD_CONFIRM_FRAMES,
    SEMANTIC_BLOCKED_CONFIRM_FRAMES,
    TURN_SIDE_MARGIN,
    SCENE_HARD_CLEAR_THRESHOLD,
    SCENE_SOFT_CLEAR_THRESHOLD,
    SCENE_FORWARD_CLEAR_THRESHOLD,
    SCENE_HOLD_RESET_THRESHOLD,
    ACTIVE_MODEL,
    PROVIDER,
)
import config as _config_module


# ---------------------------------------------------------------------------
# Exceptions and helpers
# ---------------------------------------------------------------------------
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


def _is_placeholder_text(value: str) -> bool:
    v = value.strip().lower()
    if not v:
        return True
    placeholders = {
        "...",
        "\u2026",
        "n/a",
        "none",
        "1 sentence max",
        "brief factual description",
        "short factual description",
        "awaiting first cycle\u2026",
        "awaiting first cycle...",
    }
    return v in placeholders


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Persona reactive thoughts
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# JSON extraction / normalization
# ---------------------------------------------------------------------------
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
# Type aliases
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


# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# VLM semantic contract
# ---------------------------------------------------------------------------
def _apply_vlm_semantic_contract(vlm_scene: dict, telemetry: dict) -> tuple[dict, float, str]:
    """Gate unstable semantic jumps from a single VLM frame."""
    frontier = _normalize_frontier(str(vlm_scene.get("frontier", ""))) or "unknown"
    traversability = _normalize_scene_level(str(vlm_scene.get("traversability", ""))) or "medium"
    novelty = _normalize_scene_level(str(vlm_scene.get("novelty", ""))) or "medium"
    hazard = _normalize_hazard(str(vlm_scene.get("hazard", ""))) or "none"

    visual_obstacle = bool(telemetry.get("visual_obstacle", False))
    side_peak = max(
        telemetry.get("obstacle_left_ratio", 0.0),
        telemetry.get("obstacle_right_ratio", 0.0),
    )
    hard_evidence = visual_obstacle or side_peak >= 0.35
    blocked_evidence = visual_obstacle or side_peak >= 0.30

    notes: list[str] = []
    confidence = 0.8

    if hazard == "hard" and not hard_evidence:
        state.semantic_pending_hard += 1
        if state.semantic_pending_hard < SEMANTIC_HARD_CONFIRM_FRAMES:
            hazard = "soft"
            confidence = min(confidence, 0.45)
            notes.append("hard_unconfirmed")
    else:
        state.semantic_pending_hard = SEMANTIC_HARD_CONFIRM_FRAMES if hazard == "hard" else 0

    if frontier == "blocked" and not blocked_evidence:
        state.semantic_pending_blocked += 1
        if state.semantic_pending_blocked < SEMANTIC_BLOCKED_CONFIRM_FRAMES:
            frontier = "unknown"
            confidence = min(confidence, 0.45)
            notes.append("blocked_unconfirmed")
    else:
        state.semantic_pending_blocked = SEMANTIC_BLOCKED_CONFIRM_FRAMES if frontier == "blocked" else 0

    if hazard == "none" and frontier == "forward" and side_peak < 0.03 and not visual_obstacle:
        confidence = min(confidence, 0.7)

    vetted = {
        "observation": str(vlm_scene.get("observation", "")).strip(),
        "commentary": str(vlm_scene.get("commentary", "")).strip(),
        "headlight": _as_bool(vlm_scene.get("headlight", False)),
        "frontier": frontier,
        "traversability": traversability,
        "novelty": novelty,
        "hazard": hazard,
    }
    reason = ",".join(notes) if notes else "ok"
    return vetted, confidence, reason


# ---------------------------------------------------------------------------
# Local inference functions
# ---------------------------------------------------------------------------
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
        return "high"  # darkness = unexplored territory
    side_peak = max(
        telemetry.get("obstacle_left_ratio", 0.0),
        telemetry.get("obstacle_right_ratio", 0.0),
    )
    if side_peak >= 0.30:
        return "high"  # nearby objects = something interesting to investigate
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


# ---------------------------------------------------------------------------
# Objective / thought composition
# ---------------------------------------------------------------------------
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
        "commentary": "",
        "headlight": telemetry.get("brightness") == "Dark",
        "frontier": _infer_frontier(telemetry),
        "traversability": _infer_traversability(telemetry),
        "novelty": _infer_novelty(telemetry),
        "hazard": _infer_hazard(telemetry),
    }


def build_strategy_intent(vlm: dict, telemetry: dict, source: str, personality_key: str) -> StrategyIntent:
    observation = str(vlm.get("observation", "")).strip()
    commentary = str(vlm.get("commentary", "")).strip()
    if _is_placeholder_text(commentary):
        commentary = ""
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
    thought = commentary if commentary else _compose_persona_thought(personality_key, observation, objective)

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
# VLM integration
# ---------------------------------------------------------------------------
def _system_prompt(personality_key: str) -> str:
    salience = PERSONA_SALIENCE.get(personality_key, PERSONA_SALIENCE["benson"])
    style = PERSONA_COMMENTARY_STYLE.get(personality_key, PERSONA_COMMENTARY_STYLE["benson"])
    return (
        SCENE_INTERPRETER_PROMPT
        + f" Salience profile: {salience}"
        + f" Commentary style: {style}."
        + RESPONSE_SCHEMA
    )


def _build_prompt(telemetry: dict) -> str:
    salience = PERSONA_SALIENCE.get(state.personality_key, PERSONA_SALIENCE["benson"])
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
    prior_commentary = state.last_vlm_commentary if state.last_vlm_commentary else "none"

    return (
        f"{mem_lines}"
        " Current system telemetry: Ultrasonic distance=unavailable (sensor hardware offline)."
        f" Lighting={telemetry.get('brightness', 'Unknown')}."
        f" Dominant Floor Color={telemetry.get('dominant_color', 'Unknown')}."
        f" Firmware nav state: {telemetry.get('nav_state', 'UNKNOWN')}."
        f" Side obstacle ratios: left={telemetry.get('obstacle_left_ratio', 0.0):.2f}, right={telemetry.get('obstacle_right_ratio', 0.0):.2f}."
        f" Persona salience hints: {salience}"
        f"{obs_warn}"
        f" Last planner objective: {state.last_objective}."
        f" Last VLM commentary: {prior_commentary}."
        f" Last action: {state.last_action} (repeated {state.action_streak}x)."
        f"{streak_warning}"
        " Provide a fresh commentary line; do not repeat the previous commentary verbatim."
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
            model=_config_module.ACTIVE_MODEL,
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
        "model": _config_module.ACTIVE_MODEL,
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
    if _config_module.PROVIDER == "gemini":
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
    commentary = str(parsed.get("commentary", "")).strip()
    if _is_placeholder_text(commentary):
        commentary = ""
    if len(commentary) > 220:
        commentary = commentary[:220].rstrip() + "..."

    frontier = str(parsed.get("frontier", "")).strip().lower()
    traversability = str(parsed.get("traversability", "")).strip().lower()
    novelty = str(parsed.get("novelty", "")).strip().lower()
    hazard = str(parsed.get("hazard", "")).strip().lower()

    return {
        "observation": observation,
        "commentary": commentary,
        "headlight": _as_bool(parsed.get("headlight", False)),
        "frontier": frontier,
        "traversability": traversability,
        "novelty": novelty,
        "hazard": hazard,
    }
