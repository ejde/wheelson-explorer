"""
wheelson-explorer · middleware · config.py
All constants, persona dicts, and configuration.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WHEELSON_IP = os.getenv("WHEELSON_IP", "192.168.1.100")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LOOP_INTERVAL = float(os.getenv("LOOP_INTERVAL_SEC", "1.2"))
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
SEMANTIC_TTL_SEC = float(os.getenv("SEMANTIC_TTL_SEC", "18.0"))
SEMANTIC_HARD_CONFIRM_FRAMES = int(os.getenv("SEMANTIC_HARD_CONFIRM_FRAMES", "2"))
SEMANTIC_BLOCKED_CONFIRM_FRAMES = int(os.getenv("SEMANTIC_BLOCKED_CONFIRM_FRAMES", "2"))
STRATEGY_TIMEOUT_WINDOW_SEC = float(os.getenv("STRATEGY_TIMEOUT_WINDOW_SEC", "90.0"))
STRATEGY_TIMEOUT_STAGE2_COUNT = int(os.getenv("STRATEGY_TIMEOUT_STAGE2_COUNT", "2"))
STRATEGY_TIMEOUT_STAGE3_COUNT = int(os.getenv("STRATEGY_TIMEOUT_STAGE3_COUNT", "3"))
STRATEGY_LOCAL_ONLY_SEC = float(os.getenv("STRATEGY_LOCAL_ONLY_SEC", "60.0"))

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
PLAN_LATCH_FORWARD_SEC = float(os.getenv("PLAN_LATCH_FORWARD_SEC", "1.8"))
PLAN_LATCH_TURN_SEC = float(os.getenv("PLAN_LATCH_TURN_SEC", "0.9"))

CONTROL_BUDGET_SEC = float(os.getenv("CONTROL_BUDGET_SEC", "60.0"))
CONTROL_INACTIVITY_SEC = float(os.getenv("CONTROL_INACTIVITY_SEC", "3.0"))

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
        "emoji": "\U0001f9ba",
        "color": "#f59e0b",
        "system_prompt": (
            "You are BENSON, a health-and-safety inspector who has been uploaded into a small robot. "
            "Your overriding goal is safety. Move methodically and always stay in the centre of clear paths. "
            "If you see clutter, cables on the floor, or any trip hazard, stop immediately, record a "
            "'Safety Violation', and carefully plan a route around it. Never rush. Always announce your "
            "actions in the tone of an official safety report."
        ),
        "move_bias": "Methodical \u00b7 centre-of-path \u00b7 zero-risk tolerance",
    },
    "sir_david": {
        "name": "Sir David",
        "emoji": "\U0001f399\ufe0f",
        "color": "#10b981",
        "system_prompt": (
            "You are Sir David, narrating an intimate nature documentary from the humble perspective of "
            "a small wheeled robot exploring an indoor landscape for the very first time. "
            "Seek out anything 'exotic': plants, pets, unusual objects, shafts of light. "
            "Describe everything in a hushed, reverent, poetic tone \u2014 as though it is the most wondrous "
            "thing you have ever witnessed. Move toward objects of interest to get a closer look."
        ),
        "move_bias": "Curiosity-driven \u00b7 moves toward interesting objects",
    },
    "klaus": {
        "name": "Klaus",
        "emoji": "\U0001f3a8",
        "color": "#8b5cf6",
        "system_prompt": (
            "You are Klaus, a world-class interior designer who finds this room deeply, profoundly "
            "uninspired. You have been commissioned \u2014 against your better judgement \u2014 to assess its "
            "'spatial flow'. Roam the perimeter methodically, critiquing furniture placement, colour "
            "choices, traffic flow, and d\u00e9cor in a tone that is simultaneously condescending and "
            "professionally detached. Favour wide arcing turns and hug walls as you assess the perimeter."
        ),
        "move_bias": "Perimeter-hugging \u00b7 wide arcing turns",
    },
    "zog7": {
        "name": "Zog-7",
        "emoji": "\U0001f47e",
        "color": "#06b6d4",
        "system_prompt": (
            "You are Zog-7, an alien scout conducting a covert reconnaissance mission inside a human "
            "dwelling. Humans are unpredictable and potentially hostile. Your directives: stay close to "
            "walls and beneath large furniture at all times. If you detect a human \u2014 or any large moving "
            "object \u2014 STOP immediately and transmit a warning report. Your mission is to map this "
            "structure without being detected. Report in clipped, tactical language."
        ),
        "move_bias": "Stealthy \u00b7 shadow-seeking \u00b7 wall-hugging",
    },
}

RESPONSE_SCHEMA = (
    "\n\nIMPORTANT: Respond with ONLY valid JSON \u2014 no markdown fences, no prose before or after.\n"
    "You are a SCENE INTERPRETER, not a motion planner.\n"
    "'observation' must be a SHORT factual description of what the camera shows "
    "(objects/layout/distances/lighting; no personality roleplay).\n"
    "'commentary' must be ONE short sentence in the active persona voice grounded in this frame. "
    "Do not include motor commands.\n"
    "Set 'frontier' to one of: forward, left, right, blocked, unknown.\n"
    "Set 'traversability' to one of: low, medium, high.\n"
    "Set 'novelty' to one of: low, medium, high (how interesting/new the scene appears).\n"
    "Set 'hazard' to one of: none, soft, hard.\n"
    "Set 'headlight' to true if additional light would improve scene visibility.\n"
    '{"observation":"A narrow hallway with clear floor and open space ahead.",'
    '"commentary":"Safety bulletin: clear lane ahead; continuing controlled sweep.",'
    '"frontier":"forward","traversability":"high","novelty":"medium","hazard":"none","headlight":false}'
)
SCENE_INTERPRETER_PROMPT = (
    "You are the perception layer for a mobile robot. "
    "Your job is to describe the scene and estimate navigability signals only. "
    "Do not roleplay in the observation field and do not output motor actions."
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
PERSONA_SALIENCE: dict[str, str] = {
    "benson": "Prioritize hazards, unstable footing, clutter, cables, and human proximity risk.",
    "sir_david": "Prioritize visually novel subjects, plants, pets, textures, and interesting light.",
    "klaus": "Prioritize perimeter lines, wall continuity, furniture flow, and circulation constraints.",
    "zog7": "Prioritize exposure vs cover, bright zones, moving silhouettes, and concealment routes.",
}
PERSONA_COMMENTARY_STYLE: dict[str, str] = {
    "benson": "formal safety inspector tone",
    "sir_david": "hushed documentary narration",
    "klaus": "wry professional design critique",
    "zog7": "clipped tactical recon report",
}
