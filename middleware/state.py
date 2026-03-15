"""
wheelson-explorer · middleware · state.py
Shared state singleton, broadcast, control queue, and related module-level vars.
"""

import asyncio
from dataclasses import dataclass, field


@dataclass
class ControlSession:
    token: str
    name: str
    joined_at: float
    last_command_at: float = 0.0


@dataclass
class AppState:
    personality_key: str = "benson"
    last_frame_b64: str = ""
    last_thought: str = "Awaiting first cycle\u2026"
    last_objective: str = "Awaiting first cycle\u2026"
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
    strategy_mode: str = "normal"
    strategy_degraded_level: int = 0
    semantic_confidence: float = 0.5
    semantic_pending_hard: int = 0
    semantic_pending_blocked: int = 0
    vlm_timeout_count: int = 0
    vlm_disabled_until_ts: float = 0.0
    last_vlm_headlight_pref: bool = False
    last_scene_frontier: str = "unknown"
    last_scene_traversability: str = "medium"
    last_scene_novelty: str = "medium"
    last_scene_hazard: str = "none"
    last_scene_observation: str = ""
    last_vlm_commentary: str = ""

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

    # Remote control / queue state
    remote_controller_name: str = ""
    remote_control_remaining_s: float = 0.0
    remote_queue_length: int = 0
    remote_is_active: bool = False


state = AppState()
_sse_subscribers: set[asyncio.Queue] = set()

# Remote control state (shared between HTTP handlers and explorer_loop)
_control_queue: list[ControlSession] = []
_queue_lock = asyncio.Lock()
_motion_supervisor: "object | None" = None  # Set to MotionSupervisor at runtime
_persona_queue: asyncio.Queue = asyncio.Queue(maxsize=4)

_chat_timestamps: list[float] = []
CHAT_RATE_LIMIT = 5  # requests per minute
CHAT_RATE_WINDOW = 60.0  # seconds


def broadcast(event: dict) -> None:
    dead: set[asyncio.Queue] = set()
    for q in _sse_subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.add(q)
    _sse_subscribers.difference_update(dead)
