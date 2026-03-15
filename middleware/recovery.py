"""
wheelson-explorer · middleware · recovery.py
Deterministic recovery FSM.
"""

from dataclasses import dataclass
from typing import Literal

from config import (
    TURN_SIDE_MARGIN,
    VISUAL_STALL_SCORE_MAX,
    MOTION_RECOVERY_SCORE,
    RECOVERY_NUDGE_STREAK,
    RECOVERY_ESCAPE_STREAK,
    RECOVERY_HARD_STREAK,
    RECOVERY_COOLDOWN_NUDGE,
    RECOVERY_COOLDOWN_ESCAPE,
    RECOVERY_COOLDOWN_HARD,
)


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
