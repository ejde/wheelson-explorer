"""Persistent, cancellable execution of bounded egocentric LLM skills."""
from motion import MotionPlan
from skills import validate_strategy


class SkillExecutive:
    def __init__(self):
        self.active = None
        self.pending = None
        self.index = 0
        self.started_at = None
        self.status = "idle"
        self.history = []
        self._seen_id = None

    def _record(self, status):
        self.status = status
        if self.active:
            self.history.append({"plan_id": self.active[0], "objective": self.active[1]["objective"],
                                 "step": self.index, "outcome": status})
            self.history = self.history[-12:]

    def cancel(self, reason="cancelled"):
        self._record(reason)
        self.active = self.pending = None
        self.started_at = None

    def submit(self, plan_id, strategy, expires_at, now):
        if plan_id == self._seen_id:
            return
        self._seen_id = plan_id
        strategy = validate_strategy(strategy)
        if strategy is None or expires_at <= now:
            return
        candidate = (plan_id, strategy, expires_at)
        # Adopt revisions at maneuver boundaries to avoid mid-arc discontinuities.
        if self.active:
            self.pending = candidate
        else:
            self.active = candidate
            self.index = 0
            self.started_at = None

    def tick(self, intent, telemetry, now, speed):
        if not self.active:
            return None
        if now >= self.active[2]:
            self.cancel("expired")
            return None
        if intent.hazard == "hard" or intent.frontier == "blocked" or telemetry.get("safety_latched"):
            self.cancel("blocked")
            return None
        step = self.active[1]["steps"][self.index]
        if self.started_at is None:
            self.started_at = now
        if now - self.started_at >= step["duration_s"]:
            # Time budget finished; this is not proof that a spatial goal was reached.
            self._record("budget_complete")
            self.index += 1
            if self.pending and self.pending[2] > now:
                self.active, self.pending = self.pending, None
                self.index = 0
            if self.index >= len(self.active[1]["steps"]):
                self.active = None
                self.started_at = None
                return None
            self.started_at = now
            step = self.active[1]["steps"][self.index]
        skill, side = step["skill"], step["direction"]
        if skill == "follow_boundary" and float(telemetry.get(f"obstacle_{side}_ratio", 0)) < 0.04:
            self.cancel("boundary_lost")
            return None
        self.status = "running"
        key = f"skill-{self.active[0]}-{self.index}"
        reason = f"skill:{skill}:{side}"
        if skill == "pause":
            return MotionPlan("stop", 0, "slow", "observe", reason, maneuver_id=key)
        if skill == "scan":
            return MotionPlan(side, int(step["duration_s"] * 1000), "slow", "scan", reason, maneuver_id=key)
        bias = 0.0 if side == "forward" else (0.3 if side == "left" else -0.3)
        if skill == "follow_boundary":
            # A visual occupancy band, not an invented metric distance.
            ratio = float(telemetry.get(f"obstacle_{side}_ratio", 0))
            bias = max(-0.3, min(0.3, (0.18 - ratio) * 2)) * (1 if side == "left" else -1)
        return MotionPlan("forward", 0, speed, "explore", reason, steer=bias, maneuver_id=key)

    def snapshot(self):
        return {"status": self.status, "objective": self.active[1]["objective"] if self.active else "",
                "step": self.index, "recent_outcomes": list(self.history[-4:])}
