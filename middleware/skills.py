"""Strict, bounded language-to-skill contract. No model-provided motor values."""
import math

SKILL_DIRECTIONS = {
    "explore": {"forward", "left", "right"},
    "follow_boundary": {"left", "right"},
    "scan": {"left", "right"},
    "pause": {"forward"},
}


def observation_is_current(result: dict, generation: int, now: float, ttl: float) -> bool:
    captured = result.get("captured_at")
    return (result.get("generation") == generation and isinstance(captured, (int, float))
            and math.isfinite(captured) and 0 <= now - captured < ttl)


def validate_strategy(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    objective = value.get("objective")
    steps = value.get("steps")
    if not isinstance(objective, str) or not objective.strip() or not isinstance(steps, list):
        return None
    if not 1 <= len(steps) <= 3:
        return None
    validated = []
    for step in steps:
        if not isinstance(step, dict):
            return None
        skill, direction = step.get("skill"), step.get("direction", "forward")
        if not isinstance(skill, str) or skill not in SKILL_DIRECTIONS:
            return None
        if not isinstance(direction, str) or direction not in SKILL_DIRECTIONS[skill]:
            return None
        duration = step.get("duration_s", 3.0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
            return None
        maximum = 1.0 if skill == "scan" else 6.0
        if not 0.5 <= duration <= maximum:
            return None
        validated.append({"skill": skill, "direction": direction, "duration_s": float(duration)})
    return {"objective": objective.strip()[:160], "steps": validated}
