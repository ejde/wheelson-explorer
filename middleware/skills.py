"""Strict, bounded language-to-skill contract. No model-provided motor values."""
import math

SKILL_DIRECTIONS = {
    "explore": {"forward", "left", "right"},
    "follow_boundary": {"left", "right"},
    "scan": {"left", "right"},
    "pause": {"forward"},
}


def observation_is_current(result: dict, generation: int, now: float, ttl: float) -> bool:
    return vlm_discard_reason(result, generation, now, ttl) is None


def vlm_discard_reason(result: dict, generation: int, now: float, ttl: float) -> str | None:
    """Why observation_is_current() rejects a VLM update, or None if current.

    The loop logs this instead of a generic message, so the next
    age-vs-generation mystery is diagnosable from the log alone. (The
    old message blamed "persona generation" for what was always this
    currency check; persona narration fails JSON parsing elsewhere
    with its own message.)
    """
    if result.get("generation") != generation:
        return f"generation changed (task {result.get('generation')} != current {generation})"
    captured = result.get("captured_at")
    if not isinstance(captured, (int, float)) or not math.isfinite(captured):
        return "captured_at missing or not finite"
    age = now - captured
    if not 0 <= age < ttl:
        return f"observation age {age:.1f}s outside [0, {ttl}s)"
    return None


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
