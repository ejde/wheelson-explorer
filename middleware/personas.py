"""User-defined personas for the Wheelson explorer middleware.

Built-in personas (benson, sir_david, klaus, zog7) live in config.py.
Users with their own Muse define additional personas in a YAML file
instead of editing Python:

    middleware/personas.yaml          # your personas (gitignored, never committed)
    middleware/personas.example.yaml  # tracked template, includes the Finch example

A persona entry is keyed by a short id (lowercase letters, digits,
underscore) and may set the fields below. Unknown fields are rejected:
a misspelled key must fail loudly, not be silently ignored. A custom
persona may also override a built-in key; the override is logged, and a
reload restores the built-in if the override is removed from the file.

Loading is explicit: main() calls apply_custom_personas() at startup,
and POST /personas/reload re-reads the file at runtime. If the file
fails to load, the current set is left untouched and the error
propagates. This module only validates and merges dictionaries; it
never touches the robot.
"""

import copy
import logging
import re
from pathlib import Path

import yaml

import config

log = logging.getLogger("wheelson")

PERSONAS_FILE = Path(__file__).with_name("personas.yaml")

# Snapshots of the built-ins, taken at import time (before any apply).
# Reload restores from these, so removing a custom persona (or a custom
# override of a built-in) from the file really removes it.
_BUILTIN_PERSONALITIES = copy.deepcopy(config.PERSONALITIES)
_BUILTIN_POLICIES = copy.deepcopy(config.PERSONA_POLICIES)
_BUILTIN_SALIENCE = copy.deepcopy(config.PERSONA_SALIENCE)
_BUILTIN_COMMENTARY = copy.deepcopy(config.PERSONA_COMMENTARY_STYLE)
_BUILTIN_TURN_MS = copy.deepcopy(config.TURN_DURATION_MS)

# Keys currently supplied by the user file (including overrides).
CUSTOM_KEYS: set[str] = set()

_BASE_SPEEDS = {"slow", "medium", "fast"}
_WALLS = {"none", "left", "right"}

_REQUIRED_FIELDS = {
    "name",
    "system_prompt",
    "base_speed",
    "safety_weight",
    "curiosity_weight",
    "perimeter_weight",
    "preferred_wall",
    "forward_burst_ms",
    "hold_bias",
}

_OPTIONAL_FIELDS = {
    "emoji": "\U0001f916",
    "color": "#888888",
    "move_bias": "",
    "salience": "Prioritize whatever is most interesting in the scene.",
    "commentary_style": "neutral tone",
    "turn_duration_ms": 600,
}

_KNOWN_FIELDS = _REQUIRED_FIELDS | set(_OPTIONAL_FIELDS)

_KEY_RE = re.compile(r"^[a-z0-9_]+$")
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _fail(message: str) -> None:
    raise ValueError(f"personas.yaml: {message}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_persona(key, raw) -> dict:
    if not isinstance(key, str) or not _KEY_RE.match(key):
        _fail(f"persona id {key!r} must be lowercase letters, digits, or underscore")
    if not isinstance(raw, dict):
        _fail(f"persona '{key}' must be a mapping of fields")
    unknown = sorted(set(raw) - _KNOWN_FIELDS)
    if unknown:
        _fail(f"persona '{key}' has unknown field(s): {', '.join(unknown)}")
    missing = sorted(_REQUIRED_FIELDS - set(raw))
    if missing:
        _fail(f"persona '{key}' is missing required field(s): {', '.join(missing)}")

    name = raw["name"]
    system_prompt = raw["system_prompt"]
    if not isinstance(name, str) or not name.strip():
        _fail(f"persona '{key}': 'name' must be a non-empty string")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        _fail(f"persona '{key}': 'system_prompt' must be a non-empty string")

    base_speed = raw["base_speed"]
    if base_speed not in _BASE_SPEEDS:
        _fail(
            f"persona '{key}': 'base_speed' must be one of {sorted(_BASE_SPEEDS)}, "
            f"got {base_speed!r}"
        )
    preferred_wall = raw["preferred_wall"]
    if preferred_wall not in _WALLS:
        _fail(
            f"persona '{key}': 'preferred_wall' must be one of {sorted(_WALLS)}, "
            f"got {preferred_wall!r}"
        )

    for field in ("safety_weight", "curiosity_weight", "perimeter_weight"):
        value = raw[field]
        if not _is_number(value) or value < 0:
            _fail(f"persona '{key}': '{field}' must be a number >= 0, got {value!r}")

    hold_bias = raw["hold_bias"]
    if not _is_number(hold_bias) or not 0.0 <= hold_bias <= 1.0:
        _fail(f"persona '{key}': 'hold_bias' must be a number between 0 and 1, got {hold_bias!r}")

    forward_burst_ms = raw["forward_burst_ms"]
    if not _is_int(forward_burst_ms) or forward_burst_ms <= 0:
        _fail(f"persona '{key}': 'forward_burst_ms' must be a positive integer, got {forward_burst_ms!r}")
    turn_duration_ms = raw.get("turn_duration_ms", _OPTIONAL_FIELDS["turn_duration_ms"])
    if not _is_int(turn_duration_ms) or turn_duration_ms <= 0:
        _fail(f"persona '{key}': 'turn_duration_ms' must be a positive integer, got {turn_duration_ms!r}")

    emoji = raw.get("emoji", _OPTIONAL_FIELDS["emoji"])
    color = raw.get("color", _OPTIONAL_FIELDS["color"])
    move_bias = raw.get("move_bias", _OPTIONAL_FIELDS["move_bias"])
    salience = raw.get("salience", _OPTIONAL_FIELDS["salience"])
    commentary_style = raw.get("commentary_style", _OPTIONAL_FIELDS["commentary_style"])
    for field, value in (
        ("emoji", emoji),
        ("color", color),
        ("move_bias", move_bias),
        ("salience", salience),
        ("commentary_style", commentary_style),
    ):
        if not isinstance(value, str):
            _fail(f"persona '{key}': '{field}' must be a string, got {value!r}")
    if not _COLOR_RE.match(color):
        _fail(f"persona '{key}': 'color' must be #rrggbb, got {color!r}")

    return {
        "name": name,
        "emoji": emoji,
        "color": color,
        "system_prompt": system_prompt,
        "move_bias": move_bias,
        "policy": {
            "base_speed": base_speed,
            "safety_weight": raw["safety_weight"],
            "curiosity_weight": raw["curiosity_weight"],
            "perimeter_weight": raw["perimeter_weight"],
            "preferred_wall": preferred_wall,
            "forward_burst_ms": forward_burst_ms,
            "hold_bias": hold_bias,
        },
        "salience": salience,
        "commentary_style": commentary_style,
        "turn_duration_ms": turn_duration_ms,
    }


def load_custom_personas(path=None) -> dict:
    """Parse the user file without touching the live registry."""
    target = Path(path) if path is not None else PERSONAS_FILE
    if not target.exists():
        return {}
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"personas.yaml: invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        _fail("top level must be a mapping of persona id -> fields")
    return {key: _validate_persona(key, raw) for key, raw in data.items()}


def restore_builtin_personas() -> None:
    """Reset every persona dict to the built-ins, dropping all customs."""
    config.PERSONALITIES.clear()
    config.PERSONALITIES.update(copy.deepcopy(_BUILTIN_PERSONALITIES))
    config.PERSONA_POLICIES.clear()
    config.PERSONA_POLICIES.update(copy.deepcopy(_BUILTIN_POLICIES))
    config.PERSONA_SALIENCE.clear()
    config.PERSONA_SALIENCE.update(copy.deepcopy(_BUILTIN_SALIENCE))
    config.PERSONA_COMMENTARY_STYLE.clear()
    config.PERSONA_COMMENTARY_STYLE.update(copy.deepcopy(_BUILTIN_COMMENTARY))
    config.TURN_DURATION_MS.clear()
    config.TURN_DURATION_MS.update(copy.deepcopy(_BUILTIN_TURN_MS))
    CUSTOM_KEYS.clear()


def apply_custom_personas(customs=None, path=None) -> list:
    """Restore built-ins, then merge the user file. Returns applied keys.

    Pass pre-loaded customs to avoid parsing twice (the reload endpoint
    loads once for its guard check, then applies the same data). If the
    file fails to load, nothing is changed and the error propagates.
    """
    if customs is None:
        customs = load_custom_personas(path)
    restore_builtin_personas()
    for key, persona in customs.items():
        overridden = key in config.PERSONALITIES
        config.PERSONALITIES[key] = {
            "name": persona["name"],
            "emoji": persona["emoji"],
            "color": persona["color"],
            "system_prompt": persona["system_prompt"],
            "move_bias": persona["move_bias"],
        }
        config.PERSONA_POLICIES[key] = dict(persona["policy"])
        config.PERSONA_SALIENCE[key] = persona["salience"]
        config.PERSONA_COMMENTARY_STYLE[key] = persona["commentary_style"]
        config.TURN_DURATION_MS[key] = persona["turn_duration_ms"]
        CUSTOM_KEYS.add(key)
        log.info(
            "[PERSONA] Custom persona '%s' loaded%s",
            key,
            " (overrides built-in)" if overridden else "",
        )
    return sorted(customs)


def builtin_persona_keys() -> set:
    """Keys the repo ships, before any user file is applied."""
    return set(_BUILTIN_PERSONALITIES)


def persona_source(key: str) -> str:
    return "custom" if key in CUSTOM_KEYS else "builtin"


def describe_personas() -> dict:
    """Everything the dashboard/API needs to list a persona."""
    out = {}
    for key, identity in config.PERSONALITIES.items():
        out[key] = {
            "name": identity["name"],
            "emoji": identity["emoji"],
            "color": identity["color"],
            "move_bias": identity["move_bias"],
            "policy": dict(config.PERSONA_POLICIES.get(key, {})),
            "salience": config.PERSONA_SALIENCE.get(key, ""),
            "commentary_style": config.PERSONA_COMMENTARY_STYLE.get(key, ""),
            "turn_duration_ms": config.TURN_DURATION_MS.get(key),
            "source": persona_source(key),
        }
    return out
