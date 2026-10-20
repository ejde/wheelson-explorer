#!/usr/bin/env python3
"""
wheelson_bridge.py — bridge between the Muse Gadget Linux SDK and the
Wheelson Explorer middleware.

Design rules (inherited from wheelson-explorer's own safety contract):
  * This bridge NEVER talks to the Wheelson firmware directly. It only calls
    the middleware's HTTP API on localhost, so MotionSupervisor stays the
    single writer to /move and Protocol 2 session/sequence/lease rules hold.
  * Driving goes through the middleware's S5 remote-control queue, exactly
    like a human on the dashboard: join -> control pulses -> leave.
    Firmware safety (750 ms lease watchdog, visual-obstacle hard-stop) and
    the middleware's renewal window stay in force underneath every pulse.
  * One `drive` call is a bounded pulse train, never sustained motion:
    `repeats` (1-5) control calls spaced 350 ms apart (the dashboard's
    key-hold cadence), then a final `stop`. If anything fails mid-train,
    a best-effort `stop` is attempted before the error is returned.
  * No dependencies beyond the Python 3.9+ standard library.

State: the queue token is cached in a state file (default
~/.wheelson-gadget/session.json, mode 0600) because each gadget command
invocation is a fresh process. The token is a queue credential for THIS
host only; the middleware expires it after CONTROL_BUDGET_SEC (60 s
default) and after that the bridge simply re-joins.

CLI (used by the executor snippet, which runs this as a child process as
the gadget's run-as account):

    python3 wheelson_bridge.py health
    python3 wheelson_bridge.py queue_status
    python3 wheelson_bridge.py persona  '{"persona": "klaus"}'
    python3 wheelson_bridge.py chat     '{"message": "What do you see?"}'
    python3 wheelson_bridge.py drive    '{"direction": "forward", "repeats": 2}'
    python3 wheelson_bridge.py leave
    python3 wheelson_bridge.py snapshot

Every command prints one JSON object: {"ok": true, "payload": {...}} or
{"ok": false, "error": "..."} and exits 0 / 1 accordingly.

Config via environment:
    WHEELSON_MIDDLEWARE_URL  default http://127.0.0.1:8000
    WHEELSON_GADGET_NAME     queue display name, default "Finch"
    WHEELSON_STATE_DIR       default ~/.wheelson-gadget
    WHEELSON_SNAPSHOT_DIR    default ~/wheelson-snapshots
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MIDDLEWARE_URL = os.environ.get("WHEELSON_MIDDLEWARE_URL", "http://127.0.0.1:8000").rstrip("/")
GADGET_NAME = os.environ.get("WHEELSON_GADGET_NAME", "Finch")
STATE_DIR = Path(os.environ.get("WHEELSON_STATE_DIR", Path.home() / ".wheelson-gadget"))
SNAPSHOT_DIR = Path(os.environ.get("WHEELSON_SNAPSHOT_DIR", Path.home() / "wheelson-snapshots"))

PERSONAS = ("benson", "sir_david", "klaus", "zog7")
DIRECTIONS = ("forward", "backward", "left", "right", "stop")
MAX_REPEATS = 5
PULSE_INTERVAL_SEC = 0.35  # matches the dashboard's key-hold repeat cadence
HTTP_TIMEOUT_SEC = 10.0


# ---------------------------------------------------------------------------
# Small HTTP client (stdlib only)
# ---------------------------------------------------------------------------
class MiddlewareError(Exception):
    def __init__(self, message: str, status: int | None = None, payload: dict | None = None):
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


def _request(method: str, path: str, body: dict | None = None, timeout: float = HTTP_TIMEOUT_SEC):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(MIDDLEWARE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get_content_type()
            if ctype == "application/json":
                return resp.status, json.loads(raw or b"{}")
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {"error": raw.decode(errors="replace")[:200]}
        raise MiddlewareError(
            str(payload.get("error") or f"HTTP {exc.code}"), status=exc.code, payload=payload
        ) from exc
    except urllib.error.URLError as exc:
        raise MiddlewareError(f"middleware unreachable at {MIDDLEWARE_URL}: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# Queue session state
# ---------------------------------------------------------------------------
def _state_path() -> Path:
    return STATE_DIR / "session.json"


def _load_token() -> str | None:
    try:
        data = json.loads(_state_path().read_text())
        return data.get("token") or None
    except (OSError, json.JSONDecodeError):
        return None


def _save_token(token: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path()
    path.write_text(json.dumps({"token": token, "name": GADGET_NAME}))
    path.chmod(0o600)


def _clear_token() -> None:
    try:
        _state_path().unlink()
    except OSError:
        pass


def _ensure_token() -> str:
    """Return a live queue token, joining (or re-joining) if needed."""
    token = _load_token()
    if token:
        # Cheap liveness check: we are controller only if queue head is us.
        # queue_status doesn't expose tokens, so instead probe with a
        # harmless `stop` — the middleware answers not_your_turn (403) if
        # the cached token is stale/expired. A stop when it IS our turn is
        # a no-op safety-wise.
        try:
            _request("POST", "/control", {"token": token, "direction": "stop"})
            return token
        except MiddlewareError as exc:
            if exc.status != 403:
                raise
            _clear_token()
    _, payload = _request("POST", "/queue/join", {"name": GADGET_NAME})
    token = payload["token"]
    _save_token(token)
    return token


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_health(_params: dict) -> dict:
    _, p = _request("GET", "/health")
    keep = (
        "status", "personality", "personality_name", "cycle_count", "hard_stops",
        "last_action", "last_thought", "last_distance_cm", "last_nav_state",
        "recovery_level", "strategy_mode", "command_source", "command_mode",
    )
    return {k: p[k] for k in keep if k in p}


def cmd_queue_status(_params: dict) -> dict:
    _, p = _request("GET", "/queue/status")
    return p


def cmd_persona(params: dict) -> dict:
    persona = params.get("persona", "")
    if persona not in PERSONAS:
        raise MiddlewareError(f"invalid persona {persona!r}; valid: {', '.join(PERSONAS)}")
    _, p = _request("POST", "/persona", {"persona": persona})
    return p


def cmd_chat(params: dict) -> dict:
    message = (params.get("message") or "").strip()
    if not message:
        raise MiddlewareError("message is required")
    _, p = _request(
        "POST", "/chat", {"message": message[:400], "name": GADGET_NAME}, timeout=30.0
    )
    return p


def cmd_drive(params: dict) -> dict:
    direction = params.get("direction", "")
    if direction not in DIRECTIONS:
        raise MiddlewareError(f"invalid direction {direction!r}; valid: {', '.join(DIRECTIONS)}")
    repeats = params.get("repeats", 1)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not (1 <= repeats <= MAX_REPEATS):
        raise MiddlewareError(f"repeats must be an int 1-{MAX_REPEATS}")

    token = _ensure_token()
    if direction == "stop":
        _request("POST", "/control", {"token": token, "direction": "stop"})
        return {"ok": True, "direction": "stop", "pulses": 1}

    sent = 0
    try:
        for i in range(repeats):
            _request("POST", "/control", {"token": token, "direction": direction})
            sent += 1
            if i < repeats - 1:
                time.sleep(PULSE_INTERVAL_SEC)
    except MiddlewareError:
        # Any failure mid-train: try to stop before surfacing the error.
        try:
            _request("POST", "/control", {"token": token, "direction": "stop"})
        except MiddlewareError:
            pass
        raise
    # Deterministic end state: stopped. Motion would also lapse on its own
    # via the middleware renewal window + firmware watchdog, but don't rely
    # on a timeout for something we can close explicitly.
    _request("POST", "/control", {"token": token, "direction": "stop"})
    return {"direction": direction, "pulses": sent, "ended": "stop"}


def cmd_leave(_params: dict) -> dict:
    token = _load_token()
    if not token:
        return {"left": False, "reason": "not_in_queue"}
    try:
        _, p = _request("POST", "/queue/leave", {"token": token})
        return {"left": True, **p}
    finally:
        _clear_token()


def cmd_snapshot(_params: dict) -> dict:
    status, raw = _request("GET", "/snapshot")
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise MiddlewareError(f"unexpected snapshot response (HTTP {status})")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / "latest.jpg"
    path.write_bytes(raw)
    # The gadget link caps a single message (~256 KB) and file.read moves
    # 64 KB base64 chunks, so return a path + metadata, not the JPEG.
    return {
        "path": str(path),
        "bytes": len(raw),
        "note": "Read it back with file.read in 64 KB chunks.",
    }


COMMANDS = {
    "health": cmd_health,
    "queue_status": cmd_queue_status,
    "persona": cmd_persona,
    "chat": cmd_chat,
    "drive": cmd_drive,
    "leave": cmd_leave,
    "snapshot": cmd_snapshot,
}


def run_command(name: str, params: dict) -> dict:
    fn = COMMANDS.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown command: {name}"}
    try:
        return {"ok": True, "payload": fn(params or {})}
    except MiddlewareError as exc:
        return {"ok": False, "error": str(exc)}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in COMMANDS:
        print(json.dumps({"ok": False, "error": f"usage: wheelson_bridge.py <{'|'.join(COMMANDS)}> [json-params]"}))
        return 1
    params = {}
    if len(argv) > 2:
        try:
            params = json.loads(argv[2])
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"bad params JSON: {exc}"}))
            return 1
    result = run_command(argv[1], params)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
