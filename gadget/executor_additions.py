"""executor_additions.py — what to add to the Muse Gadget Linux SDK.

Target file: linux/src/musegadget/executor.py  (in your local clone of
facebookincubator/muse-gadget-sdk, installed with `install.sh --from .`)

This follows the SDK's own AGENTS.md "Adding a command" recipe:
  1. Add WHEELSON_SPECS below into COMMAND_SPECS.
  2. Add the `wheelson.*` branch + the wheelson() method to Executor.
  3. Copy wheelson_bridge.py to /opt/musegadget/wheelson_bridge.py on the
     host (readable by the run-as account). The handler runs it as a child
     process with _child_options(), so all middleware traffic happens as
     the unprivileged run-as account, never as the service account.
  4. Restart: sudo systemctl restart musegadget
     Muse sees the new commands when the service re-registers.

Do NOT fork executor.py wholesale — keep these as additions so upstream
SDK updates stay a clean merge.
"""

# ---------------------------------------------------------------------------
# 1) Merge into COMMAND_SPECS in executor.py
# ---------------------------------------------------------------------------
WHEELSON_SPECS = {
    "wheelson.health": {
        "description": (
            "Wheelson Explorer status from the local middleware: personality, "
            "last action/thought, distance, nav state, recovery and strategy mode."
        ),
        "required": {},
        "optional": {},
    },
    "wheelson.queue_status": {
        "description": "Who is in the Wheelson remote-control queue and their remaining budgets.",
        "required": {},
        "optional": {},
    },
    "wheelson.persona": {
        "description": "Switch Wheelson's active persona live (no restart).",
        "required": {
            "persona": {
                "type": "string",
                "description": "One of: benson, sir_david, klaus, zog7.",
            },
        },
        "optional": {},
    },
    "wheelson.chat": {
        "description": (
            "Chat with Wheelson's active persona. Returns its in-character reply "
            "(max 3 sentences), grounded in the current scene. Rate-limited by "
            "the middleware (5/min); a 429 surfaces as an error."
        ),
        "required": {
            "message": {"type": "string", "description": "Message to Wheelson (max 400 chars)."},
        },
        "optional": {},
        "timeout_ms": 60000,
    },
    "wheelson.drive": {
        "description": (
            "Drive Wheelson via the middleware's remote-control queue (S5). Each "
            "call is a bounded pulse train: `repeats` (1-5) control pulses 350 ms "
            "apart, then an explicit stop. Firmware lease watchdog and visual "
            "hard-stop remain active. Never produces sustained motion."
        ),
        "required": {
            "direction": {
                "type": "string",
                "description": "One of: forward, backward, left, right, stop.",
            },
        },
        "optional": {
            "repeats": {
                "type": "integer",
                "description": "Pulse count, 1-5. Default 1.",
            },
        },
        "timeout_ms": 30000,
    },
    "wheelson.leave": {
        "description": "Leave the Wheelson remote-control queue and hand control back to autonomy.",
        "required": {},
        "optional": {},
    },
    "wheelson.snapshot": {
        "description": (
            "Save Wheelson's latest camera frame to ~/wheelson-snapshots/latest.jpg "
            "on this device and return its path and size. Read the JPEG back with "
            "file.read in 64 KB chunks."
        ),
        "required": {},
        "optional": {},
    },
}

# Map gadget command name -> bridge subcommand
WHEELSON_COMMANDS = {
    "wheelson.health": "health",
    "wheelson.queue_status": "queue_status",
    "wheelson.persona": "persona",
    "wheelson.chat": "chat",
    "wheelson.drive": "drive",
    "wheelson.leave": "leave",
    "wheelson.snapshot": "snapshot",
}

BRIDGE_PATH = "/opt/musegadget/wheelson_bridge.py"


# ---------------------------------------------------------------------------
# 2) Add to Executor.run's dispatch (next to the device.health branch):
#
#     if command in WHEELSON_COMMANDS: return self.wheelson(command, params)
#
# 3) Add this method to the Executor class:
# ---------------------------------------------------------------------------
def wheelson_method_for_executor(self, command: str, params: dict) -> dict:
    """Reference implementation — paste into Executor as `wheelson`."""
    import json as _json
    import subprocess as _subprocess
    import sys as _sys

    sub = WHEELSON_COMMANDS[command]
    proc = _subprocess.run(
        [_sys.executable, BRIDGE_PATH, sub, _json.dumps(params or {})],
        capture_output=True,
        timeout=90,
        cwd="/",
        **self._child_options(),
    )
    try:
        return _json.loads(proc.stdout)
    except Exception:
        return {
            "ok": False,
            "error": (proc.stderr.decode(errors="replace")[-2000:] or "wheelson bridge failed"),
        }
