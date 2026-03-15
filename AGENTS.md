# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Commands

### Middleware (Python)

```bash
cd middleware
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set WHEELSON_IP and GEMINI_API_KEY

# Run with Gemini (default provider)
python main.py --personality benson

# Run with Ollama (local VLM)
python main.py --provider ollama --model llava --personality sir_david

# Syntax check without a robot
python -c "import ast; ast.parse(open('main.py').read()); print('OK')"
```

There are no automated tests — validation is manual via the dashboard and robot behaviour.

### Firmware (Arduino / ESP32)

- Board: `ESP32 Dev Module`, partition `Huge APP (3MB No OTA/1MB SPIFFS)`, PSRAM `Enabled`
- Set `WIFI_SSID` / `WIFI_PASSWORD` in `firmware/wheelson_explorer/wheelson_explorer.ino`
- Flash via Arduino IDE 2.x; use upload speed `115200` if high-speed uploads fail

## Architecture

The project is split into firmware (C++/ESP32) and middleware (Python/FastAPI), communicating over HTTP on the local network. The middleware serves the dashboard and runs the main control loop.

### Subsumption Control Layers (priority high → low)

Each layer can suppress the layers below it. **All motor commands go through a single writer** (`MotionSupervisor`).

| Layer | What it does |
|---|---|
| **S0 Firmware Safety** | Lease-timeout hard stop; visual-obstacle latch. Always active. |
| **S1 Motion Authority** | `MotionSupervisor` — asyncio-locked single writer to `/move`, heartbeat renewal for continuous motion, command ID propagation. |
| **S2 Recovery FSM** | `RecoveryController` — deterministic nudge → escape → hard-escape escalation triggered by motion-score stall detection. |
| **S3 Persona Strategist** | `IntentPlanner` — per-persona motion doctrine. Switchable live via `POST /persona`. |
| **S4 VLM Semantic Interpreter** | Async advisory only. Outputs scene fields (`frontier`, `traversability`, `novelty`, `hazard`) + persona `commentary`. **Never issues motor commands.** |
| **S5 Human Override** | Remote control queue. Halts autonomous motion on takeover; autonomy resumes seamlessly on release/inactivity. |

### Main Loop (`explorer_loop` in `middleware/main.py`)

One asyncio task running at `LOOP_INTERVAL_SEC`. Each cycle:
1. Fetch JPEG + telemetry headers from `GET /camera` on the robot.
2. Apply pending **persona switch** (`state.pending_persona`) atomically — updates `IntentPlanner` + VLM system prompt.
3. Advance the **control queue** — expire timed-out sessions, update `state.remote_*` fields.
4. Fire an async **VLM task** if the refresh interval has elapsed (non-blocking).
5. Consume a completed VLM result → apply semantic contract (TTL + multi-frame confirmation gates).
6. Fall back to **local scene inference** when VLM is stale or in degraded/`local_only` mode.
7. Build `StrategyIntent` → run `IntentPlanner.plan()`.
8. **S5 gate**: if `state.remote_is_active`, skip S2–S3 execution entirely.
9. Otherwise: run Recovery FSM or apply the planner's `MotionPlan` via `MotionSupervisor`.
10. Broadcast SSE event to all connected dashboard clients.

### Key Classes (`middleware/main.py`)

- **`AppState`** — module-level dataclass singleton (`state`). All shared mutable state between the loop and HTTP handlers lives here.
- **`MotionSupervisor`** — asyncio-locked single writer. Exposed as module-level `_motion_supervisor` after loop startup so HTTP handlers (e.g. `POST /control`) can call `remote_move()` directly without going through the planner.
- **`IntentPlanner`** — stateful; holds latched plans and per-persona arc counters. Call `set_persona()` to switch live.
- **`RecoveryController`** — pure deterministic FSM; no VLM involvement.
- **`ControlSession`** — dataclass tracking token, name, `joined_at`, `last_command_at` for queue management (`_control_queue` list + `_queue_lock`).

### VLM Safety Contract

The VLM is non-blocking and non-authoritative for motion:
- Semantics expire after `SEMANTIC_TTL_SEC`; local inference fills the gap.
- Risky single-frame labels (`hard`, `blocked`) require multi-frame confirmation before acting.
- Timeout escalation: stage 1 → degraded mode, stage 2 → `local_only` lockout for `STRATEGY_LOCAL_ONLY_SEC`.

### Remote Control & Persona Switching

- `POST /queue/join` issues a token. `POST /control {token, direction}` calls `MotionSupervisor.remote_move()` with a 600 ms timed command — repeated by the frontend at ~350 ms while a key is held.
- After `CONTROL_INACTIVITY_SEC` (3 s) of no commands, `state.remote_is_active` becomes false and autonomy resumes; the session stays in queue until budget (`CONTROL_BUDGET_SEC`, 60 s) expires or the user calls `POST /queue/leave`.
- `POST /persona` sets `state.pending_persona`; consumed by the loop on the next cycle.
- Queue/controller state is broadcast each SSE cycle: `remote_controller`, `remote_queue_length`, `remote_remaining_s`, `remote_is_active`.

### Dashboard (`middleware/dashboard.html`)

Single static HTML file served by FastAPI — no build step. Connects to `GET /stream` (SSE). Key panels: persona switcher (4 emoji buttons → `POST /persona`), control queue with countdown timer bar, D-pad overlay on the camera (visible only when it's your turn, with pointer-capture for touch and WASD/arrow key support).

### Adding a New Persona

Five touch-points all in `middleware/main.py`, plus one in `dashboard.html`:
1. `PERSONALITIES` dict — name, emoji, color, system_prompt, move_bias.
2. `PERSONA_POLICIES` — base_speed, safety/curiosity/perimeter weights, forward_burst_ms, hold_bias.
3. `PERSONA_SALIENCE` and `PERSONA_COMMENTARY_STYLE` — VLM prompt hints.
4. `TURN_DURATION_MS` — default turn length in ms.
5. `_plan_<key>()` method in `IntentPlanner`, wired into `IntentPlanner.plan()`.
6. `PERSONALITIES` object in `dashboard.html` — emoji, color, bias text, voice hints.
