# Wheelson Explorer

Autonomous exploration stack for CircuitMess Wheelson using a **subsumption-type architecture**:
- firmware safety/reflexes,
- deterministic local planning,
- async VLM scene semantics,
- persona-specific strategy,
- **remote human override** with queued multi-viewer control.

The system is designed so robot motion remains safe and deterministic even when VLM calls are slow or unavailable — and human operators can take direct control at any time via the live dashboard.

## Architecture (Subsumption-Type)

Control is layered by priority. Higher-priority layers can suppress lower-priority intent.

1. **S0 Firmware Safety (hard realtime-ish)**
- Lease timeout stop (`lease_timeout` latch)
- Visual obstacle hard-stop (`visual_obstacle` latch)
- Exposes camera + telemetry headers + active command metadata

2. **S1 Motion Authority (middleware)**
- Single writer to `/move` (`MotionSupervisor`)
- Heartbeat renewal for continuous motion commands
- Command metadata propagation (`command_id`, `source`, `mode`)
- Smooth-control cadence: continuous forward lease with short latched plans to avoid action thrash

3. **S2 Recovery FSM (middleware)**
- Deterministic nudge/escape/hard-escape escalation
- Motion-score based no-progress recovery

4. **S3 Persona Strategist (middleware)**
- Persona-specific motion doctrine and objectives
- Distinct policies for `benson`, `sir_david`, `klaus`, `zog7`
- Can be switched live via dashboard or `POST /persona` without restarting

5. **S4 Semantic Interpreter (VLM, async advisory)**
- VLM outputs scene semantics (`frontier`, `traversability`, `novelty`, `hazard`, `headlight`, `observation`) plus persona commentary text (`commentary`)
- No direct motor commands from VLM

6. **S5 Human Override (remote control)**
- Takes priority over all planner/recovery layers
- VLM scene analysis and persona thoughts continue running normally
- On takeover: autonomous motion halts; on release: autonomy resumes seamlessly

### Control/Dataflow Diagram

```mermaid
flowchart TD
    CAM["Wheelson Camera + Telemetry Headers"] --> VLM["S4 Semantic Interpreter (VLM, async)"]
    CAM --> LSCENE["Local Scene Inference (deterministic fallback)"]
    VLM --> CONTRACT["Semantic Contract (TTL + Confirm Gates)"]
    LSCENE --> CONTRACT

    HUMAN["S5 Human Override (remote control queue)"] --> AUTH
    CONTRACT --> STRAT["S3 Persona Strategist (objectives + doctrine)"]
    STRAT --> REC["S2 Recovery FSM (nudge/escape/hard_escape)"]
    REC --> AUTH["S1 Motion Authority (single writer + lease heartbeat)"]
    STRAT --> AUTH

    AUTH --> FW["S0 Firmware Safety (lease timeout + visual hard-stop)"]
    FW --> MOTORS["Nuvoton Motor Control"]

    TO["VLM Timeout Escalation"] --> MODE["Mode: normal -> degraded -> local_only"]
    MODE --> STRAT
```

## VLM Safety Contract

The VLM is intentionally non-blocking and non-authoritative for motion.

- **Async only**: planner continues without waiting for VLM completion.
- **Thought cadence**: persona commentary updates on VLM refresh cadence, then persists between refreshes.
- **Semantic TTL**: stale VLM semantics expire (`SEMANTIC_TTL_SEC`).
- **Confirmation gates**: risky one-frame labels (`hard`, `blocked`) require confirmation across frames.
- **Timeout escalation**:
  - stage 1: timeout fallback (pause + scan behavior)
  - stage 2: degraded mode
  - stage 3: temporary `local_only` lockout (`STRATEGY_LOCAL_ONLY_SEC`)
- **Modes**:
  - `normal`
  - `degraded`
  - `local_only`

## Personas (Distinct Strategy Layer)

Each persona has distinct:
- movement doctrine,
- objective style,
- commentary style,
- degraded/timeout behavior.

| Persona | Emoji | Movement Signature | Primary Objective Style |
|---|---|---|---|
| `benson`    | 🦺 | conservative, safety pauses, risk-first reroutes   | compliance/safety sweep |
| `sir_david` | 🎙️ | curiosity arcs, novelty-biased approach/reframe    | documentary exploration |
| `klaus`     | 🎨 | perimeter acquisition + wide signature arcs        | spatial-flow/perimeter audit |
| `zog7`      | 👾 | edge-cover seeking, concealment pauses, tactical resets | covert mapping / exposure minimization |

Personas can be switched live from the dashboard or via `POST /persona` without restarting the middleware.

## Repository Structure

```text
wheelson-explorer/
├── firmware/
│   └── wheelson_explorer/
│       └── wheelson_explorer.ino
├── middleware/
│   ├── main.py            # FastAPI app + CLI entry point
│   ├── config.py          # Constants, persona dicts, env vars
│   ├── state.py           # AppState singleton, queue, broadcast
│   ├── loop.py            # explorer_loop (main control cycle)
│   ├── motion.py          # WheelsonClient, MotionSupervisor
│   ├── scene.py           # VLM calls, scene semantics, StrategyIntent
│   ├── planner.py         # IntentPlanner (persona-specific planning)
│   ├── recovery.py        # RecoveryController FSM
│   ├── remote.py          # Queue/control/persona/chat endpoints
│   ├── routes.py          # Dashboard, SSE, snapshot, health routes
│   ├── dashboard.html
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

## Hardware Notes

- CircuitMess Wheelson (ESP32)
- 2.4 GHz Wi-Fi
- Camera is primary perception channel

### Ultrasonic Status

The current stack treats ultrasonic as unavailable in runtime planning/safety.
- Middleware distance is treated as `999.0cm` placeholder.
- Firmware safety uses **visual obstacle telemetry** and lease safety.

## Firmware

File: `firmware/wheelson_explorer/wheelson_explorer.ino`

### Arduino Setup

- Board: `ESP32 Dev Module`
- Partition: `Huge APP (3MB No OTA/1MB SPIFFS)`
- PSRAM: `Enabled`
- Upload speed: `115200` if high-speed uploads fail

Set Wi-Fi credentials in the sketch:

```cpp
#define WIFI_SSID "WIFI_SSID"
#define WIFI_PASSWORD "WIFI_PASSWORD"
```

### Firmware API

#### `GET /camera`
Returns JPEG bytes plus telemetry headers:

- `X-Distance-CM`
- `X-Raw-Pulse-US`
- `X-Visual-Obstacle`
- `X-Obstacle-Left-Ratio`
- `X-Obstacle-Right-Ratio`
- `X-Nav-State`
- `X-Brightness`
- `X-Dominant-Color`
- `X-Active-Command-Id`
- `X-Active-Command-Source`
- `X-Active-Command-Mode`
- `X-Safety-Latched`
- `X-Safety-Reason`

#### `GET /status`
Returns state JSON including:

- `distance_cm`, `ip`
- `nav_state`
- `command_id`, `source`, `mode`
- `safety_latched`, `safety_reason`

#### `POST /move`
Supports command metadata:

- `command_id`
- `source`
- `mode`

Supported payload forms:

```json
{ "command": "set_speed", "level": "slow|medium|fast", "command_id": "...", "source": "...", "mode": "..." }
```

```json
{ "command": "set_light", "level": "0..255", "command_id": "...", "source": "...", "mode": "..." }
```

```json
{ "command": "move_indefinitely", "direction": "forward|backward|left|right|stop", "command_id": "...", "source": "...", "mode": "..." }
```

```json
{ "action": "forward|backward|left|right|stop", "duration_ms": 500, "command_id": "...", "source": "...", "mode": "..." }
```

## Middleware

File: `middleware/main.py`

### Install

```bash
cd middleware
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Core Environment Variables

```env
WHEELSON_IP=192.168.1.112
GEMINI_API_KEY=...
LOOP_INTERVAL_SEC=1.2
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434
```

### Important Optional Controls

```env
# VLM timing
OLLAMA_STRATEGY_MIN_INTERVAL_SEC=8.0
OLLAMA_VLM_TIMEOUT_SEC=14.0
OLLAMA_TIMEOUT_BACKOFF_SEC=24.0

# Semantic contract
SEMANTIC_TTL_SEC=18.0
SEMANTIC_HARD_CONFIRM_FRAMES=2
SEMANTIC_BLOCKED_CONFIRM_FRAMES=2

# Timeout escalation
STRATEGY_TIMEOUT_WINDOW_SEC=90.0
STRATEGY_TIMEOUT_STAGE2_COUNT=2
STRATEGY_TIMEOUT_STAGE3_COUNT=3
STRATEGY_LOCAL_ONLY_SEC=60.0

# Persona strategy tuning
FORWARD_BURST_DEFAULT_MS=850
KLAUS_ARC_INTERVAL=3
KLAUS_ARC_TURN_MS=760
TIMEOUT_SCAN_STEPS=2
TIMEOUT_SCAN_TURN_MS=700
PLAN_LATCH_FORWARD_SEC=1.8
PLAN_LATCH_TURN_SEC=0.9

# Remote control
CONTROL_BUDGET_SEC=60.0        # seconds each person in the queue gets to drive
CONTROL_INACTIVITY_SEC=3.0     # seconds of no input before autonomy quietly resumes
```

### Run

```bash
# Gemini
python main.py --provider gemini --personality benson

# Ollama
python main.py --provider ollama --model llava --personality klaus
```

CLI:

```text
python main.py [--provider gemini|ollama] [--model MODEL] [--personality NAME] [--port PORT]
```

## Remote Control

Any browser connected to the dashboard can queue up to drive Wheelson directly.

### How it works

1. Enter a name and click **JOIN** — you're added to the queue.
2. When it's your turn, the D-pad appears on the camera feed and keyboard control activates.
3. Drive with **WASD / arrow keys** or the on-screen D-pad. Holding a key sends repeated commands at ~350 ms intervals; releasing sends a stop.
4. You have **60 seconds** (configurable). If you go idle for **3 seconds**, autonomy quietly resumes — but your slot stays in the queue until your budget expires or you click LEAVE.
5. On session end (timeout or LEAVE), the next person in the queue becomes the active controller. Wheelson returns to full autonomous mode when the queue is empty.

All viewers see a live countdown and who's currently in control via SSE. Safety layers (firmware obstacle detection, lease timeout) remain active regardless of who is driving.

### Remote Control API

| Endpoint | Body | Description |
|---|---|---|
| `POST /queue/join` | `{"name": "Alice"}` | Join the queue; returns `{token, position, budget_sec}` |
| `POST /queue/leave` | `{"token": "..."}` | Leave the queue immediately |
| `POST /control` | `{"token": "...", "direction": "forward\|backward\|left\|right\|stop"}` | Send a movement command (only works if you're first in queue) |
| `GET /queue/status` | — | Current queue state and remaining budgets |

### Persona Switching API

| Endpoint | Body | Description |
|---|---|---|
| `POST /persona` | `{"persona": "benson\|sir_david\|klaus\|zog7"}` | Switch active persona live |

The switch is applied on the next loop cycle: planner doctrine and VLM system prompt update atomically, no restart needed.

### Chat API

| Endpoint | Body | Description |
|---|---|---|
| `POST /chat` | `{"message": "What do you see?"}` | Chat with the active persona; returns in-character reply (max 3 sentences) |

Responses are context-aware — the persona references the current scene and its recent observations. Rate-limited to 5 requests per minute (returns 429 with `retry_after_sec` on exceed). Chat uses text-only VLM calls, separate from the main scene-analysis loop.

## Runtime Telemetry / Health

### Dashboard

Default: [http://localhost:8000](http://localhost:8000)

Features: live camera feed, persona switcher, internal monologue typewriter, distance bar, event log, remote control queue panel with D-pad overlay, in-character chat.

### `GET /health` (middleware)

Includes high-value runtime state:

- `strategy_source`
- `strategy_mode`
- `strategy_degraded_level`
- `strategy_age_sec`
- `semantic_confidence`
- `semantic_pending_hard`
- `semantic_pending_blocked`
- `vlm_timeout_count`
- `vlm_disabled_remaining_sec`
- scene fields (`scene_frontier`, `scene_traversability`, `scene_novelty`, `scene_hazard`)
- command authority fields (`command_id`, `command_source`, `command_mode`)
- remote control fields (`remote_controller`, `remote_queue_length`, `remote_remaining_s`, `remote_is_active`)

## How Persona Interacts with VLM Now

- VLM does **not** roleplay personas and does **not** emit actions.
- Persona is applied in two safe ways:
  1. **Salience hints** to VLM prompt (what to notice in scene interpretation).
  2. **Local strategist doctrine** for movement/objectives/commentary.

This preserves deterministic control while still allowing persona-specific behavior.

## Known Operational Guidance

- If Ollama is slow or timing out, robot continues in local mode with staged degradation.
- `nav=HOLD` is expected when using timed motion pulses (`duration_ms > 0`).
- Prefer local vision telemetry and motion consistency metrics over ultrasonic.

## License

MIT
