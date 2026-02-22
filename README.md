# Wheelson Explorer 🤖

An autonomous explorer for the CircuitMess Wheelson. The robot streams camera frames over Wi-Fi; a Python middleware feeds them to a Vision-Language Model (Gemini or a local Ollama model), and the VLM's decision steers the motors — all visualised through a live browser dashboard with per-personality voice narration.

Four **personalities** give the robot a completely different character, narration style, and movement philosophy.

---

## Repository Structure

```
wheelson-explorer/
├── firmware/
│   └── wheelson_explorer/
│       └── wheelson_explorer.ino   ← ESP32 firmware (no external libraries)
├── middleware/
│   ├── main.py                     ← FastAPI + explorer loop
│   ├── dashboard.html              ← live browser UI (SSE, Web Speech API)
│   ├── requirements.txt
│   └── .env.example
├── .gitignore
└── README.md
```

---

## Hardware Requirements

| Item | Notes |
|------|-------|
| CircuitMess Wheelson | Any hardware revision (ESP32-D0WDQ6) |
| HC-SR04 Ultrasonic Sensor | Wired to GPIO 13 (TRIG) / GPIO 12 (ECHO) |
| USB cable (Micro-USB) | For flashing only — runs untethered after |
| Wi-Fi network (2.4 GHz) | Wheelson does not support 5 GHz |

### HC-SR04 Wiring

```
HC-SR04 VCC  →  Wheelson 3.3 V or 5 V
HC-SR04 GND  →  Wheelson GND
HC-SR04 TRIG →  GPIO 13
HC-SR04 ECHO →  GPIO 12  (add a 1 kΩ / 2 kΩ voltage divider if powering at 5 V)
```

> To use different pins, update `#define TRIG_PIN` and `#define ECHO_PIN` at the top of `wheelson_explorer.ino`.

---

## Firmware Setup

### Prerequisites

1. **Arduino IDE 2.x** — [download](https://www.arduino.cc/en/software)
2. **ESP32 board support** — *File → Preferences → Additional Boards Manager URLs*, add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
   Then *Tools → Board → Boards Manager* → search **esp32** → install **esp32 by Espressif Systems**.

> **No external libraries required.** The firmware uses only the ESP32 Arduino core (`WiFi.h`, `WebServer.h`, `Wire.h`, `esp_camera.h`). JSON is handled with plain `String` helpers.

### Arduino IDE Board Settings (critical)

| Setting | Value |
|---------|-------|
| Board | `ESP32 Dev Module` |
| Partition Scheme | `Huge APP (3MB No OTA / 1MB SPIFFS)` |
| **PSRAM** | **`Enabled`** ← required for camera frame buffer |
| Upload Speed | `115200` ← use this if the default 921600 causes upload errors |
| Port | `/dev/cu.usbserial-XXXX` (macOS) |

### Flashing

1. Open `firmware/wheelson_explorer/wheelson_explorer.ino`.
2. Edit the Wi-Fi credentials near the top:
   ```cpp
   #define WIFI_SSID     "YourNetworkName"
   #define WIFI_PASSWORD "YourPassword"
   ```
3. Apply the board settings above (especially **PSRAM → Enabled** and upload speed **115200**).
4. Connect Wheelson via USB, then click **Upload**.

#### Upload troubleshooting

If you see `The chip stopped responding` or `StopIteration` in the esptool output:

- **Set Upload Speed to 115200** — the default 921600 drops out on many USB-to-serial adapters.
- **Hold the BOOT button** — while the IDE shows `Connecting……`, hold the physical BOOT button for 2 s, then release. This forces the ESP32 into bootloader mode.
- **Check your USB cable** — charge-only cables have no data lines; the upload will fail silently.

### Verify Firmware

Open **Serial Monitor** at **115200 baud** and press the reset button. You should see:

```
╔══════════════════════════════════╗
║  Wheelson Autonomous Explorer    ║
╚══════════════════════════════════╝
[NUVOTON] Reset complete
[MOTORS] I2C ready
[CAMERA] Ready (160×120 JPEG)
[WIFI] Connecting to 'YourNetwork'......
[WIFI] Connected! IP: 192.168.1.112
[HTTP] Server started on port 80
```

Then smoke-test the endpoints:

```bash
# Live camera frame (open in browser)
open http://192.168.1.112/camera

# Status JSON
curl http://192.168.1.112/status

# Test move command (should spin wheels for 500 ms)
curl -X POST http://192.168.1.112/move \
  -H "Content-Type: application/json" \
  -d '{"action":"forward","duration_ms":500}'
```

> **Unplug and run on battery** — once flashed, the firmware runs on any USB power source. Wheelson will auto-connect to Wi-Fi on boot.

---

## Middleware Setup

### Prerequisites

- Python 3.10+
- One of the following VLM providers:
  - **Gemini** — [Google AI Studio API key](https://aistudio.google.com/app/apikey) (free tier: 5 req/min)
  - **Ollama** — [install Ollama](https://ollama.com/download) and pull a vision model

### Install

```bash
cd middleware
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
WHEELSON_IP=192.168.1.112        # IP printed in Serial Monitor on boot
GEMINI_API_KEY=AIza...           # only needed for --provider gemini
LOOP_INTERVAL_SEC=4.0            # seconds between VLM cycles
                                 # (Gemini free tier auto-raises this to 12 s)
PORT=8000
OLLAMA_BASE_URL=http://localhost:11434   # optional, default shown
```

### Run — Gemini provider

```bash
# Default model: gemini-2.0-flash
python main.py --provider gemini --personality sir_david

# Explicitly choose a model
python main.py --provider gemini --model gemini-2.5-flash --personality benson
```

**Gemini rate limiting** — the free tier allows 5 requests per minute. The middleware automatically enforces a 12-second minimum cycle interval when `--provider gemini` is active, regardless of `LOOP_INTERVAL_SEC`.

Available Gemini models (as of Feb 2026):

| Model | Notes |
|-------|-------|
| `gemini-2.0-flash` | Default — fast, vision-capable, stable |
| `gemini-2.5-flash` | Newer, slightly smarter |
| `gemini-2.5-pro` | Best reasoning, slower |
| `gemini-2.0-flash-lite` | Fastest, smallest |

### Run — Ollama provider (local, no rate limits)

```bash
# Pull a vision-capable model first (one-time)
ollama pull llava          # 4.7 GB — recommended
ollama pull llava:13b      # 8 GB — better reasoning
ollama pull moondream      # 1.7 GB — fastest, less accurate

# Run
python main.py --provider ollama --model llava --personality zog7

# Verify Ollama is running
curl http://localhost:11434          # should print "Ollama is running"
ollama list                          # shows installed models
```

**Ollama requires a vision-capable model.** Text-only models (`qwen2.5-coder`, `llama3`, etc.) will not understand the camera images.

---

## Personalities

| Key | Name | Character | Voice (Web Speech API) |
|-----|------|-----------|------------------------|
| `benson` | BENSON 🦺 | Health & Safety inspector | Alex / Daniel (authoritative) |
| `sir_david` | Sir David 🎙️ | Nature documentary narrator | Daniel / Arthur (British) |
| `klaus` | Klaus 🎨 | Interior designer (appalled) | Martha / Victoria (stern) |
| `zog7` | Zog-7 👾 | Alien scout | Zarvox / Trinoids (robotic) |

Each personality injects a custom **system prompt** and receives context-rich per-cycle prompts:

```
Distance sensor: 42.3 cm. Last action: forward (repeated 2x). Reply with ONLY the JSON.
```

The VLM responds with:

```json
{
  "action": "forward",
  "duration_ms": 400,
  "thought": "A filing cabinet. Utterly beige. Moving on."
}
```

---

## Safety

Two independent safety layers prevent collisions:

| Layer | Trigger | Response |
|-------|---------|----------|
| **Middleware** (25 cm zone) | `distance_cm < 25` and action is `forward` | Replaces action with a random left/right turn |
| **Stuck detection** | Same `forward` action 5× in a row | Forces a random turn regardless of VLM output |
| **LLM context** | Streak ≥ 5 | Prompt warns "you are probably stuck — turn NOW" |
| **Firmware** (10 cm hard stop) | `distance_cm < 10` and action is `forward` | Overrides to `stop` at the hardware level |

---

## Dashboard

Open **http://localhost:8000** after starting the middleware.

| Element | Description |
|---------|-------------|
| 📷 Camera feed | Live JPEG, auto-flipped 180° (corrects inverted camera mount) |
| 💬 Thought bubble | VLM narration with fast typewriter animation |
| 🔊 Voice | Web Speech API — personality-specific voice, mute button in header |
| 🟢 Action badge | Color-coded: green=forward, red=stop, yellow=turn, orange=backward |
| 📏 Distance meter | Bar + value; amber < 30 cm, red < 10 cm |
| 📜 Event log | Rolling log of last 20 thoughts + actions |

**Voice note** — browsers require a user gesture before playing audio. Click the **🔊 VOICE** button once after loading the page; any thought that arrived before you clicked will play immediately.

---

## CLI Reference

```
python main.py [--provider gemini|ollama] [--model MODEL] [--personality NAME] [--port PORT]

--provider   gemini (default) or ollama
--model      model name; defaults: gemini-2.0-flash / llava
--personality  benson | sir_david | klaus | zog7  (default: benson)
--port       dashboard port (default: 8000, overrides $PORT)
```

---

## Firmware API Reference

### `GET /camera`
Returns raw JPEG bytes.  
`X-Distance-CM` response header: current HC-SR04 reading in cm.

### `GET /status`
```json
{ "distance_cm": "42.6", "ip": "192.168.1.112" }
```

### `POST /move`
```json
{ "action": "forward|backward|left|right|stop", "duration_ms": 500 }
```
Response:
```json
{ "ok": true, "action": "forward", "duration_ms": 500, "distance_cm": "42.6", "safety": false }
```
`"safety": true` means the firmware's hard-stop overrode the requested action.

---

## License

MIT
