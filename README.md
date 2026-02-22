# Wheelson Explorer 🤖

An autonomous explorer powered by **Google Gemini 1.5 Flash**. The CircuitMess Wheelson streams camera frames over Wi-Fi; a Python middleware feeds them to a Vision-Language Model, and the VLM's decision steers the motors — all viewed through a live browser dashboard.

Four **personalities** each give the robot a completely different character, narration style, and movement philosophy.

---

## Repository Structure

```
wheelson-explorer/
├── firmware/
│   └── wheelson_explorer/
│       └── wheelson_explorer.ino   ← ESP32 firmware
├── middleware/
│   ├── main.py                     ← FastAPI + explorer loop
│   ├── dashboard.html              ← live browser UI
│   ├── requirements.txt
│   └── .env.example
├── .gitignore
└── README.md
```

---

## Hardware Requirements

| Item | Notes |
|------|-------|
| CircuitMess Wheelson | Any hardware revision |
| HC-SR04 Ultrasonic Sensor | Wired to GPIO 13 (TRIG) and GPIO 12 (ECHO) |
| USB cable (Micro-USB) | For flashing firmware |
| Wi-Fi network (2.4 GHz) | Wheelson does not support 5 GHz |

### HC-SR04 Wiring

```
HC-SR04 VCC  →  Wheelson 3.3V or 5V
HC-SR04 GND  →  Wheelson GND
HC-SR04 TRIG →  GPIO 13
HC-SR04 ECHO →  GPIO 12  (use a voltage divider if powering at 5 V)
```

> If you use different pins, update `#define TRIG_PIN` and `#define ECHO_PIN` at the top of `wheelson_explorer.ino`.

---

## Firmware Setup

### Prerequisites

1. **Arduino IDE 2.x** — [download](https://www.arduino.cc/en/software)
2. **ESP32 board support** — in Arduino IDE: *File → Preferences → Additional boards manager URLs*, add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
   Then *Tools → Board → Boards Manager* → search **esp32** → install **esp32 by Espressif Systems**.
3. **ArduinoJson library** — *Sketch → Include Library → Manage Libraries* → search **ArduinoJson** → install.

### Flashing

1. Open `firmware/wheelson_explorer/wheelson_explorer.ino` in Arduino IDE.
2. Edit the Wi-Fi credentials at the top of the file:
   ```cpp
   #define WIFI_SSID     "YourNetworkName"
   #define WIFI_PASSWORD "YourPassword"
   ```
3. Connect Wheelson via USB.
4. Select:
   - **Board**: `ESP32 Dev Module`
   - **Partition Scheme**: `Huge APP (3MB No OTA / 1MB SPIFFS)`
   - **Port**: the `/dev/cu.usbserial-XXXX` port that appears
5. Click **Upload** (→). Compilation takes ~60 s.
6. Open **Serial Monitor** at **115200 baud**. You should see:
   ```
   [WIFI] Connected! IP: 192.168.1.105
   [HTTP] Server started on port 80
   ```

### Test the firmware

```bash
# Check the camera
open http://192.168.1.105/camera

# Check status JSON
curl http://192.168.1.105/status

# Send a test move command
curl -X POST http://192.168.1.105/move \
  -H "Content-Type: application/json" \
  -d '{"action":"forward","duration_ms":500}'
```

---

## Middleware Setup

### Prerequisites

Python 3.10+, and a [Google AI Studio API key](https://aistudio.google.com/app/apikey).

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
WHEELSON_IP=192.168.1.105        # IP from Serial Monitor
GEMINI_API_KEY=AIza...           # your Gemini key
LOOP_INTERVAL_SEC=4.0            # seconds between VLM cycles
PORT=8000
```

### Run

```bash
# Choose a personality: benson | sir_david | klaus | zog7
python main.py --personality sir_david
```

The dashboard will be live at **http://localhost:8000**.

---

## Personalities

| Key | Name | Character | Movement |
|-----|------|-----------|----------|
| `benson` | BENSON 🦺 | Health & Safety Inspector | Methodical, centre-of-path |
| `sir_david` | Sir David 🎙️ | Nature documentary narrator | Curiosity-driven, seeks points of interest |
| `klaus` | Klaus 🎨 | Interior designer (appalled) | Perimeter-hugging, wide turns |
| `zog7` | Zog-7 👾 | Alien scout | Stealthy, wall-hugging, hides from humans |

Each personality gets a custom **system prompt** injected into every Gemini request. The VLM returns:

```json
{
  "action": "forward",
  "duration_ms": 400,
  "thought": "A suspiciously low-hanging cable. Classic occupational hazard. Logging violation."
}
```

---

## Safety

- **Firmware hard-stop**: the ESP32 checks the distance sensor *before* driving. If distance < 10 cm and action is `forward`, it substitutes `stop` with a serial log entry.
- **Middleware hard-stop**: Python applies the same guard *before* sending the `/move` request. Both layers act independently.
- The dashboard distance meter flashes **red** and logs `⚠ HARD STOP` when triggered.

---

## Dashboard

| Element | Description |
|---------|-------------|
| 📷 Camera feed | Live JPEG updated every cycle |
| 💬 Thought bubble | VLM narration with typewriter animation |
| 🟢 Action badge | Color-coded: green=forward, red=stop, yellow=turn, orange=backward |
| 📏 Distance meter | Bar + value; red when < 10 cm |
| 📜 Event log | Rolling log of last 20 thoughts + actions |

---

## API Reference

### `GET /camera`
Returns a raw JPEG frame.  
Response header `X-Distance-CM`: current ultrasonic sensor reading in cm.

### `GET /status`
```json
{ "distance_cm": "42.6", "ip": "192.168.1.105" }
```

### `POST /move`
```json
{ "action": "forward|backward|left|right|stop", "duration_ms": 500 }
```
Response:
```json
{ "ok": true, "action": "forward", "duration_ms": 500, "distance_cm": "42.6", "safety": false }
```

---

## License

MIT
