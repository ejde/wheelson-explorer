# Wheelson Hybrid Gadget

Put your Muse on Wheelson **without touching the Wheelson firmware**.

A Linux box on the same LAN as Wheelson runs both the existing
wheelson-explorer middleware and the Muse Gadget Linux SDK. The gadget
registers seven narrow `wheelson.*` commands; each one calls the
middleware's HTTP API on localhost. Your Muse never talks to the ESP32
directly, so the explorer's safety contract survives intact:

```
You ⇄ Muse app / chat ⇄ your Muse (cloud)
                              │  link.invoke: wheelson.*
                              ▼
                   musegadget service (Linux host)
                              │  wheelson_bridge.py (run-as account)
                              ▼
              POST/GET http://127.0.0.1:8000  (middleware)
                              │
        S3 persona · S5 queue · S1 MotionSupervisor (single writer)
                              ▼  Protocol 2 (session / sequence / lease)
                   S0 firmware safety (ESP32, unchanged)
                              ▼
                           motors
```

## Commands your Muse gets

The gadget joins the remote-control queue and chats under the name in
`WHEELSON_GADGET_NAME` (default `Muse`). Set it to your own Muse's name
so the dashboard shows who's driving.

| Command | Middleware call | Notes |
|---|---|---|
| `wheelson.health` | `GET /health` | Trimmed status: persona, action, thought, distance, nav, strategy |
| `wheelson.queue_status` | `GET /queue/status` | Who's driving, remaining budgets |
| `wheelson.persona` | `POST /persona` | `benson \| sir_david \| klaus \| zog7`, live switch |
| `wheelson.chat` | `POST /chat` | In-persona reply; middleware rate-limits 5/min |
| `wheelson.drive` | `POST /queue/join` + `/control` | Bounded pulse train (1–5 pulses, 350 ms apart), then explicit `stop` |
| `wheelson.leave` | `POST /queue/leave` | Hands control back to autonomy |
| `wheelson.snapshot` | `GET /snapshot` | Saves JPEG on the host, returns path+size; read back via `file.read` (64 KB chunks — the link caps single messages) |

Driving semantics matter: a `drive` call is **not** sustained motion.
It's the dashboard's key-hold cadence packaged as one invocation, and it
always ends in `stop`. Underneath, the middleware renewal window (0.7 s),
the 3 s inactivity hand-back, the 60 s queue budget, and the firmware's
750 ms lease watchdog + visual hard-stop all still apply.

## Files here

- `wheelson_bridge.py` — stdlib-only bridge + CLI. Token state at
  `~/.wheelson-gadget/session.json` (0600). Tested.
- `executor_additions.py` — the exact `COMMAND_SPECS` entries + `Executor`
  branch/method to merge into the SDK's `linux/src/musegadget/executor.py`,
  per the SDK's own AGENTS.md "Adding a command" recipe. Additions, not a
  fork, so upstream updates stay mergeable.
- `tests/test_wheelson_bridge.py` — 8 tests against a fake middleware that
  reproduces the real endpoints' validation (`not_your_turn`,
  `invalid_direction`, `invalid_persona`, dup-name 409…). All passing:
  `python3 -m unittest discover -s tests -v`

## Host: Raspberry Pi 4 (chosen 2026-10-20)

A Raspberry Pi 4 is the reference host. It is on the SDK's supported list
and its built-in Bluetooth LE is all pairing needs.

- **OS:** Raspberry Pi OS Bookworm (64-bit) — ships Python 3.11 for the
  middleware and is a supported gadget OS (Bullseye or later).
- **Network:** same LAN as Wheelson. Wired ethernet preferred for a
  stable middleware → Wheelson link; BLE pairing works either way, and
  during pairing the app offers "the current connection" — the Wi-Fi
  fields it sends are ignored because the Pi is already online.
- **Bluetooth LE is pairing-only.** The gadget pairs with the Muse phone
  app over BLE (ECDH/AES-GCM session setup, app confirmation replaces a
  physical button — a Pi has none). After pairing it operates over the
  network via an encrypted Noise session and stays connected across
  reboots. BLE advertises again only if you run `sudo musegadget pair`.
- Python 3.11+ for the middleware; the gadget installer wants a
  sudo-capable account *to install*, but see the security setup below.
- Wheelson powered on and reachable at its LAN IP.

## Security setup (do this part exactly)

The gadget service runs commands as its run-as account — *that account's
access is Muse's access*, and the stock SDK also exposes
`system.run` / `file.read` / `file.write`. So:

1. Create a dedicated account with **no sudo**: `sudo adduser wheelson`
   (skip sudo group membership). Install with `--run-as wheelson`.
2. That account's home holds only: the wheelson-explorer checkout, the
   gadget state dir, and snapshots. Nothing else lives there.
3. The SDK token (`mgst_…`) is stored by the installer at
   `/var/lib/musegadget/sdk_token`, root-only. The bridge's queue token is
   a separate, short-lived middleware token (60 s budget) in a 0600 file.
4. Pair on a trusted network: community pairing has no manufacturer
   verification (SDK's own warning).

## Install runbook

Run on the Pi 4. Read every script before running it.

Prep (once): flash Raspberry Pi OS Bookworm 64-bit with Raspberry Pi
Imager (set hostname, enable SSH, join your LAN), plug the Pi into wired
ethernet if you can, boot it, and `sudo apt update && sudo apt full-upgrade`.

```bash
# 1. Host account + middleware
sudo adduser wheelson            # no sudo for this account
sudo -iu wheelson
git clone https://github.com/ejde/wheelson-explorer.git
cd wheelson-explorer/middleware
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env             # set WHEELSON_IP to Wheelson's LAN IP + your VLM key
.venv/bin/python main.py --personality benson   # smoke test; dashboard on :8000
curl http://127.0.0.1:8000/health               # expect "status":"ok"
# (run it under systemd for daily use — a unit wrapping the line above)

# 2. Gadget SDK with the Wheelson commands merged in
cd ~
git clone https://github.com/facebookincubator/muse-gadget-sdk.git
#   merge executor_additions.py into muse-gadget-sdk/linux/src/musegadget/executor.py:
#   COMMAND_SPECS entries + dispatch branch + wheelson() method

# 3. SDK token: gadgets.muse.ai -> Account -> SDK tokens (mgst_…)
#    Terms are personal / non-commercial; one token covers up to 50 devices.

# 4. Install + pair (as your sudo account, not wheelson)
cd ~/muse-gadget-sdk/linux
curl -fsSL https://raw.githubusercontent.com/facebookincubator/muse-gadget-sdk/main/linux/install.sh -o /tmp/install.sh
less /tmp/install.sh
bash install.sh --from . --sdk-token mgst_… --run-as wheelson
sudo cp <wheelson-explorer-checkout>/gadget/wheelson_bridge.py /opt/musegadget/
sudo systemctl restart musegadget
```

Pairing (installer opens a 10-minute window; rerun `sudo musegadget pair`
if it lapses):

1. Muse app on your phone: **Settings > Devices > Developer mode** on.
2. **Settings > Devices > Add Device** (the **+**). Pick `MuseGadgetXXXXXX`
   (Linux gadgets have no hyphen in the name).
3. Accept the community-device warning; for Wi-Fi choose the current
   connection — the host is already online, no password needed.

Verify:

```bash
sudo musegadget info                              # pairing state
sudo journalctl -u musegadget -f                  # look for: registered with the Muse
```

Then, in chat with your Muse, in order: "check the Wheelson gadget health" →
"what's Wheelson's status" (`wheelson.health`) → "switch Wheelson to
klaus" → "take a snapshot" → "drive forward one pulse" → "leave the
queue". First drive should be somewhere with clear floor in front and a
hand near the robot.

## Bonus: Wheelson can message you

The Linux SDK ships `musegadget send-user-msg` — local programs can post
into your Muse chat with no credentials of their own:

```bash
musegadget send-user-msg "Wheelson hit a hard stop in the hallway."
```

Worth wiring later to middleware events (hard-stop, recovery escalation,
low battery if telemetry ever exposes it), optionally with
`--session-id <fixed-uuid>` to keep robot notes in their own side chat.

## Troubleshooting

- Pairing dies at Wi-Fi with "Couldn't connect": the installer sets
  BlueZ `ExchangeMTU = 256`; if it's missing, that's the symptom.
- Phone shows GATT status 133: stale phone Bluetooth state — toggle the
  phone's Bluetooth off/on and pair again.
- `middleware unreachable` from any `wheelson.*` command:
  `curl http://127.0.0.1:8000/health` on the host; the middleware isn't up.
- Commands missing in chat: `sudo systemctl restart musegadget` — new
  commands only appear when the service re-registers.

## Limits to know

- SDK is experimental, not a supported product; Meta can change/withdraw
  access. The bridge degrades gracefully: lose the gadget and the
  dashboard + autonomy are untouched.
- `wheelson.chat` inherits the middleware's 5-requests/minute limit.
- Snapshot JPEGs don't ride back in the command result (link message
  cap); they're files on the host by design.
