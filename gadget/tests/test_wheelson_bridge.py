"""Tests for wheelson_bridge against a fake middleware.

The fake reproduces the real wheelson-explorer middleware semantics that
the bridge depends on (from middleware/remote.py + routes.py):
  * POST /queue/join {name} -> {token, position, budget_sec}; 409 on dup name
  * POST /control {token, direction} -> 403 not_your_turn / session_expired,
    400 invalid_direction
  * POST /queue/leave {token} -> {ok, removed}; leaving as controller releases
  * POST /persona {persona} -> 400 invalid_persona with valid list
  * POST /chat {message, name} -> {response, personality}
  * GET  /health, GET /queue/status, GET /snapshot (JPEG bytes / 503)

Run:  python3 -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FAKE_JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"
PERSONAS = ("benson", "sir_david", "klaus", "zog7")
ACTIONS = ("forward", "backward", "left", "right", "stop")


class FakeMiddleware(BaseHTTPRequestHandler):
    queue = []          # list of {"token","name"}
    controls = []       # list of directions received, in order
    persona = "benson"
    token_counter = 0

    def log_message(self, *args):  # silence
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {
                "status": "ok", "personality": self.persona,
                "personality_name": "Benson", "cycle_count": 42,
                "last_action": "forward", "last_thought": "Clear ahead",
                "last_distance_cm": 87.0, "extra_field": "should be trimmed",
            })
        elif self.path == "/queue/status":
            self._json(200, {
                "queue": [{"name": s["name"], "position": i + 1, "remaining_s": 60.0}
                          for i, s in enumerate(self.queue)],
                "budget_sec": 60.0,
            })
        elif self.path == "/snapshot":
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(FAKE_JPEG)))
            self.end_headers()
            self.wfile.write(FAKE_JPEG)
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self):
        body = self._body()
        if self.path == "/queue/join":
            name = (body.get("name") or "").strip()
            if any(s["name"] == name for s in self.queue):
                return self._json(409, {"error": "already_in_queue"})
            FakeMiddleware.token_counter += 1
            token = f"tok-{FakeMiddleware.token_counter}"
            self.queue.append({"token": token, "name": name})
            self._json(200, {"token": token, "position": len(self.queue), "budget_sec": 60.0})
        elif self.path == "/control":
            if not self.queue or self.queue[0]["token"] != body.get("token"):
                return self._json(403, {"error": "not_your_turn"})
            if body.get("direction") not in ACTIONS:
                return self._json(400, {"error": "invalid_direction"})
            self.controls.append(body["direction"])
            self._json(200, {"ok": True})
        elif self.path == "/queue/leave":
            before = len(self.queue)
            self.queue[:] = [s for s in self.queue if s["token"] != body.get("token")]
            self._json(200, {"ok": True, "removed": before - len(self.queue)})
        elif self.path == "/persona":
            if body.get("persona") not in PERSONAS:
                return self._json(400, {"error": "invalid_persona", "valid": list(PERSONAS)})
            FakeMiddleware.persona = body["persona"]
            self._json(200, {"ok": True, "persona": body["persona"]})
        elif self.path == "/chat":
            if not (body.get("message") or "").strip():
                return self._json(400, {"error": "empty message"})
            self._json(200, {"response": "I see a clear floor ahead.", "personality": self.persona})
        else:
            self._json(404, {"error": "not_found"})


class BridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeMiddleware)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["WHEELSON_MIDDLEWARE_URL"] = f"http://127.0.0.1:{cls.port}"
        os.environ["WHEELSON_STATE_DIR"] = os.path.join(cls.tmp.name, "state")
        os.environ["WHEELSON_SNAPSHOT_DIR"] = os.path.join(cls.tmp.name, "snaps")

        # Import AFTER env is set (bridge reads env at import time).
        import wheelson_bridge
        cls.bridge = wheelson_bridge

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

    def setUp(self):
        FakeMiddleware.queue = []
        FakeMiddleware.controls = []
        FakeMiddleware.persona = "benson"
        self.bridge._clear_token()

    def test_health_trims_fields(self):
        r = self.bridge.run_command("health", {})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["payload"]["personality"], "benson")
        self.assertNotIn("extra_field", r["payload"])

    def test_persona_valid_and_invalid(self):
        r = self.bridge.run_command("persona", {"persona": "klaus"})
        self.assertTrue(r["ok"], r)
        r = self.bridge.run_command("persona", {"persona": "nope"})
        self.assertFalse(r["ok"])

    def test_chat(self):
        r = self.bridge.run_command("chat", {"message": "What do you see?"})
        self.assertTrue(r["ok"], r)
        self.assertIn("clear floor", r["payload"]["response"])

    def test_drive_pulses_then_stops_and_reuses_token(self):
        r = self.bridge.run_command("drive", {"direction": "forward", "repeats": 3})
        self.assertTrue(r["ok"], r)
        # fresh join (no probe) + 3 pulses + final stop
        self.assertEqual(FakeMiddleware.controls, ["forward", "forward", "forward", "stop"])
        token_file = self.bridge._state_path()
        self.assertTrue(token_file.exists())
        self.assertEqual(oct(token_file.stat().st_mode)[-3:], "600")
        # Second drive reuses the cached token: probe stop + pulse + final stop,
        # and no second join (queue still holds exactly one session)
        FakeMiddleware.controls = []
        r = self.bridge.run_command("drive", {"direction": "left", "repeats": 1})
        self.assertTrue(r["ok"], r)
        self.assertEqual(FakeMiddleware.controls, ["stop", "left", "stop"])
        self.assertEqual(len(FakeMiddleware.queue), 1)

    def test_drive_rejects_bad_input_locally(self):
        self.assertFalse(self.bridge.run_command("drive", {"direction": "spin"})["ok"])
        self.assertFalse(self.bridge.run_command("drive", {"direction": "forward", "repeats": 9})["ok"])
        self.assertEqual(FakeMiddleware.controls, [])

    def test_leave_clears_state(self):
        self.bridge.run_command("drive", {"direction": "stop"})
        r = self.bridge.run_command("leave", {})
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["payload"]["left"])
        self.assertFalse(self.bridge._state_path().exists())
        self.assertEqual(FakeMiddleware.queue, [])

    def test_snapshot_saves_file(self):
        r = self.bridge.run_command("snapshot", {})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["payload"]["bytes"], len(FAKE_JPEG))
        with open(r["payload"]["path"], "rb") as f:
            self.assertEqual(f.read(), FAKE_JPEG)

    def test_unknown_command(self):
        self.assertFalse(self.bridge.run_command("explode", {})["ok"])


if __name__ == "__main__":
    unittest.main()
