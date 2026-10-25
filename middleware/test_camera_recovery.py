"""
wheelson-explorer · middleware · test_camera_recovery.py
Tests for the camera-feed recovery fix (2026-10-24 run: a dead feed made
the loop log a traceback every 2 s forever with no recovery attempt).
"""

import asyncio
import unittest
from unittest.mock import patch

import perception
from perception import CameraStream, CameraOutage


class FakeClient:
    """fetch_frame script: a list of outcomes ('ok', 'fail', 'hang')."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def fetch_frame(self):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if outcome == "fail":
            raise RuntimeError("camera wedged")
        if outcome == "hang":
            await asyncio.sleep(3600)
        return b"fake-jpeg", {"distance_cm": 999.0, "analysis_age_s": 0.0}


class TestCameraStreamRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_hung_fetch_is_capped_and_task_survives(self):
        stream = CameraStream(FakeClient(["hang"]))
        with patch.object(perception, "CAMERA_FETCH_TIMEOUT_SEC", 0.05), patch.object(
            perception, "CAMERA_INTERVAL_SEC", 0.01
        ):
            stream.task = asyncio.create_task(stream._run())
            await asyncio.sleep(0.3)
            self.assertGreaterEqual(stream.consecutive_failures, 1)
            self.assertIn("timed out", stream.last_error)
            self.assertFalse(stream.task.done(), "producer task must survive a hung fetch")
            await stream.close()

    async def test_transient_failures_recover(self):
        stream = CameraStream(FakeClient(["fail", "fail", "ok"]))
        with patch.object(perception, "CAMERA_FETCH_TIMEOUT_SEC", 1.0), patch.object(
            perception, "CAMERA_INTERVAL_SEC", 0.01
        ):
            stream.task = asyncio.create_task(stream._run())
            await asyncio.sleep(0.3)
            self.assertIsNotNone(stream.latest)
            self.assertEqual(stream.consecutive_failures, 0)
            self.assertGreaterEqual(stream.frame_id, 1)
            await stream.close()

    async def test_restart_replaces_task(self):
        stream = CameraStream(FakeClient(["ok"]))
        with patch.object(perception, "CAMERA_INTERVAL_SEC", 0.01):
            stream.task = asyncio.create_task(stream._run())
            await asyncio.sleep(0.05)
            old_task = stream.task
            await stream.restart()
            self.assertEqual(stream.restart_count, 1)
            self.assertTrue(old_task.done())
            self.assertIsNot(stream.task, old_task)
            self.assertFalse(stream.task.done())
            await stream.close()

    def test_snapshot_still_expires_on_stale_frame(self):
        stream = CameraStream(FakeClient(["ok"]))
        stream.latest = (b"jpeg", {"captured_at": 100.0, "frame_id": 1})
        with self.assertRaises(RuntimeError) as ctx:
            stream.snapshot(100.0 + 0.8 + 0.01)
        self.assertIn("expired", str(ctx.exception))
        jpeg, telemetry = stream.snapshot(100.0 + 0.1)
        self.assertEqual(jpeg, b"jpeg")

    def test_status_shape(self):
        stream = CameraStream(FakeClient(["ok"]))
        status = stream.status()
        for key in (
            "frame_id",
            "consecutive_failures",
            "last_error",
            "last_fetch_sec",
            "restart_count",
            "has_frame",
        ):
            self.assertIn(key, status)


class TestCameraOutage(unittest.TestCase):
    def test_sequence_log_restart_quiet_log_recovered(self):
        outage = CameraOutage(restart_after=5, log_interval_sec=30.0)
        self.assertEqual(outage.note_expired(0.0), "log")
        self.assertEqual(outage.note_expired(1.0), None)
        self.assertEqual(outage.note_expired(2.0), None)
        self.assertEqual(outage.note_expired(3.0), None)
        self.assertEqual(outage.note_expired(4.0), "restart")
        self.assertEqual(outage.note_expired(5.0), None)
        # Past the log interval since the restart log line.
        self.assertEqual(outage.note_expired(40.0), "log")
        self.assertTrue(outage.note_recovered())
        self.assertFalse(outage.note_recovered())
        # A fresh episode starts with a log line again.
        self.assertEqual(outage.note_expired(50.0), "log")


if __name__ == "__main__":
    unittest.main()
