"""Independent camera producer; consumers never wait on image acquisition.

Recovery behaviour (added after the 2026-10-24 run, where a dead camera
feed made the explorer loop log a traceback every 2 s forever):
- A single fetch is capped by CAMERA_FETCH_TIMEOUT_SEC, so a hung
  /camera request can never stall the producer task.
- Consecutive failures are counted and throttled-logged, and the live
  feed state is exposed via status() for /health.
- CameraOutage tracks a run of expired snapshots for the loop: it says
  "log" once when the feed is first lost, "restart" once the outage has
  lasted CAMERA_OUTAGE_RESTART_AFTER snapshots (the loop then restarts
  the producer), and stays quiet otherwise so a dead feed cannot spam
  the log. The robot stays held throughout; only a fresh frame ends
  the episode.
"""

import asyncio
import contextlib
import logging
import time

from config import (
    CAMERA_INTERVAL_SEC,
    PERCEPTION_MAX_AGE_SEC,
    CAMERA_FETCH_TIMEOUT_SEC,
    CAMERA_OUTAGE_LOG_INTERVAL_SEC,
)

log = logging.getLogger("wheelson")


class CameraStream:
    def __init__(self, client):
        self.client = client
        self.latest = None
        self.frame_id = 0
        self.task = None
        # Feed health, for /health and for the loop's outage handling.
        self.consecutive_failures = 0
        self.last_error = ""
        self.last_fetch_sec = 0.0
        self.last_success_ts = 0.0
        self.restart_count = 0
        self._last_failure_log_ts = 0.0

    async def start(self):
        await self._capture()
        self.task = asyncio.create_task(self._run())

    async def close(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None

    async def restart(self):
        """Drop the current producer task and start a fresh one.

        Used when snapshots stay expired even though the producer task
        should be retrying: a fresh task heals a producer stuck inside a
        hung fetch or wedged client state.
        """
        await self.close()
        self.task = asyncio.create_task(self._run())
        self.restart_count += 1

    async def _capture(self):
        # Conservative age includes request/capture/transfer latency.
        captured_at = asyncio.get_running_loop().time()
        fetch_start = asyncio.get_running_loop().time()
        jpeg, telemetry = await self.client.fetch_frame()
        self.last_fetch_sec = asyncio.get_running_loop().time() - fetch_start
        self.frame_id += 1
        self.latest = (jpeg, dict(telemetry, frame_id=self.frame_id, captured_at=captured_at))
        self.consecutive_failures = 0
        self.last_error = ""
        self.last_success_ts = asyncio.get_running_loop().time()

    def _note_failure(self, message):
        self.consecutive_failures += 1
        self.last_error = message
        now = time.monotonic()
        if now - self._last_failure_log_ts >= CAMERA_OUTAGE_LOG_INTERVAL_SEC:
            self._last_failure_log_ts = now
            log.warning(
                "Camera acquisition failed (%d consecutive): %s",
                self.consecutive_failures,
                message,
            )

    async def _run(self):
        while True:
            await asyncio.sleep(CAMERA_INTERVAL_SEC)
            try:
                await asyncio.wait_for(self._capture(), timeout=CAMERA_FETCH_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                self._note_failure(f"fetch timed out after {CAMERA_FETCH_TIMEOUT_SEC}s")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._note_failure(str(exc))

    def status(self):
        """Cheap, JSON-safe snapshot of feed health for /health."""
        return {
            "frame_id": self.frame_id,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_fetch_sec": round(self.last_fetch_sec, 3),
            "restart_count": self.restart_count,
            "has_frame": self.latest is not None,
        }

    def snapshot(self, now):
        if self.latest is None or now - self.latest[1]["captured_at"] > PERCEPTION_MAX_AGE_SEC:
            raise RuntimeError("Camera observation expired")
        jpeg, telemetry = self.latest
        if now - telemetry["captured_at"] + telemetry.get("analysis_age_s", 0.0) > PERCEPTION_MAX_AGE_SEC:
            raise RuntimeError("Obstacle analysis expired")
        return jpeg, dict(telemetry)


class CameraOutage:
    """One feed-loss episode: log once, restart once, then stay quiet.

    The loop feeds it every expired snapshot; it returns "log" when the
    episode starts (and at most every log_interval_sec after that),
    "restart" once when the episode reaches restart_after snapshots,
    and None for the quiet cycles in between. note_recovered() ends the
    episode and reports whether one was in progress.
    """

    def __init__(self, restart_after, log_interval_sec=CAMERA_OUTAGE_LOG_INTERVAL_SEC):
        self.restart_after = restart_after
        self.log_interval_sec = log_interval_sec
        self.active = False
        self.count = 0
        self.restarted = False
        self._last_log_ts = 0.0

    def note_expired(self, now_ts):
        self.count += 1
        if not self.active:
            self.active = True
            self._last_log_ts = now_ts
            return "log"
        if not self.restarted and self.count >= self.restart_after:
            self.restarted = True
            self._last_log_ts = now_ts
            return "restart"
        if now_ts - self._last_log_ts >= self.log_interval_sec:
            self._last_log_ts = now_ts
            return "log"
        return None

    def note_recovered(self):
        was_active = self.active
        self.active = False
        self.count = 0
        self.restarted = False
        self._last_log_ts = 0.0
        return was_active
