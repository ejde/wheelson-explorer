"""Independent camera producer; consumers never wait on image acquisition."""
import asyncio
import contextlib
import logging
from config import CAMERA_INTERVAL_SEC, PERCEPTION_MAX_AGE_SEC


class CameraStream:
    def __init__(self, client):
        self.client = client
        self.latest = None
        self.frame_id = 0
        self.task = None

    async def start(self):
        await self._capture()
        self.task = asyncio.create_task(self._run())

    async def close(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def _capture(self):
        # Conservative age includes request/capture/transfer latency.
        captured_at = asyncio.get_running_loop().time()
        jpeg, telemetry = await self.client.fetch_frame()
        self.frame_id += 1
        self.latest = (jpeg, dict(telemetry, frame_id=self.frame_id, captured_at=captured_at))

    async def _run(self):
        while True:
            await asyncio.sleep(CAMERA_INTERVAL_SEC)
            try:
                await self._capture()
            except Exception as exc:
                logging.getLogger("wheelson").warning("Camera acquisition failed: %s", exc)

    def snapshot(self, now):
        if self.latest is None or now - self.latest[1]["captured_at"] > PERCEPTION_MAX_AGE_SEC:
            raise RuntimeError("Camera observation expired")
        jpeg, telemetry = self.latest
        if now - telemetry["captured_at"] + telemetry.get("analysis_age_s", 0.0) > PERCEPTION_MAX_AGE_SEC:
            raise RuntimeError("Obstacle analysis expired")
        return jpeg, dict(telemetry)
