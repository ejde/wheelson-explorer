"""
wheelson-explorer · middleware · test_remote_queue.py
Tests for the 2026-10-25 fixes: queue pruning of expired sessions and
the raised motion-request timeout.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import config
import remote
import state as state_module
from remote import JoinQueueRequest, queue_join, queue_leave, queue_status
from state import ControlSession, _control_queue


def _body(resp):
    return json.loads(resp.body.decode())


class TestQueuePruning(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _control_queue.clear()
        self._saved_supervisor = state_module._motion_supervisor
        state_module._motion_supervisor = None

    async def asyncTearDown(self):
        _control_queue.clear()
        state_module._motion_supervisor = self._saved_supervisor

    def _seed(self, name, age_sec):
        loop = asyncio.get_running_loop()
        session = ControlSession(
            token=f"token-{name}-{age_sec}",
            name=name,
            joined_at=loop.time() - age_sec,
        )
        _control_queue.append(session)
        return session

    async def test_expired_session_pruned_on_join(self):
        self._seed("finch", config.CONTROL_BUDGET_SEC + 5)
        resp = await queue_join(JoinQueueRequest(name="finch"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(_body(resp)["position"], 1)
        self.assertEqual(len(_control_queue), 1)
        self.assertEqual(_control_queue[0].name, "finch")

    async def test_live_session_still_blocks_duplicate(self):
        self._seed("finch", 5)
        resp = await queue_join(JoinQueueRequest(name="finch"))
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(len(_control_queue), 1)

    async def test_status_prunes_expired(self):
        self._seed("old", config.CONTROL_BUDGET_SEC + 5)
        self._seed("live", 5)
        resp = await queue_status()
        names = [entry["name"] for entry in _body(resp)["queue"]]
        self.assertEqual(names, ["live"])

    async def test_prune_releases_expired_controller(self):
        self._seed("old", config.CONTROL_BUDGET_SEC + 5)
        supervisor = MagicMock()
        supervisor.release_remote = AsyncMock()
        state_module._motion_supervisor = supervisor
        resp = await queue_join(JoinQueueRequest(name="finch"))
        self.assertEqual(resp.status_code, 200)
        supervisor.release_remote.assert_awaited_once()

    async def test_leave_still_works(self):
        session = self._seed("finch", 5)
        from remote import LeaveQueueRequest

        resp = await queue_leave(LeaveQueueRequest(token=session.token))
        self.assertEqual(_body(resp)["removed"], 1)
        self.assertEqual(_control_queue, [])


class TestMotionTimeout(unittest.TestCase):
    def test_timeout_raised_but_under_firmware_lease(self):
        # Firmware motor lease is 750 ms; the request cap must stay under it
        # so a dead robot is still caught by the lease watchdog.
        self.assertEqual(config.MOTION_REQUEST_TIMEOUT_SEC, 0.6)
        self.assertLess(config.MOTION_REQUEST_TIMEOUT_SEC, 0.75)


class TestPostMoveDeadline(unittest.IsolatedAsyncioTestCase):
    async def test_post_move_uses_config_timeout(self):
        from motion import WheelsonClient

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True}
        http = MagicMock()
        http.post = AsyncMock(return_value=resp)
        client = WheelsonClient(http)
        await client._post_move({"command": "claim"})
        _, kwargs = http.post.call_args
        self.assertEqual(kwargs["timeout"], config.MOTION_REQUEST_TIMEOUT_SEC)


if __name__ == "__main__":
    unittest.main()
