"""Stop-on-loss ordering of cuasp.MotorCommandDispatcher (cuasp_siamfc local change, 2026-11-11).

In C-UASP the stop on a tracked -> lost transition was a separate asyncio task created in
submit(), which raced with a send already in progress: the wire could carry `stop[0]=1`
followed by the in-flight `abs[0,1]=...`, cancelling the stop (found by
test_cuasp_siamfc_e2e.py, ~2 runs in 10). The worker task now sends the stop itself,
after any in-flight command (see PROVENANCE.md "Local changes").

These tests make the race deterministic with a motors fake whose coroutines can be held
at a gate: the loss is submitted exactly while the encoder read, or the abs_rotate, is in
progress. With the original create_task() stop, test_stop_follows_an_abs_rotate_in_flight
fails (the stop is recorded before the in-flight abs_rotate completes).
"""

from __future__ import annotations

import asyncio

import pytest


class GatedMotors:
    """Async motors fake: records calls; encoder read or abs_rotate can be held at a gate."""

    def __init__(self, hold: str | None = None):
        self.calls: list[tuple] = []
        self.hold = hold  # None | "enc" | "abs"
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def connect_motors(self) -> bool:
        return True

    async def get_encoder_positions(self) -> tuple[float, float]:
        self.calls.append(("enc",))
        if self.hold == "enc":
            self.entered.set()
            await self.gate.wait()
        return (0.0, 0.0)

    async def abs_rotate(self, az: float, po: float) -> None:
        self.calls.append(("abs_begin", az, po))
        if self.hold == "abs":
            self.entered.set()
            await self.gate.wait()
        self.calls.append(("abs_end", az, po))

    async def stop(self) -> None:
        self.calls.append(("stop",))

    async def disconnect_motors(self) -> None:
        pass


@pytest.fixture
def cuasp(cuasp_module, monkeypatch):
    monkeypatch.setattr(cuasp_module, "MOTOR_COMMAND_HZ", 50.0)
    return cuasp_module


def _ok(cuasp, az=1.0, po=0.5):
    return cuasp.CuasCorrection(reliable=True, reason="ok", d_az_motor_units=az, d_po_motor_units=po)


def _lost(cuasp):
    return cuasp.CuasCorrection(reliable=False, reason="siamfc_lost")


async def test_loss_during_encoder_read_drops_the_command_and_stops(cuasp):
    motors = GatedMotors(hold="enc")
    d = cuasp.MotorCommandDispatcher(motors)
    await d.start()
    try:
        d.submit(_ok(cuasp), frame_index=1, capture_ns=1)
        await asyncio.wait_for(motors.entered.wait(), 1.0)  # worker is inside the encoder read
        d.submit(_lost(cuasp), frame_index=2, capture_ns=2)  # the loss arrives now
        motors.gate.set()
        await asyncio.sleep(0.1)
        assert motors.calls == [("enc",), ("stop",)], motors.calls  # the superseded command never goes out
    finally:
        await d.stop()


async def test_stop_follows_an_abs_rotate_in_flight(cuasp):
    motors = GatedMotors(hold="abs")
    d = cuasp.MotorCommandDispatcher(motors)
    await d.start()
    try:
        d.submit(_ok(cuasp), frame_index=1, capture_ns=1)
        await asyncio.wait_for(motors.entered.wait(), 1.0)  # abs_rotate is being written
        d.submit(_lost(cuasp), frame_index=2, capture_ns=2)
        await asyncio.sleep(0.05)  # with a separate stop task, the stop would run now
        motors.gate.set()
        await asyncio.sleep(0.1)
        assert motors.calls == [("enc",), ("abs_begin", 1.0, 0.5), ("abs_end", 1.0, 0.5), ("stop",)], motors.calls
    finally:
        await d.stop()


async def test_idle_dispatcher_sends_the_stop_promptly(cuasp):
    motors = GatedMotors()
    d = cuasp.MotorCommandDispatcher(motors)
    await d.start()
    try:
        d.submit(_ok(cuasp), frame_index=1, capture_ns=1)
        await asyncio.sleep(0.1)
        d.submit(_lost(cuasp), frame_index=2, capture_ns=2)
        await asyncio.sleep(0.02)
        assert motors.calls[-1] == ("stop",)
        assert motors.calls.count(("stop",)) == 1
    finally:
        await d.stop()


async def test_reacquisition_right_after_a_loss_stops_first_then_resumes(cuasp):
    motors = GatedMotors()
    d = cuasp.MotorCommandDispatcher(motors)
    await d.start()
    try:
        d.submit(_ok(cuasp, 1.0, 0.0), frame_index=1, capture_ns=1)
        await asyncio.sleep(0.1)
        # loss and re-acquisition before the worker runs: the new valid command replaces
        # the invalid one in the single-slot queue, but the stop must still go out first
        d.submit(_lost(cuasp), frame_index=2, capture_ns=2)
        d.submit(_ok(cuasp, 2.0, 0.0), frame_index=3, capture_ns=3)
        await asyncio.sleep(0.1)
        tail = motors.calls[motors.calls.index(("stop",)):]
        assert tail == [("stop",), ("enc",), ("abs_begin", 2.0, 0.0), ("abs_end", 2.0, 0.0)], motors.calls
    finally:
        await d.stop()


async def test_repeated_unreliable_samples_stop_only_once(cuasp):
    motors = GatedMotors()
    d = cuasp.MotorCommandDispatcher(motors)
    await d.start()
    try:
        d.submit(_ok(cuasp), frame_index=1, capture_ns=1)
        await asyncio.sleep(0.1)
        for i in range(5):
            d.submit(_lost(cuasp), frame_index=2 + i, capture_ns=2 + i)
            await asyncio.sleep(0.01)
        assert motors.calls.count(("stop",)) == 1
    finally:
        await d.stop()


async def test_a_failing_stop_is_reported_not_raised(cuasp, capsys):
    motors = GatedMotors()

    async def broken_stop():
        raise ConnectionResetError("socket closed")

    motors.stop = broken_stop
    d = cuasp.MotorCommandDispatcher(motors)
    await d.start()
    try:
        d.submit(_ok(cuasp), frame_index=1, capture_ns=1)
        await asyncio.sleep(0.1)
        d.submit(_lost(cuasp), frame_index=2, capture_ns=2)
        await asyncio.sleep(0.05)
        # (status.last_reason is overwritten right after by the cancelled request's reason)
        assert d.status.connected is False and d.status.error_count == 1
        assert "ConnectionResetError" in d.status.last_error
        assert "stop on loss failed" in capsys.readouterr().out
    finally:
        await d.stop()
