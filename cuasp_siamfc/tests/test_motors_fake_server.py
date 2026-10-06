"""Integration tests for PCMMMotorAdapterTCP against FakePCMMServer.

Nothing here needs real PCMM/KAS hardware: FakePCMMServer (tests/conftest.py)
speaks the exact wire protocol control/kas/README.md documents, over a real
TCP socket on 127.0.0.1. This is the only way to meaningfully test the
connection/timeout/reconnect behavior that currently has zero coverage.
"""

import asyncio

import pytest

from suncubes.motors import PCMMMotorAdapterTCP


async def _wait_until(predicate, timeout_s=1.0, interval_s=0.01):
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


@pytest.fixture
async def adapter(fake_pcmm_server):
    a = PCMMMotorAdapterTCP(
        fake_pcmm_server.host,
        fake_pcmm_server.port,
        connect_timeout_s=1.0,
        io_timeout_s=0.3,
    )
    yield a
    try:
        await a.close_motors(send_stop=False)
    except Exception:
        pass


async def test_connect_succeeds_against_fake_server(adapter):
    connected = await adapter.connect_motors_async()
    assert connected is True
    assert adapter.isOK is True


async def test_abs_rotate_sends_one_atomic_line(adapter, fake_pcmm_server):
    await adapter.connect_motors_async()
    await adapter.abs_rotate(10.0, 5.0)
    assert await _wait_until(lambda: len(fake_pcmm_server.received_lines) >= 1)
    assert fake_pcmm_server.received_lines[-1] == "abs[0,1]=10.000000,5.000000"


async def test_abs_rotate_clamps_to_the_configured_envelope(fake_pcmm_server):
    a = PCMMMotorAdapterTCP(
        fake_pcmm_server.host, fake_pcmm_server.port,
        connect_timeout_s=1.0, io_timeout_s=0.3,
        axis1_position_min_deg=-29.5, axis1_position_max_deg=29.5,
        axis2_position_min_deg=-19.5, axis2_position_max_deg=19.5,
    )
    await a.connect_motors_async()
    await a.abs_rotate(999.0, -999.0)
    assert await _wait_until(lambda: len(fake_pcmm_server.received_lines) >= 1)
    assert fake_pcmm_server.received_lines[-1] == "abs[0,1]=29.500000,-19.500000"
    await a.close_motors(send_stop=False)


async def test_stop_sends_stop_command(adapter, fake_pcmm_server):
    await adapter.connect_motors_async()
    await adapter.stop()
    assert await _wait_until(lambda: "stop[0]=1" in fake_pcmm_server.received_lines)


async def test_get_status_round_trips_15_field_schema(adapter, fake_pcmm_server):
    fake_pcmm_server.status_fields = [
        3, 9, "TRUE", "TRUE", "TRUE", 0.0, 0.0, 1.0, 2.0, 1.01, 1.99, 0.01, 0.002, "FALSE", "FALSE",
    ]
    await adapter.connect_motors_async()
    status = await adapter.get_status()
    assert status.controller_state == 3
    assert status.position_reference_sequence == 9
    assert status.axis1_reference_deg == 1.0
    assert status.axis2_actual_deg == 1.99


async def test_get_encoder_positions_round_trip(adapter, fake_pcmm_server):
    fake_pcmm_server.encoder_values = {0: 12.5, 1: -3.75}
    await adapter.connect_motors_async()
    positions = await adapter.get_encoder_positions()
    assert positions == (12.5, -3.75)


async def test_query_timeout_forces_reconnect_not_desync(adapter, fake_pcmm_server):
    """A query that never gets a reply must not leave the client waiting to
    match a later reply to the wrong request -- the documented safety
    property from control/kas/README.md. The adapter should mark itself
    disconnected and a fresh connect_motors_async() should recover cleanly."""
    await adapter.connect_motors_async()
    fake_pcmm_server.drop_next_reply = True

    with pytest.raises(TimeoutError):
        await adapter.get_status()
    assert adapter.isOK is False

    reconnected = await adapter.connect_motors_async()
    assert reconnected is True
    status = await adapter.get_status()
    assert status is not None  # a normal reply is now correctly matched again
