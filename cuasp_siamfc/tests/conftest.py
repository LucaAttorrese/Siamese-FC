"""Shared fixtures for the C-UASP test suite.

Nothing in this suite touches physical camera or PCMM/KAS hardware: vision
tests use synthetic data (real cv2.aruco-rendered images or hand-built
dataclasses), and motor-protocol tests run against `FakePCMMServer`, an
in-process asyncio TCP server speaking the exact wire protocol documented in
`control/kas/README.md`, on `127.0.0.1`.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

import pytest


# ============================================================================
# Fake PCMM TCP server
# ============================================================================


@dataclass
class FakePCMMServer:
    """In-process stand-in for the KAS TCP server, speaking the real protocol.

    Records every line received so tests can assert on exactly what a motor
    adapter sent (e.g. one atomic `abs[0,1]=...` line, never two separate
    axis writes). Encoder/status replies are configurable so tests can drive
    specific `PCMMStatus` scenarios without needing a real controller.
    """

    host: str = "127.0.0.1"
    encoder_values: dict = field(default_factory=lambda: {0: 0.0, 1: 0.0})
    # 15-field position-v1 schema by default; see motors.py _parse_status_reply.
    status_fields: list = field(default_factory=lambda: [
        0, 1, "TRUE", "TRUE", "TRUE", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "FALSE", "FALSE",
    ])
    drop_next_reply: bool = False
    reply_delay_s: float = 0.0

    def __post_init__(self) -> None:
        self.port: int | None = None
        self.received_lines: list[str] = []
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                text = raw.decode("ascii", errors="replace").strip()
                if not text:
                    continue
                self.received_lines.append(text)

                if text.startswith("abs[0,1]=") or text.startswith("stop[0]="):
                    continue  # fire-and-forget, no reply expected by the client

                if text.startswith("enc[") and text.endswith("]?"):
                    axes_text = text[len("enc["):-len("]?")]
                    axes = [int(a.strip()) for a in axes_text.split(",")]
                    reply = "enc[{}]={}\n".format(
                        ",".join(str(a) for a in axes),
                        ",".join(str(self.encoder_values[a]) for a in axes),
                    )
                    await self._maybe_reply(writer, reply)
                    continue

                if text == "status?":
                    reply = "status=" + ",".join(str(v) for v in self.status_fields) + "\n"
                    await self._maybe_reply(writer, reply)
                    continue
                # Unknown request: no reply, matching a real PCMM ignoring
                # anything outside its supported command set.
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()

    async def _maybe_reply(self, writer: asyncio.StreamWriter, reply: str) -> None:
        if self.drop_next_reply:
            self.drop_next_reply = False
            return  # simulate a query that never gets a reply (timeout path)
        if self.reply_delay_s > 0.0:
            await asyncio.sleep(self.reply_delay_s)
        writer.write(reply.encode("ascii"))
        await writer.drain()


@pytest.fixture
async def fake_pcmm_server():
    server = FakePCMMServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


# ============================================================================
# cuasp.py loader (it is a top-level script under control/, not a package
# member of suncubes -- `pythonpath = ["control"]` in pyproject.toml makes a
# plain `import cuasp` work directly since control/ is on sys.path).
# ============================================================================


@pytest.fixture
def cuasp_module():
    import cuasp  # noqa: PLC0415 (import inside fixture keeps sys.path timing correct)
    return cuasp


def assert_finite(value: float) -> None:
    assert math.isfinite(value), f"expected a finite value, got {value!r}"
