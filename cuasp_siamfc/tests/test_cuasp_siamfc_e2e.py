"""End-to-end: ids_siamfc camera source -> CameraAPI -> cuasp.main() -> motor dispatcher -> FakePCMMServer.

Only the camera frames, the tracker and the detector are fakes (no IDS camera, no weights,
no GPU); the adapter, its state machine, the camera application service, cuasp's PBVS
mapping, the dispatcher and the TCP motor adapter are the real code. Scenario: the drone
is detected and confirmed, tracked to the upper right of the principal point, then
leaves the frame -> reliable abs[0,1] commands, then exactly the stop[0]=1 on loss.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import math
import time

import numpy as np
import pytest

from suncubes import camera as cam
from suncubes.siamfc.detector import Detection
from suncubes.siamfc_source import IdsSiamFCCorrectionSourceAdapter

W, H = 640, 480
K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
CALIB = {"camera_matrix": K, "dist_coeffs": np.zeros(5), "R_lc": np.eye(3), "t_lc_m": np.zeros((3, 1)),
         "image_width": W, "image_height": H}
TRACKED = (420.0, 140.0)  # 100 px right of and above the principal point
LOSE_AFTER_S = 0.6


class ContinuousProducer:
    """A camera that always has a new frame: every latest() call is a new capture."""

    def __init__(self):
        self.seq = 0
        self.gray = np.zeros((H, W), np.uint8)

    def start(self):
        pass

    def stop(self):
        pass

    def latest(self):
        self.seq += 1
        return cam._FramePacket(gray=self.gray, ts_capture_ns=time.monotonic_ns(), sequence=self.seq)

    def stats(self):
        return {"sequence": self.seq}


@dataclass
class _Res:
    center_xy: tuple
    size_wh: tuple
    peak: float = 10.0
    apce: float = 20.0


class Scenario:
    """Tracker + detector fakes sharing one timeline: the drone leaves the frame LOSE_AFTER_S after lock."""

    def __init__(self):
        self.locked_at = None
        self.inits = 0

    def gone(self):
        return self.locked_at is not None and time.monotonic() - self.locked_at > LOSE_AFTER_S

    # tracker
    def init(self, frame, center_xy, size_wh):
        self.inits += 1
        if self.locked_at is None:
            self.locked_at = time.monotonic()

    def update(self, frame):
        center = (W + 50.0, TRACKED[1]) if self.gone() else TRACKED
        return _Res(center, (60.0, 40.0))

    # detector
    def detect(self, gray, roi_xywh=None):
        if self.gone():
            return []
        return [Detection((TRACKED[0] - 30.0, TRACKED[1] - 20.0, 60.0, 40.0), 0.9)]


class InlineExecutor:
    def submit(self, fn, *args):
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001
            f.set_exception(exc)
        return f

    def shutdown(self, wait=False, cancel_futures=False):
        pass


async def test_siamfc_source_drives_the_motors_and_stops_them_on_loss(cuasp_module, fake_pcmm_server, monkeypatch, capsys):
    cuasp = cuasp_module
    scenario = Scenario()
    built = {}

    def camera_api_factory(cfg):
        assert cfg.mode == "ids_siamfc"  # cuasp built the siamfc configuration from settings
        api = cam.CameraAPI(cfg)
        adapter = IdsSiamFCCorrectionSourceAdapter(
            cfg.ids_opencv, cfg.gating, producer=ContinuousProducer(), tracker=scenario, detector=scenario,
            calibration=CALIB, executor_factory=InlineExecutor,
        )
        api._app = cam.CameraApplicationService(adapter, cfg.gating)
        built["adapter"] = adapter
        return api

    monkeypatch.setattr(cuasp, "CameraAPI", camera_api_factory)
    monkeypatch.setattr(cuasp, "SHOW_DEBUG_WINDOW", False)
    monkeypatch.setattr(cuasp, "PRINT_EVERY_S", 0.0)
    monkeypatch.setattr(cuasp, "RUN_DURATION_S", 1.6)
    monkeypatch.setattr(cuasp, "ENABLE_MOTOR_COMMANDS", True)  # enabled for this test only, against the fake server
    monkeypatch.setattr(cuasp, "MOTOR_IP", fake_pcmm_server.host)
    monkeypatch.setattr(cuasp, "MOTOR_PORT", int(fake_pcmm_server.port))

    await cuasp.main()

    lines = fake_pcmm_server.received_lines
    abs_idx = [i for i, line in enumerate(lines) if line.startswith("abs[0,1]=")]
    stop_idx = [i for i, line in enumerate(lines) if line == "stop[0]=1"]
    assert scenario.inits == 1 and not built["adapter"].started  # one lock; camera stopped by main()
    assert abs_idx, f"no absolute command sent: {lines[:10]}"
    # Two stops: the one on loss (tracked -> lost transition) and the safe stop that the
    # C-UASP motor adapter always sends when the dispatcher shuts down (disconnect_motors).
    out = capsys.readouterr().out
    assert out.count("[MOTOR] target lost (reason=siamfc_lost) -> stop[0]=1") == 1
    assert len(stop_idx) == 2, f"expected the loss stop and the shutdown stop, got {len(stop_idx)}"
    assert stop_idx[0] > abs_idx[0]
    assert not [i for i in abs_idx if i > stop_idx[0]], "commands sent after the target was lost"

    # Value and sign: the drone is right of and above the optical axis; with the default
    # settings the effective signs are +1 on both axes, encoders start at 0.
    az, po = (float(v) for v in lines[abs_idx[0]].split("=", 1)[1].split(","))
    exp_az = math.degrees(math.atan(0.1))
    exp_po = math.degrees(math.atan2(0.1, math.hypot(0.1, 1.0)))
    assert az == pytest.approx(exp_az, abs=0.01)
    assert po == pytest.approx(exp_po, abs=0.01)
