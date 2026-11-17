"""cuasp_siamfc (2026-11-16): IDS camera settings for the camera in use (U3-33DxXCP-M).

- ``_set_gain_db``: explicit dB -> linear factor conversion (C-UASP wrote dB into a factor node);
- ``_set_link_throughput_limit_bps``: bandwidth limit that capped the camera at 43.2 fps;
- the settings and the config dict built by cuasp.py;
- ``_IdsLatestFrameProducer`` with fake IDS modules: what is written to the camera, in which
  order, and that an invalid gain makes the start fail instead of being ignored.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import time

import pytest

from suncubes import camera as cam
from suncubes import settings


class _Node:
    def __init__(self, name, log, value=0.0, lo=None, hi=None, inc=None, unit="", entry=None):
        self.name, self._log, self.value = name, log, value
        self._lo, self._hi, self._inc, self._unit, self.entry = lo, hi, inc, unit, entry

    def Unit(self):
        return self._unit

    def Minimum(self):
        if self._lo is None:
            raise RuntimeError("no minimum")
        return self._lo

    def Maximum(self):
        if self._hi is None:
            raise RuntimeError("no maximum")
        return self._hi

    def Increment(self):
        if self._inc is None:
            raise RuntimeError("no increment")
        return self._inc

    def Value(self):
        return self.value

    def SetValue(self, v):
        if (self._lo is not None and v < self._lo) or (self._hi is not None and v > self._hi):
            raise RuntimeError(f"{self.name}: {v} out of range")  # like the SDK
        self.value = v
        self._log.append((self.name, v))

    def SetCurrentEntry(self, e):
        self.entry = e
        self._log.append((self.name, e))

    def Execute(self):
        self._log.append((self.name, "execute"))


class _NodeMap:
    def __init__(self, nodes):
        self.nodes = {n.name: n for n in nodes}

    def FindNode(self, name):
        if name not in self.nodes:
            raise KeyError(name)
        return self.nodes[name]


def _camera_nodes(log, gain_unit=""):
    """The relevant nodes of the U3-33DxXCP-M as read back on 2026-11-16."""
    return [
        _Node("PixelFormat", log, entry="Mono8"),
        _Node("Width", log, 2040, lo=264, hi=2040, inc=24),
        _Node("Height", log, 1536, lo=16, hi=1544, inc=8),
        _Node("AcquisitionMode", log, entry="Continuous"),
        _Node("AcquisitionFrameRateTargetEnable", log, True),
        _Node("AcquisitionFrameRateTarget", log, 70.0, lo=0.001, hi=74_250_000.0),
        _Node("ExposureTime", log, 20003.0, lo=25.48, hi=1_998_461.6),
        _Node("Gain", log, 1.0, lo=1.0, hi=15.848932266235352, unit=gain_unit),
        _Node("DeviceLinkThroughputLimit", log, 162_000_000, lo=10_000_000, hi=300_000_000, inc=2_000_000, unit="Bps"),
        _Node("PayloadSize", log, 2040 * 1536),
        _Node("AcquisitionStart", log),
        _Node("AcquisitionStop", log),
    ]


# ---------------------------------------------------------------------------- gain


def test_gain_db_is_converted_to_the_linear_factor_of_this_camera():
    log = []
    nm = _NodeMap(_camera_nodes(log))
    assert cam._set_gain_db(nm, 0.0) == 1.0  # C-UASP wrote 0.0 here: rejected, silently ignored
    assert cam._set_gain_db(nm, 6.0) == pytest.approx(10 ** 0.3)  # C-UASP would have written 6.0 (15.6 dB)
    assert cam._set_gain_db(nm, 24.0) == pytest.approx(15.848932, rel=1e-6)  # the camera maximum
    assert [v for n, v in log if n == "Gain"] == pytest.approx([1.0, 10 ** 0.3, 15.848932], rel=1e-6)


@pytest.mark.parametrize("gain_db", [25.0, -1.0, math.nan])
def test_gain_outside_the_camera_range_raises_and_writes_nothing(gain_db):
    log = []
    with pytest.raises(ValueError):
        cam._set_gain_db(_NodeMap(_camera_nodes(log)), gain_db)
    assert log == []


def test_gain_node_declared_in_db_is_written_unchanged():
    log = []
    nm = _NodeMap([_Node("Gain", log, 0.0, lo=0.0, hi=24.0, unit="dB")])
    assert cam._set_gain_db(nm, 6.0) == 6.0


def test_missing_gain_node_is_accepted_only_for_zero_db():
    assert cam._set_gain_db(_NodeMap([]), 0.0) == 1.0
    with pytest.raises(RuntimeError):
        cam._set_gain_db(_NodeMap([]), 3.0)


# ---------------------------------------------------------------------------- link throughput


def test_link_throughput_limit_is_set_in_bytes_per_second():
    log = []
    nm = _NodeMap(_camera_nodes(log))
    assert cam._set_link_throughput_limit_bps(nm, 300_000_000) == 300_000_000
    assert log == [("DeviceLinkThroughputLimit", 300_000_000)]


@pytest.mark.parametrize("limit", [400_000_000, 5_000_000, 301_000_000])  # above max, below min, off-step
def test_invalid_link_throughput_limit_raises_and_writes_nothing(limit):
    log = []
    with pytest.raises(ValueError):
        cam._set_link_throughput_limit_bps(_NodeMap(_camera_nodes(log)), limit)
    assert log == []


def test_link_throughput_limit_in_an_unexpected_unit_is_refused():
    log = []
    nm = _NodeMap([_Node("DeviceLinkThroughputLimit", log, 1, lo=1, hi=10**10, unit="bps")])  # bits/s
    with pytest.raises(ValueError):
        cam._set_link_throughput_limit_bps(nm, 300_000_000)
    assert log == []


# ---------------------------------------------------------------------------- settings and config


def test_settings_match_the_camera_in_use():
    assert (settings.CV_IDS_WIDTH, settings.CV_IDS_HEIGHT) == (2040, 1536)
    assert settings.CV_IDS_EXPOSURE_TIME_US == 15000.0
    assert settings.CV_IDS_LINK_THROUGHPUT_LIMIT_BPS == 300_000_000
    assert settings.CV_IDS_GAIN_DB == 6.0  # user decision 2026-11-17 (was 0.0)
    settings.validate()  # raises ValueError on malformed settings


@pytest.mark.parametrize("name, bad", [("CV_IDS_GAIN_DB", -3.0), ("CV_IDS_LINK_THROUGHPUT_LIMIT_BPS", -1)])
def test_settings_validation_rejects_negative_gain_and_throughput(monkeypatch, name, bad):
    monkeypatch.setattr(settings, name, bad)
    with pytest.raises(ValueError, match=name):
        settings.validate()


def test_cuasp_forwards_the_ids_settings(cuasp_module):
    ids = cuasp_module._build_camera_config().ids_opencv["ids_peak"]
    assert (ids["width"], ids["height"]) == (2040, 1536)
    assert ids["exposure_time_us"] == 15000.0 and ids["gain_db"] == 6.0
    assert ids["link_throughput_limit_bps"] == 300_000_000
    assert str(cuasp_module.CALIBRATION_YAML).endswith("calibration_legacy_20260921_x_test.yaml")  # user decision


# ---------------------------------------------------------------------------- producer with fake IDS modules


def _fake_ids(nodemap, log):
    class _DataStream:
        def NumBuffersAnnouncedMinRequired(self):
            return 2

        def AllocAndAnnounceBuffer(self, size):
            return object()

        def QueueBuffer(self, buf):
            pass

        def StartAcquisition(self):
            log.append(("StartAcquisition", None))

        def StopAcquisition(self):
            log.append(("StopAcquisition", None))

        def Flush(self, mode):
            pass

        def WaitForFinishedBuffer(self, timeout_ms):
            time.sleep(0.005)
            raise TimeoutError

    device = SimpleNamespace(
        DataStreams=lambda: [SimpleNamespace(OpenDataStream=lambda: _DataStream())],
        RemoteDevice=lambda: SimpleNamespace(NodeMaps=lambda: [nodemap]),
    )
    devices = [SimpleNamespace(OpenDevice=lambda access: device)]

    class _Devices(list):
        def empty(self):
            return not self

        def size(self):
            return len(self)

    manager = SimpleNamespace(Update=lambda: None, Devices=lambda: _Devices(devices))
    ids_peak = SimpleNamespace(
        Library=SimpleNamespace(Initialize=lambda: None, Close=lambda: log.append(("Library.Close", None))),
        DeviceManager=SimpleNamespace(Instance=lambda: manager),
        DeviceAccessType_Exclusive="exclusive",
        DataStreamFlushMode_DiscardAll="discard",
    )
    return ids_peak, SimpleNamespace()


def _producer(monkeypatch, cuasp_module, nodemap, log, **overrides):
    monkeypatch.setattr(cam, "_load_ids_modules", lambda: _fake_ids(nodemap, log))
    cfg = {**cuasp_module._build_camera_config().ids_opencv["ids_peak"], **overrides}
    return cam._IdsLatestFrameProducer(cfg)


def test_producer_writes_the_configured_camera_settings(monkeypatch, cuasp_module):
    log = []
    nm = _NodeMap(_camera_nodes(log))
    prod = _producer(monkeypatch, cuasp_module, nm, log)
    prod.start(timeout_s=5.0)
    prod.stop()
    written = {n: v for n, v in log}
    assert written["Width"] == 2040 and written["Height"] == 1536
    assert written["DeviceLinkThroughputLimit"] == 300_000_000
    assert written["ExposureTime"] == 15000.0
    assert written["Gain"] == pytest.approx(10 ** 0.3)  # 6 dB -> factor 1.995 (camera readback 2026-11-17)
    names = [n for n, _ in log]
    assert names.index("DeviceLinkThroughputLimit") < names.index("AcquisitionFrameRateTarget") < names.index("StartAcquisition")


def test_producer_start_fails_on_a_gain_the_camera_cannot_apply(monkeypatch, cuasp_module):
    log = []
    nm = _NodeMap(_camera_nodes(log))
    prod = _producer(monkeypatch, cuasp_module, nm, log, gain_db=30.0)  # 31.6x > 15.85x
    with pytest.raises(RuntimeError, match="IDS capture start failed"):
        prod.start(timeout_s=5.0)
    assert "Gain" not in [n for n, _ in log] and "StartAcquisition" not in [n for n, _ in log]
    assert ("Library.Close", None) in log  # cleaned up
