# `control/suncubes` runtime library

This directory contains the Python runtime components for visual pointing and PCMM communication.

## Current files

| File | Status | Responsibility |
|---|---|---|
| `camera.py` | implemented | IDS acquisition, KLT/ArUco tracking, PnP, PBVS, camera health/debug |
| `motors.py` | implemented | TCP/UDP adapters for PCMM/KAS position reference and status queries |
| `settings.py` | implemented | typed, commented, fail-fast-validated runtime settings (single live config source for `cuasp.py` and `camera.py`) |
| `log_pointing.py` | placeholder | intended runtime binary logging and parquet conversion |

**Resolved:** this directory previously lacked a package initializer/packaging metadata, and `camera.py`/`cuasp.py` referred to a `SunCubes` (capitalized) import while this directory is named `suncubes`. On a case-insensitive filesystem this looked like it should work; it did not actually resolve, because CPython's import machinery matches package/module names case-sensitively against the directory listing regardless of OS path case-insensitivity (verified: `import SunCubes` raised `ModuleNotFoundError` even with `control/` on `sys.path`). The fix:
- every `from SunCubes import ...` / `from SunCubes.xxx import ...` in `control/cuasp.py` was changed to the canonical lower-case `from suncubes import ...` / `from suncubes.xxx import ...`, matching this directory's actual on-disk name;
- internal imports inside the package (`camera.py`'s own settings lookup, its CSV-mode `cv_aruco_helper` import) now use relative imports (`from . import settings`, `from .cv_aruco_helper import ...`) instead of an absolute `SunCubes`/`suncubes` reference, per the "use relative imports internally" rule;
- `control/suncubes/__init__.py` was added so this is a real package, not an implicit namespace package;
- a root-level `pyproject.toml` was added (`[tool.setuptools] package-dir = {"" = "control"}`, discovering the `suncubes` package under `control/`), giving the project proper packaging metadata; `control/cuasp.py` still also does its own `sys.path.insert()` so it keeps working as a directly-run script even without an editable install.

Do not reintroduce a `SunCubes`-cased import anywhere in this tree.

**Resolved (configuration source):** `settings.py` was previously 0 bytes, so every `CV_CUAS_*`/`CV_ARUCO_*`/`CV_IDS_*`/`CV_PBVS_*`/`IP`/`PORT` lookup silently fell back to a hardcoded default buried in `camera.py`/`cuasp.py` source. It is now populated with every one of those ~90 names, grouped by subsystem (IDS acquisition, ArUco detector tuning, KLT/detection ROI, multi-marker target fusion, PBVS Z resolution, PBVS sign/axis mapping, gating, motor/network), each set to the exact value that was already in effect as the code-level default — this was a zero-behavior-change population, not a re-tuning. A `settings.validate()` function performs fail-fast type/range checks and is called from `cuasp.py`'s `_validate_configuration()` at startup. Note that `camera.py`'s `load_camera_config_yaml()`/`CameraConfig.from_yaml()` path still exists as a separate, optional entry point for other consumers, but `cuasp.py`'s live runtime builds `CameraConfig` directly in Python from `settings.py` and never loads a YAML file — so there is no active dual-source conflict to reconcile today, only this one file to keep authoritative.

One correction made while populating this file: the rigid multi-marker target-point offsets (`CV_CUAS_TARGET_MIDPOINT_OFFSET_CM`, `CV_CUAS_ARUCO_0_TARGET_OFFSET_CM`, `CV_CUAS_ARUCO_1_TARGET_OFFSET_CM`) are restored to the original standalone `cv_CUAS.py` script's values (`(0,5,0)`, `(-18,5,0)`, `(18,5,0)` cm) rather than the `(0,0,0)` the multi-marker migration into `camera.py` had silently introduced as its code-level default. These read as real rig measurements, not placeholders, but have not been re-verified against the physical target since being restored — treat as provisional pending a physical re-check, flagged inline in `settings.py`.

---

# Camera module

## Public API

Primary facade:

```python
CameraAPI
```

Important methods:

```python
CameraAPI.from_yaml(path)
CameraAPI.start()
CameraAPI.get_latest() -> CameraCorrection
CameraAPI.health() -> CameraHealth
CameraAPI.get_debug_snapshot() -> dict
CameraAPI.fast_stats() -> dict
CameraAPI.stop()
```

`CameraAPI` is intended to be consumed by the system supervisor, not by the PCMM/KAS layer directly.

## `CameraCorrection`

Fields:

```text
reliable: bool
reason: str
d_az_rad: float
d_po_rad: float
confidence: Optional[float]
age_s: Optional[float]
marker_id: Optional[int]
z_m: Optional[float]
ts_capture_ns: Optional[int]
ts_consumed_ns: int
```

Semantics:
- `d_az_rad` and `d_po_rad` are angular pointing-error measurements in radians;
- `reliable=False` means the consumer must not use the angular values for new motion authority;
- `age_s` is based on the current host-side frame timestamp semantics;
- the same measurement may be returned repeatedly until a new frame is processed.

## Live thread model

### Thread 1: `ids-latest-frame`

Owns IDS SDK acquisition.

Per frame:

```text
WaitForFinishedBuffer
 -> host monotonic timestamp
 -> SDK image view
 -> NumPy copy
 -> _LatestFrameSlot.publish()
 -> QueueBuffer(buffer)
```

The NumPy copy is essential because the IDS SDK buffer is immediately returned to the driver and may be reused.

`_LatestFrameSlot` stores one `_FramePacket`:

```text
gray
ts_capture_ns
sequence
```

A new publish replaces the slot reference; old arrays remain alive only while another Python object/thread still references them.

### Thread 2: `camera-api-worker`

Scheduled around `RuntimeConfig.worker_hz` (default 70 Hz).

For each iteration:
- call `CameraApplicationService.process_once()`;
- read the latest frame packet;
- if sequence is unchanged, reuse/age the previous candidate;
- if new, execute KLT and current-frame pose/PBVS;
- consume or schedule asynchronous marker detection;
- publish the latest immutable `CameraCorrection` under a short lock;
- do not execute catch-up bursts after an overrun.

### Thread 3: `aruco-detector`

A `ThreadPoolExecutor(max_workers=1)` worker.

It is scheduled only when due and only if a previous detector job is not still running.

Detection modes:
- `track_roi`;
- `recover_roi`;
- `scout`;
- `full_recovery`.

It returns a `_DetectionResult` containing the detector frame, sequence, timestamp, marker ID, and four corners. If the result belongs to an older frame than the current high-rate frame, KLT propagates those corners to the current frame before accepting them.

## KLT

`SparseCornerTracker` uses sparse pyramidal Lucas-Kanade optical flow on the four known marker corners.

Default normal profile:

```text
window             15 x 15
max pyramid level   2
max iterations     10
epsilon             0.03
```

Default robust retry:

```text
window             21 x 21
max pyramid level   3
max iterations     15
epsilon             0.01
forward/backward    forced
```

Additional checks include:
- LK status/error;
- forward/backward error threshold;
- finite/in-frame points;
- convex quadrilateral;
- area-ratio bounds;
- minimum area;
- side-ratio bound.

The tracker may use CUDA sparse PyrLK only when available and benchmarked faster than CPU.

## ArUco/AprilTag detection

OpenCV `cv2.aruco` is used as the detector API. The configured dictionary may be an ArUco or OpenCV AprilTag dictionary.

Default corner refinement setting is `SUBPIX`, with configurable window/iterations/accuracy. Marker detection supplies an absolute identity and absolute four-corner localization; KLT supplies high-rate inter-frame propagation.

## PnP/PBVS

Given four image corners and known marker side length, the module estimates marker pose relative to the camera. It prefers OpenCV single-marker pose estimation and has a square-PnP fallback (`SOLVEPNP_IPPE_SQUARE` when available).

Pose is validated by reprojecting the marker corners and checking RMSE/max pixel errors.

Marker-local target point:

```text
p_T_C = t_CM + R_CM * p_T_M
```

Camera-to-laser transform:

```text
p_T_L = R_LC * p_T_C + t_LC
```

Angular outputs:

```text
d_az = atan2(x_L, z_L)
d_po = atan2(-y_L, sqrt(x_L^2 + z_L^2))
```

## Camera configuration

`load_camera_config_yaml()` currently reads:

```yaml
camera:
  enable: true
  provider_mode: ids_opencv   # or csv
  debug_view: false

  runtime:
    background_worker: true
    worker_hz: 70.0

  gating:
    max_age_s: 0.25
    confidence_min: 0.15
    z_min_m: 0.25
    max_abs_angle_rad: 0.0

  ids_peak:
    # IDS-specific fields

  aruco:
    # marker/detection fields

  calibration:
    path: ../../calibration/calibration.yaml

  csv:
    path: ...
```

In addition, many fast-tracking values are looked up dynamically from `suncubes.settings`, which is now the single populated, live configuration source for this tuning surface (see "Resolved (configuration source)" above); this YAML block remains a separate, optional entry point not used by `cuasp.py`'s runtime.

## Camera technical debt / hazards

1. `ts_capture_ns` is currently a host timestamp taken after a completed IDS buffer is returned, not a proven exposure timestamp.
2. ~~`settings.py` is empty, and `_setting()` silently falls back to defaults on any import exception.~~ Resolved: `settings.py` is populated (see above) and `_setting()` only falls back on `ImportError`/`ModuleNotFoundError`, not a bare `except Exception`, so a real bug inside it is no longer silently swallowed.
3. ~~`SunCubes` import case/package does not match the uploaded `suncubes` path.~~ Resolved: imports use the canonical `suncubes` case, internal imports are relative, and the package has `__init__.py` + root `pyproject.toml` metadata.
4. CSV mode depends on `suncubes.cv_aruco_helper.CvArucoStream`, which is still absent from this repository.
5. runtime intrinsic scaling assumes pure image resizing; crop/ROI/binning need explicit geometry.
6. `_LatestFrameSlot._replaced` counts replacement of a non-empty slot, not necessarily frames that a consumer failed to read; sequence gaps are a better skipped-frame metric.
7. `CameraCorrection` should eventually expose a sample/frame sequence to make duplicate-consumption prevention explicit.

---

# Motors module

## Public adapters

```python
PCMMMotorAdapterTCP
PCMMMotorAdapterTCP_sync
PCMMMotorAdapterUDP       # legacy
```

The TCP sync/async adapters share `_PCMMTCPBase`.

## Units

All public motor angular values are:
- position: degrees;
- velocity: degrees/s;
- time: seconds.

Do not pass camera radians directly into these methods.

## TCP protocol

Commands are ASCII, newline terminated.

### Atomic absolute pair

```text
abs[0,1]=axis1_deg,axis2_deg\n
```

This is the preferred command primitive for the main C-UASP visual loop.

### Encoder queries

```text
enc[0]?\n
enc[1]?\n
enc[0,1]?\n
```

### Controller status

```text
status?\n
```

Current parser accepts:
- 15-field `position-v1` schema;
- 12-field legacy error-servo schema for diagnostics.

`PCMMStatus` contains:

```text
controller_state
position_reference_sequence
position_reference_fresh
position_reference_valid
tcp_connected
axis1_velocity_command_deg_s
axis2_velocity_command_deg_s
axis1_reference_deg
axis2_reference_deg
axis1_actual_deg
axis2_actual_deg
reference_dt_s
reference_age_s
position_reference_clamped
controller_fault
status_schema
```

A future KAS release should expose a protocol/build handshake because field count alone cannot prove that state-number semantics match the Python client.

### Stop

```text
stop[0]=1\n
```

## Absolute versus relative API

`abs_rotate()` sends one bounded atomic absolute pair.

`rel_rotate()`:
- maintains an internal `_position_target_deg` accumulator;
- initializes it from both encoders if no target exists;
- adds the requested deltas;
- sends the resulting absolute pair.

`set_visual_error()` is only a compatibility alias for `rel_rotate()`; it does not implement a dedicated visual-servo control law.

Therefore the system supervisor should not use it as an automatic interpretation of every `CameraCorrection`.

## Connection behavior

TCP setup enables:
- `TCP_NODELAY`;
- `SO_KEEPALIVE`;
- connect and I/O timeouts.

Socket access is serialized by `RLock`.

If a request/reply query times out, the adapter closes the socket. This prevents a late reply from being consumed as the response to a later query.

The asyncio adapter uses `asyncio.to_thread()` around the synchronous socket core. Keep a single logical command producer and await operations so the executor does not become an uncontrolled command queue.

## Python-side command limits

Default clamps currently are approximately:

```text
axis 1: -29.5 .. +29.5 deg
axis 2: -19.5 .. +19.5 deg
```

These are Python-side limits and do not replace controller/hardware limits.

---

# Proposed logging schema

`log_pointing.py` should eventually log enough information to reconstruct every command.

Recommended per-update record:

```text
schema_version
session_id
host_monotonic_ns
camera_sample_sequence
camera_ts_sensor_ns          optional
camera_ts_host_rx_ns
camera_ts_consumed_ns
camera_reliable
camera_reason
camera_confidence
marker_id
z_m
d_az_rad
d_po_rad
visual_new_sample
encoder_az_deg
encoder_pol_deg
base_reference_az_deg
base_reference_pol_deg
command_az_deg
command_pol_deg
pcmm_state
pcmm_reference_sequence
pcmm_reference_fresh
pcmm_reference_age_s
pcmm_reference_clamped
pcmm_fault
camera_track_ms
camera_pose_ms
camera_detection_ms
software_build_id
kas_build_id
calibration_id
config_hash
```

During operation, write a compact versioned binary format. Convert to parquet outside the motion-critical loop.

---

# Recommended package cleanup

Target structure:

```text
pyproject.toml
src/cuasp/
  __init__.py
  camera.py
  motors.py
  logging.py
  config.py
  supervisor.py
```

The current tree was preserved instead of moving to a `src/cuasp/` layout, and the "at minimum" fallback has been done: `control/suncubes/__init__.py` exists, the canonical package case/name is `suncubes`, and no code depends on an external `SunCubes`-cased package anymore. The `src/cuasp/` restructuring above remains a possible future option, not a currently-open gap.
