# Provenance

Files in this folder were copied verbatim on 2026-10-06 from the C-UASP repository
(`C:\Users\attor\AppData\Local\GitHubDesktop\C-UASP`). C-UASP itself is never modified;
all SiamFC integration changes happen on these copies only.

- C-UASP HEAD: `4df9608a79db484d30196f4088f63f8939a40d3e`
- C-UASP working tree at copy time was NOT clean: `control/suncubes/settings.py` had uncommitted
  local modifications, so the copied `settings.py` reflects the working tree, not HEAD.
- Not copied: calibration scripts and their tests (`test_calibration_synthetic_intrinsics.py`,
  `test_camera_get_images_pure.py`, `test_finalize_runtime_calibration.py`), KAS projects, logs.

## SHA256 at copy time (before any local change)

```
c72932cf66dc8d088238e5169402cace0b415a1f553b05bd367a1410884bbec5 *./calibration/boresight.yaml
40e88000f871fca7a74940bbcef89e725628868b2252cab71ed24c821ff58ce6 *./calibration/calibration.yaml
657540ff1450027930ca4766d4a1d404b54f5b48a5eb899e8774bd048dfb25d4 *./calibration/calibration_legacy_20260921_x_test.yaml
2bb551b597c93c0beb98746eecff46ba392753c7c079b7189b7194bef87823d2 *./control/cuasp.py
29a6eb163766f5a599ee6e8aed4313f61e8c01b385ec758f2e6d47ce6d6d9632 *./control/suncubes/README.md
4542ad48feca1ccc4a19ea2059500ca61fbad6a6545c57f8dd9c148e74dd88f6 *./control/suncubes/__init__.py
d87af3a5a75da5bf823339cdd384277f3fae3d38ad0724b1f13ffacdf3da333d *./control/suncubes/camera.py
76b463b13008f01223764cc5670ad5392d16b952aaff89e72c6c64e99a862453 *./control/suncubes/log_pointing.py
a9f78fbf3bbfb32748ed4391255fc11fa3538cd6318eafef0fb747c109a509da *./control/suncubes/motors.py
799e7bd5d08d1c996df8b6b3579557c82d0aabb9d69595a8f77f10122e0a444c *./control/suncubes/settings.py
8eea61cb01fa8afe24b5df9d3850ed6e2a8fbd30610267dbfc3af1b60c5e7fc0 *./pyproject.toml
70ffdf68cb5ce8f35353b65924084d9ee7cf95abb24a4e5234c9125f59cb5162 *./tests/conftest.py
07fd4451c4af043bdebcb984058be4abd2fc6d5a179b2749ad99e1af81827939 *./tests/test_camera_config_routing.py
a7076a3358e193d5c0c40cb4f4a5117c59d3b77bd92496bf7ee8e58846242376 *./tests/test_motor_dispatcher.py
960aceeb76cbeef14130517dd829d13c14d3f0e3078b9903d21e37bd09a06896 *./tests/test_motors_fake_server.py
8145a0dbbcde105f6c0dffbc8eb9afdab3a8d902b760f73afe46f6f119648af0 *./tests/test_motors_protocol.py
76d66c0723a9f86c2f3e37d5f72f080ca2b64dadb33dab19acec99bcf980b496 *./tests/test_multi_marker_detection_e2e.py
88605138ad8c6afa26d925cf9805aa29041cc08ac3dba01e24e103a0c48f39ad *./tests/test_multi_marker_fusion.py
54be2f812b7fdd71a5d605655eec87382b779c209784556d8ffcc90eccf00c11 *./tests/test_multi_marker_tiled_preprocessor.py
38cea8a8870dd8638ba40bc66dfff59127afc1647b62146a2088169161a7c795 *./tests/test_pbvs_correction.py
70a92877469a2f818413d9cc0ddddfe2ab93d8fe60c48c1f5010f7dcacc29148 *./tests/test_rotation_and_covariance.py
5172b97aee5f6d068f337523adf37fe6229e27c068369764bdced41b427bbcd0 *./tests/test_settings_validate.py
acb58b2a96b54e75c8ceb94c5869e5785db22b8a9a9a59f84b9a9a8fabd89739 *./tests/test_visual_sample_gating.py
```

## Local changes to the copied files

Checked on 2026-11-11 against the SHA256 above: six copied files differ from C-UASP; all the
others (including `control/suncubes/motors.py`, the calibration files and
`tests/test_motor_dispatcher.py`) are unchanged. Every change below is marked in the code
with "cuasp_siamfc".

### Behaviour changes with respect to C-UASP

These alter how the copied runtime behaves, not only what it is configured to do.

1. **Stop on target loss is sequenced by the motor dispatcher worker**
   (`control/cuasp.py`, `MotorCommandDispatcher`, 2026-11-11).
   - C-UASP: on a tracked -> lost transition, `submit()` started the stop as a separate
     task, `asyncio.create_task(_call_method_maybe_async(self.motors.stop, ...))`. If the
     worker `_run()` was in the middle of `_send()` (encoders read, `abs_rotate` about to be
     written), the two competed for the `motors.py` socket lock, and the wire could carry
     `stop[0]=1` followed by the in-flight `abs[0,1]=...`, computed from the last reliable
     sample. That cancelled the stop: after the loss the axes moved again towards the last
     target instead of stopping. Found by `tests/test_cuasp_siamfc_e2e.py` (about 2 runs in
     10); reproduced deterministically: call order `enc, abs_begin, stop, abs_end`.
   - Here: `submit()` only sets `_stop_pending`; the worker sends the stop
     (`_send_stop_on_loss()`) at the top of its next iteration, before anything else, so it
     always follows any command already in flight (`enc, abs_begin, abs_end, stop`). In
     addition, `_send()` drops its command if the loss arrives while the encoders are being
     read, so in that case only the stop goes out. A failing stop is logged and marks the
     connection as broken, like a failing send (in C-UASP the exception was left unhandled
     in the detached task).
   - Unchanged: the stop still fires exactly once per tracked -> lost transition; latest-only
     queue, stale-command drop, rate limiting, `abs_rotate` against a fresh encoder read, the
     safe stop on shutdown. Cost: the stop can wait for the end of one in-flight send (one
     encoder round trip + one write, a few ms).
   - Tests: `tests/test_motor_dispatcher_stop_order.py` (new, deterministic race cases);
     the original `tests/test_motor_dispatcher.py` passes unchanged; the end-to-end test
     passed 30 runs out of 30 after the change.
   - The same race exists in C-UASP (ArUco mode too). C-UASP is not modified (project rule);
     porting the fix there is a decision for its maintainers.

2. **Motor commands disabled by default** (`control/suncubes/settings.py`, 2026-11-10):
   `CV_CUAS_ENABLE_MOTOR_COMMANDS = 0` (C-UASP: 1), so a first SiamFC run cannot move the
   motors by mistake. Enable deliberately for the motors-on bench test only.

### Integration changes (SiamFC tracker, PLAN.md Phases 3-4)

- `control/cuasp.py`: `CV_TARGET_TRACKER` selection (`TARGET_TRACKER`, default `"siamfc"`);
  `_siamfc_source_dicts()` / `_resolve_repo_path()`; `_validate_configuration()` skips the
  ArUco checks and validates the SiamFC source configuration in siamfc mode;
  `_build_camera_config()` builds `provider_mode = "ids_siamfc"` with `CV_SIAMFC_CONF_MIN`
  as gating confidence; start-up banner (tracker, fixed range, weight paths and SHA256);
  debug window (SiamFC box, detections, status line, "fixed" laser range label); message
  `[MOTOR] marker lost` -> `[MOTOR] target lost`. PBVS signs, gain and clamp are unchanged.
- `control/suncubes/camera.py`: `_build_source()` routes `provider_mode == "ids_siamfc"` to
  `suncubes.siamfc_source.IdsSiamFCCorrectionSourceAdapter` (lazy import); comment on the
  modes in `CameraConfig`. Nothing else changes in this file.
- `control/suncubes/settings.py`: `CV_TARGET_TRACKER`, `CV_SIAMFC_*`, `CV_DETECTOR_*`
  settings and their checks in `validate()` (plus behaviour change 2 above).
- `pyproject.toml`: `siamfc` extra (torch, torchvision, scipy, ultralytics) and comments.
- `tests/test_camera_config_routing.py`: the original ArUco tests run with
  `TARGET_TRACKER = "aruco"` and assert the same things; SiamFC tests added.
- `tests/test_settings_validate.py`: tests for the new settings appended; the original
  tests are unchanged.

New files (not copies): `control/suncubes/siamfc_source.py`, `control/suncubes/siamfc/`,
`tools/`, the other `tests/test_siamfc_*`, `tests/test_eval_detector.py`,
`tests/test_replay_tool.py`, `tests/test_train_detector_stop.py`,
`tests/test_cuasp_siamfc_e2e.py`, `tests/test_motor_dispatcher_stop_order.py`.

## Baseline test status at copy time

`python -m pytest` in this folder: **92 passed, 1 failed**. The failure,
`tests/test_multi_marker_fusion.py::test_settings_default_target_offsets_match_restored_rig_values`
(`CV_CUAS_ARUCO_1_TARGET_OFFSET_CM` x = 21.0 cm in settings vs 18.0 cm expected by the test),
is pre-existing: it fails identically in C-UASP itself. It concerns the ArUco
multi-marker path only and is left untouched here.
