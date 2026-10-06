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

## Baseline test status at copy time

`python -m pytest` in this folder: **92 passed, 1 failed**. The failure,
`tests/test_multi_marker_fusion.py::test_settings_default_target_offsets_match_restored_rig_values`
(`CV_CUAS_ARUCO_1_TARGET_OFFSET_CM` x = 21.0 cm in settings vs 18.0 cm expected by the test),
is pre-existing: it fails identically in C-UASP itself. It concerns the ArUco
multi-marker path only and is left untouched here.
