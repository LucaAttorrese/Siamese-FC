# cuasp_siamfc — SiamFC target tracking for the C-UASP pointing system

Python port of the SiamFC tracker, integrated as a camera correction source for
a copy of the C-UASP runtime (`control/cuasp.py` + `control/suncubes/`). The
goal is to replace the ArUco/PnP target measurement with visual tracking of the
drone itself, keeping the `CameraCorrection` -> PBVS -> PCMM/KAS motor
dispatch contract unchanged.

The C-UASP repository is **never modified**: the files it contributes were
copied here verbatim first (see [PROVENANCE.md](PROVENANCE.md)) and all
integration changes are made on these copies.

## Project status

| Part | Status |
|---|---|
| SiamFC port (converter, network, tracker) | Done and validated offline (V1, V2, V5 below) |
| Detector datasets (DUT-Anti-UAV, Anti-UAV300 → grayscale YOLO set) | Done, see `../datasets/MANIFEST.md` |
| Detector training (YOLO11s) | Done on the training PC (RTX 5060 Laptop), 2026-11-09: run `yolo11s_uav_gray_1280_e40`, 40 epochs in stages with stop/resume, ~10 min per epoch. `best.pt` (epoch 36, gitignored, SHA256 `3a7c173244899df1ac5185916cbb8196bd0959b3d5626acc9f9cc3976d8040d5`): val mAP50 0.954 / mAP50-95 0.649; **test mAP50 0.949 / mAP50-95 0.593, P 0.962, R 0.869**. Per-source and per-size breakdown with `tools/eval_detector.py`: DUT mAP50 0.941 / mAP50-95 0.670, Anti-UAV300 0.959 / 0.546; recall at conf 0.25 drops on Anti-UAV300 boxes of 16–32 px (0.17) and 32–64 px (0.76), while DUT stays at 0.90–0.95 at every size |
| Drone detector module (`control/suncubes/siamfc/detector.py`) | Done: `DroneDetector` protocol, Ultralytics implementation (full frame or ROI, SHA256 check of the weights), `DetectionConfirmer` (N consistent detections before a target is confirmed). 29 ms per full 2592x1944 frame on the training PC |
| Camera adapter `ids_siamfc` (`control/suncubes/siamfc_source.py`) | Done: tracking state machine (searching / tracking / lost), fixed-Z geometry, full-frame detector verification, routing `provider_mode = "ids_siamfc"` in `camera.py`. Replayed with the real models on DUT-Anti-UAV tracking videos (details and known limitation in `../PLAN.md`, Phase 3) |
| Wiring into `cuasp.py` / `settings.py` (Phase 4) | Done: `CV_TARGET_TRACKER = "siamfc"` (default) selects the SiamFC source; all `CV_SIAMFC_*` / `CV_DETECTOR_*` settings, validated at start-up; SiamFC box in the debug window; weight hashes in the banner. Motor commands disabled by default. Stop-on-loss race of the inherited motor dispatcher fixed (see below). End-to-end test against the fake PCMM server (`tests/test_cuasp_siamfc_e2e.py`, 30/30 runs) |
| Validation on drone data, hyperparameter search, bench tests (V3, V4, V4b, V6–V8) | Not started |

## Behaviour changes with respect to C-UASP

Besides adding the SiamFC tracker, this copy of the C-UASP runtime behaves differently
from the original in the following respects (full list of local changes, with the
reasons, in [PROVENANCE.md](PROVENANCE.md), "Local changes to the copied files"):

1. **Stop on target loss is sequenced by the motor dispatcher worker**
   (`control/cuasp.py`, `MotorCommandDispatcher`). In C-UASP the stop was a separate
   asyncio task that could be overtaken by an `abs_rotate` already in flight, so the wire
   carried `stop[0]=1` followed by `abs[0,1]=...` and the axes moved again after the loss.
   Here the stop always goes out after any in-flight command, and a command whose encoder
   read was still in progress when the loss arrived is dropped. The same race is still
   present in C-UASP itself.
2. **Motor commands are disabled by default** (`CV_CUAS_ENABLE_MOTOR_COMMANDS = 0` in
   `control/suncubes/settings.py`; 1 in C-UASP).
3. **The camera gain is set in explicit units** (`camera.py`, `_set_gain_db()`).
   `CV_IDS_GAIN_DB` is converted to the linear factor the camera expects. A gain the camera
   cannot apply makes the camera start fail instead of being silently ignored. In C-UASP
   the dB value was written as a factor.
4. **The camera link bandwidth limit is set by the runtime**
   (`CV_IDS_LINK_THROUGHPUT_LIMIT_BPS` = 300 MB/s). It is applied with unit and range
   checks; an invalid value fails the start.
5. **Camera settings match the camera in use** (U3-33DxXCP-M, measured on 2026-11-16):
   2040x1536 instead of 2592x1944 (which the camera silently rejected), and a 15 ms
   exposure instead of 20 ms. Result: 62.6 fps instead of 43.2 fps. Gain +6 dB since
   2026-11-17: at 0 dB the indoor frames on an overcast day were too dark for the detector
   in front of dark backgrounds.

## Using this repository on the training PC

Since 2026-11-06 the training and runtime machine is a laptop with an
**RTX 5060 Laptop GPU (8 GB)**, Ryzen 7 260 and 31 GB RAM. It replaces the RTX 5090 PC
planned earlier. The procedure below also applies to any other Blackwell GPU
(RTX 50xx, `sm_120`).

Commands are for **Windows PowerShell**, run from `Siamese-FC\cuasp_siamfc` unless
stated otherwise. On Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python`.

### 1. Prerequisites

| Item | Requirement |
|---|---|
| GPU driver | NVIDIA driver supporting **CUDA 13.0** (R580 or newer). Check with `nvidia-smi` (top-right "CUDA Version" ≥ 13.0). |
| Python | **3.13 or newer** (tested: 3.13.3 on the development laptop, 3.13.17 on the training PC). Do not use 3.12 or older on Windows: there `time.monotonic()` is `GetTickCount64()` with a **16 ms** step, coarser than one frame at 70 Hz (14.3 ms). C-UASP uses `monotonic_ns` for sample age and freshness checks, and `test_multi_marker_detection_e2e.py::test_read_candidate_reuses_cached_result_for_unchanged_sequence` fails intermittently. From 3.13 it is `QueryPerformanceCounter()` (0.1 µs). Check with `python -c "import time; print(time.get_clock_info('monotonic'))"`. |
| Git | To get the code. |
| Disk | About **35 GB** free: venv ~5 GB, datasets ~10 GB extracted plus ~10 GB of zips while downloading (deleted afterwards), YOLO dataset 3.6 GB, training runs a few GB. |
| Internet | To download packages, weights, datasets, and the YOLO starting weights (`yolo11s.pt`; Ultralytics also downloads `yolo26n.pt` for its AMP check). |
| Live camera only | IDS peak SDK (runtime and GenTL producers, from the IDS installer; training PC: 26.06.1) plus its Python bindings from PyPI: `pip install ids_peak==1.17.0 ids_peak_ipl==1.18.0` (installed on the training PC on 2026-11-11, `abi3` wheels for Python ≥ 3.11). Not needed for training or offline tests. |

### 2. Get the code

The work is on the branch **`siamfc-python`** of
<https://github.com/LucaAttorrese/Siamese-FC>. Only committed and pushed changes
reach the new PC. Weights, datasets, venv and training runs are gitignored and must
be recreated (steps 3, 4, 6). Copying them from the laptop also works, except the venv,
which needs the `cu130` build on a Blackwell GPU.

```powershell
git clone -b siamfc-python https://github.com/LucaAttorrese/Siamese-FC.git
cd Siamese-FC\cuasp_siamfc
```

### 3. Python environment

The RTX 5060 Laptop (like any RTX 50xx) is a Blackwell GPU (`sm_120`). PyTorch must come
from the **`cu130`** wheel index. The `cu126` build used on the development laptop stops at
`sm_90` and **does not run** on Blackwell GPUs.

Install torch **and torchvision from the same CUDA index before anything else**.
Otherwise `pip install ultralytics` pulls a CPU-only torchvision from PyPI, and NMS fails
on the GPU (`Could not run 'torchvision::nms' with arguments from the 'CUDA' backend`).
This happened on the laptop on 2026-10-07.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -m pip install numpy opencv-contrib-python PyYAML scipy pytest pytest-asyncio ultralytics
```

Check the installation:

```powershell
.\.venv\Scripts\python -c "import torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list()); b = torch.rand(8, 4, device='cuda'); b[:, 2:] += b[:, :2]; print('nms on', torchvision.ops.nms(b, torch.rand(8, device='cuda'), 0.5).device); print('cv2', cv2.__version__, 'aruco', hasattr(cv2, 'aruco'))"
```

Expected: `2.14.1+cu130 0.29.1+cu130 NVIDIA GeForce RTX 5060 Laptop GPU`, an arch list
that contains `sm_120`, `nms on cuda:0`, and `aruco True`.

`ultralytics` also installs `opencv-python` next to `opencv-contrib-python`. Both
provide the `cv2` module. If `aruco` prints `False` (the ArUco pipeline in `cuasp.py`
needs it), run
`.\.venv\Scripts\python -m pip uninstall -y opencv-python opencv-contrib-python` and then
`.\.venv\Scripts\python -m pip install opencv-contrib-python`.

Versions validated (2026-11-06):

| Package | Development laptop (RTX 4060) | Training PC (RTX 5060 Laptop) |
|---|---|---|
| Python | 3.13.3 | 3.13.17 |
| torch / torchvision | 2.14.1+cu126 / 0.29.1+cu126 | 2.14.1+cu130 / 0.29.1+cu130 |
| ultralytics | 8.4.174 | 8.4.203 |
| numpy | 2.5.3 | 2.5.4 |
| opencv-contrib-python | 5.0.0.93 | 5.0.0.93 (`opencv-python` 5.0.0.93 also present, `aruco` available) |
| scipy | 1.18.1 | 1.18.1 |
| PyYAML | 6.0.3 | 6.0.3 |
| pytest / pytest-asyncio | 9.1.1 / 1.4.0 | 9.2.0 / 1.4.1.post1 |

### 4. SiamFC weights

Download the CFNet networks (see "Network weights" below for the source), check the
hash, unzip, and convert the grayscale network to PyTorch:

```powershell
cd ..                                   # Siamese-FC
mkdir pretrained; cd pretrained
curl.exe -o cfnet-networks.zip ftp://ftp.robots.ox.ac.uk/pub/outgoing/tvg/cfnet/cfnet-networks.zip
(Get-FileHash cfnet-networks.zip -Algorithm SHA256).Hash   # 440E9125A0C377C29523954CF61D07342CD5384CEB13792B3DCDB342A2743579
..\cuasp_siamfc\.venv\Scripts\python -c "import zipfile; zipfile.ZipFile('cfnet-networks.zip').extractall('.')"
cd ..\cuasp_siamfc\control
..\.venv\Scripts\python -m suncubes.siamfc.convert_matconvnet ..\..\pretrained\networks\baseline-conv5_gray_e100.mat
..\.venv\Scripts\python -m suncubes.siamfc.convert_matconvnet ..\..\pretrained\networks\baseline-conv5_e55.mat   # RGB, for comparisons only
cd ..
```

The converter writes `baseline-conv5_gray_e100.pth` next to the `.mat`, plus a
`.json` with the converter version and the SHA256 of input and output.

### 5. Tests and latency benchmark

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python tools\bench_siamfc.py --weights ..\pretrained\networks\baseline-conv5_gray_e100.pth
.\.venv\Scripts\python tools\eval_sequence.py ..\demo-sequences\vot15_bag --weights ..\pretrained\networks\baseline-conv5_gray_e100.pth --gray
```

- `pytest`: everything passes except
  `test_multi_marker_fusion.py::test_settings_default_target_offsets_match_restored_rig_values`.
  That failure is pre-existing in C-UASP (see `PROVENANCE.md`) and unrelated to SiamFC.
  On Python 3.12 or older on Windows,
  `test_multi_marker_detection_e2e.py::test_read_candidate_reuses_cached_result_for_unchanged_sequence`
  also fails intermittently because of the 16 ms clock (see Prerequisites): use 3.13.
- `bench_siamfc.py`: record the numbers in "Validation status" below (V5, one row per machine).
- `eval_sequence.py`: must give the same accuracy as on the laptop (mean IoU 0.796,
  mean centre error 5.9 px). Only the timing changes between machines.

### 6. Datasets

Follow `..\datasets\MANIFEST.md`: download links (Google Drive with Baidu mirrors),
SHA256 of every archive, extraction (visible-light part of Anti-UAV300 only), and the
command that builds the grayscale YOLO set `..\datasets\yolo_uav_gray`. The
resulting image counts are listed there.

### 7. Detector training

```powershell
.\.venv\Scripts\python tools\train_detector.py --data ..\datasets\yolo_uav_gray\data.yaml --imgsz 1280 --epochs 40 --workers 2 --name yolo11s_uav_gray_1280_e40
```

Stop and resume (details in the docstring of `tools/train_detector.py`):

```powershell
# clean stop at the end of the current epoch (from another terminal)
.\.venv\Scripts\python tools\train_detector.py --name yolo11s_uav_gray_1280_e40 --request-stop
# resume from the last completed epoch
.\.venv\Scripts\python tools\train_detector.py --data ..\datasets\yolo_uav_gray\data.yaml --name yolo11s_uav_gray_1280_e40 --resume --workers 2
```

- `--request-stop` only creates the file `..\runs\detector\<name>\STOP`; creating it by
  hand works the same. The training finishes the current epoch, including its
  validation and the save of `weights\last.pt`, then exits with code 3 and prints the
  resume command. The test evaluation is skipped, and `STOP` is deleted.
- The request is ignored during the last epoch: a checkpoint of the final epoch cannot be
  resumed, so the run simply finishes.
- The clean stop deliberately bypasses Ultralytics' own end of training, which calls
  `strip_optimizer` on `last.pt` and makes it impossible to resume.
- `--resume` keeps the checkpoint's settings: epochs, image size, batch and seed. Only
  `--workers` and `--patience` can be changed. When the run completes, the script
  evaluates `best.pt` on the test split as for a run without interruptions.
- A hard interruption (Ctrl+C, crash, power loss) loses only the epoch in progress, and
  `--resume` works the same way. On Windows, Ctrl+C also reaches the dataloader workers,
  so it is never a clean stop: use `--request-stop`.
- A new run refuses a `--name` whose folder already exists. Otherwise Ultralytics would
  silently train into `<name>2` and `STOP` would be looked for in the wrong folder.
- Each completed epoch prints one `EPOCH_DONE {...}` line with train and validation
  minutes, which is handy for monitoring.

- Outputs go to `..\runs\detector\<name>\` (gitignored): `weights\best.pt`,
  `weights\last.pt`, `results.csv`, plots. When training ends, the script also
  evaluates `best.pt` on the test split.
- **Always pass `--workers 2`** on Windows. The script default is 4, and validation
  uses twice as many workers. Each worker is a full Python process that reserves
  about 1.7 GB of virtual memory. On the laptop this exhausted the page file
  (`WinError 1455`) and hung the first validation. Close WSL (`wsl --shutdown`) during
  training if it is not needed: it holds about 4 GB.
- `--batch -1` (default) lets Ultralytics choose the batch size for 60 % of GPU
  memory. Both 8 GB machines (development laptop and training PC) get batch 3 at 1280 px.
- Duration on `yolo_uav_gray` at 1280 px, batch 3, `--workers 2`:

  | Machine | Train | Validation | Epoch | 40 epochs |
  |---|---|---|---|---|
  | Development laptop (RTX 4060 Laptop) | – | – | ~16 min | ~10.7 h |
  | Training PC (RTX 5060 Laptop), measured 2026-11-06, first epoch of `yolo11s_uav_gray_1280_e40` | 8.9 min | 1.2 min | **10.1 min** (604 s in `results.csv`) | **~6.7 h** |

  Starting or resuming a run adds about 45 s (AMP check, label cache, AutoBatch), and the
  final test evaluation about 1.5 min. Before the first real epoch, a synthetic stand-in
  of the dataset (larger JPEGs) had predicted 10–11 min per epoch. Keep the laptop on
  mains power, in a performance power mode, with sleep disabled.
- Do not let the PC sleep during training: stop the run with `--request-stop` and
  resume it instead (see above).
- Test evaluation by hand, for example of a run resumed with plain Ultralytics instead of
  `--resume` (which runs the test evaluation itself):

  ```powershell
  .\.venv\Scripts\python -c "from ultralytics import YOLO; m = YOLO(r'..\runs\detector\yolo11s_uav_gray_1280_e40\weights\best.pt'); r = m.val(data=r'..\datasets\yolo_uav_gray\data.yaml', split='test', imgsz=1280); print(r.box.map50, r.box.map)"
  ```

### 8. Live runtime with the IDS camera

`control\cuasp.py` runs the **SiamFC** pipeline by default
(`CV_TARGET_TRACKER = "siamfc"` in `control\suncubes\settings.py`, camera
`provider_mode = "ids_siamfc"`); set it to `"aruco"` for the original C-UASP ArUco
pipeline. In SiamFC mode `cuasp.py` needs the `siamfc` extra (torch, torchvision,
ultralytics; step 3), the SiamFC weights (step 4) and the detector weights (step 7). The
start-up banner prints the weight paths and the SHA256 actually loaded; a hash that does
not match `CV_SIAMFC_WEIGHTS_SHA256` / `CV_DETECTOR_WEIGHTS_SHA256` stops the start-up
(set the setting to `""` to skip the check, or update it after converting the weights on
another machine). The fixed range for the parallax correction is
`CV_SIAMFC_TARGET_RANGE_M` (21 m). Before any live run:

- install the IDS peak SDK (IDS installer) and its Python bindings into the venv:
  `.\.venv\Scripts\python -m pip install ids_peak==1.17.0 ids_peak_ipl==1.18.0`. Check with
  the camera connected that `ids_peak.DeviceManager` lists it (the GenTL producers GEVK,
  U3VK and uEye are found automatically on the training PC). Set `CV_IDS_GENTL_CTI_PATH`
  in `control\suncubes\settings.py` only if the SDK needs an explicit GenTL producer path;
- note on `camera.py` (inherited from C-UASP): `_buffer_to_mono8_numpy()` calls
  `ConvertTo("Mono8")` with a string, which `ids_peak_ipl` 1.18 rejects (it expects a
  `PixelFormatName`); the error is swallowed and the image is used as delivered. This is
  correct as long as the camera really delivers Mono8 (`PixelFormat` is set at start-up).
  On the first run with the camera, check that the preview of `tools/record_ids.py` looks
  right;
- check the network settings of the motor controller: `IP` / `PORT` in
  `control\suncubes\settings.py` (currently `192.168.0.109:8123`);
- motor commands are **disabled by default** in this copy
  (`CV_CUAS_ENABLE_MOTOR_COMMANDS = 0`, decision of 2026-11-10; it is 1 in C-UASP), as
  required by the C-UASP safety rules. Enable them only deliberately, for the motors-on
  bench test (V8);
- **first camera test (2026-11-16)**: the camera is an IDS **U3-33DxXCP-M** (Sony
  **IMX900**, global shutter, 2064x1552 sensor, 2.25 µm pixels, firmware 3.80.26143). It
  boots from `UserSet1` with a **2040x1536 ROI at offset (8, 8)**. It could not deliver
  the 2592x1944 that C-UASP's `settings.py` requested (larger than the sensor):
  `_try_set_node` swallowed the rejection and the camera kept its ROI. `settings.py` now
  requests 2040x1536. That is the calibration size, so the runtime's
  `_scale_intrinsics_to_runtime` (which uses the real frame size) applies a factor of 1,
  and the 27 % focal ambiguity inherited from C-UASP does not occur with this camera.
  Notes on the calibration:
  - the user chose (2026-11-16) to keep `calibration_legacy_20260921_x_test.yaml`
    (fx 24119 px, cx 1140), the override already used by `cuasp.py`, over the more recent
    `calibration.yaml` (fx 21062 px, cx 697), accepting the inconsistencies;
  - with 2.25 µm pixels these fx values correspond to a 47–54 mm focal length, while both
    YAMLs name an "Edmund_Optics_16mm_UC" lens (fx ≈ 7100 px);
  - the calibration is only valid for the ROI it was computed with (2040x1536 at (8, 8),
    stored in `UserSet1`). The runtime does not set the offsets: do not change the ROI
    or save user sets with IDS peak Cockpit;
- **frame rate**: 43.2 fps at first (zero losses). The limit was not the exposure but the
  camera's `DeviceLinkThroughputLimit` (162 MB/s from `UserSet1`, link speed 500 MB/s).
  The runtime now sets it (`CV_IDS_LINK_THROUGHPUT_LIMIT_BPS` = 300 MB/s), and with the
  15 ms exposure chosen on 2026-11-16 the camera runs at **62.6 fps** (verified: 0 frames
  lost, 16.0 ms per frame). Maximum frame rate reported by the camera at this ROI:

  | Exposure | 162 MB/s | 300 MB/s |
  |---|---|---|
  | 20 ms | 43.2 fps | 47.7 fps |
  | 15 ms | 43.2 fps | 62.6 fps |
  | ≤ 12 ms | 43.2 fps | 67.9 fps |

  70 Hz is out of reach at 2040x1536. The supervisor loop (`CV_CUAS_LOOP_HZ` = 70) only
  submits motor commands on new frames (`_is_new_visual_sample`), so a slower camera is
  safe;
- **gain units**: on this camera `Gain` is a linear factor (1.0–15.85, i.e. 0–24 dB).
  `CV_IDS_GAIN_DB` is now converted explicitly (behaviour change 3 above). +2.5 dB
  restores the brightness of the old 20 ms exposure. The setting is +6 dB since 2026-11-17
  (factor 1.995). With 0 dB the detector lost the drone in front of the door and dark
  corners of the room: 34-69 % tracking against 87-97 % at +6 dB, recordings B6 in
  `PLAN.md`;
- `tools/record_ids.py` (v2) records what the camera really did:
  - the actual frame size and sampled gray levels;
  - a read-only readback of the camera nodes after the recording (`camera.readback`);
  - `checks`: mismatches against the requested size and the calibration, saturated or
    very dark frames.

```powershell
cd control
..\.venv\Scripts\python cuasp.py
```

## Network weights: CFNet "Baseline-conv5", not the original 2016 SiamFC

> **Important.** The tracker does **not** use the original SiamFC-3s network
> (`2016-08-17.net.mat`, ECCVW 2016) that the MATLAB code in this repository
> (`../tracking/tracker.m`) was written for: that file is no longer
> downloadable (the project page now returns 404).
>
> It uses instead the **"Baseline-conv5" (improved SiamFC)** network published
> by the same authors with CFNet:
> J. Valmadre, L. Bertinetto, J. F. Henriques, A. Vedaldi, P. H. S. Torr,
> *End-to-end representation learning for Correlation Filter based tracking*,
> CVPR 2017 — <https://github.com/bertinetto/cfnet>.

The runtime uses the **grayscale-trained** variant
`baseline-conv5_gray_e100.mat`, which matches the Mono8 IDS camera. The RGB variant
`baseline-conv5_e55.mat` is kept for benchmark comparisons only.

Source: `cfnet-networks.zip` (linked as `https://bit.ly/cfnet_networks` from the
`torrvision/siamfc-tf` README) ->
`ftp://ftp.robots.ox.ac.uk/pub/outgoing/tvg/cfnet/cfnet-networks.zip`,
downloaded 2026-10-06 into `../pretrained/` (gitignored).

| File | SHA256 |
|---|---|
| `cfnet-networks.zip` | `440e9125a0c377c29523954cf61d07342cd5384ceb13792b3dcdb342a2743579` |
| `networks/baseline-conv5_gray_e100.mat` | `2cefa19ae7a69092f3beaf0c6a9ccc048dc483a67c3f9d23e483809a6a7d577f` |
| `networks/baseline-conv5_e55.mat` | `3df4b30801c5472dff5a6fd5f1b3de63a46e581a84877def8b63a0a43561e3e6` |

Differences from the original SiamFC that the port follows (reference code:
`../reference/cfnet/*.m`, downloaded unmodified from
<https://github.com/bertinetto/cfnet/tree/master/src/tracking>):

- architecture: same AlexNet-like branch, but `pool2` has stride 1 (total
  stride 4 instead of 8), and `conv5` outputs 32 channels instead of 256.
  The score map is 33x33 instead of 17x17;
- the final `adjust` layer is a scalar BatchNorm instead of a gain + bias;
- tracker logic from CFNet's `tracker.m` / `tracker_step.m` /
  `make_scale_pyramid.m`: `s_x = 255/127 * s_z`, a two-stage scale pyramid,
  and a rolling-average template update (`zLR`);
- hyper-parameters from CFNet `run_baseline5_evaluation.m` (paper Table 2): scaleStep
  1.0470, scalePenalty 0.9825, scaleLR 0.68, wInfluence 0.175, zLR 0.0102, responseUp 8.

## Licences of third-party components

| Component | Licence | Notes |
|---|---|---|
| CFNet / SiamFC reference code and weights | See the CFNet and siamese-fc repositories | Weights downloaded from the authors' FTP |
| Ultralytics (`ultralytics`, YOLO11) — drone detector | **AGPL-3.0** | Accepted for system validation (decision of 2026-11-06). Using it in a product that is not released as open source requires an Ultralytics Enterprise licence, or a detector under a permissive licence (for example YOLOX / RT-DETR, Apache-2.0 in their original implementations). |
| DUT-Anti-UAV, Anti-UAV300 datasets | Research datasets, see their repositories | Used for training and validation only |

## Known reference quirks (reproduced on purpose)

- `get_subwindow_tracking.m` places a crop of side `sz` at
  `round(pos - (sz + 1) / 2)` (1-based), so every crop is centred **one pixel
  before** `pos`. For the largest scale of the two-stage pyramid this means
  one row/column of mean-value padding. The port reproduces it exactly.
  The exemplar is cropped with the same offset, so it largely cancels: on
  synthetic Mono8 sequences (`tests/test_siamfc_tracker_synthetic.py`, 3 seeds
  x 60 frames) the mean signed centre error is < 0.2 px and the max error is
  ~1 px. The residual bias on real data is still to be measured against the
  ArUco reference in validation V7.
- MATLAB `imresize` bicubic (Keys a = -0.5, antialiasing when shrinking,
  symmetric borders) is reproduced exactly with separable weight matrices
  (`siamfc/imresize.py`), instead of OpenCV `INTER_CUBIC` (a = -0.75, no
  antialiasing).
- The response peak is searched in MATLAB column-major order (`find(..., 1)`),
  and the scale choice keeps the first maximum.

## Implementation notes

- Crops are copied to the GPU in the frame's own dtype (uint8 for the IDS Mono8
  frames). Only the in-image part is copied, and the mean-value padding and the
  float32 conversion happen on the device (`SiamFCTracker._crop_to_device`). The values
  are identical to the NumPy reference `crop_with_padding`, which is checked by
  `tests/test_siamfc_crop_device.py`.

## Validation status

| Level | Result |
|---|---|
| V1 conversion | `tests/test_siamfc_convert.py`: the folded PyTorch net matches the unfolded MatConvNet semantics (synthetic DagNN). Gray input == replicated RGB. No MATLAB available, so there is no tensor-level comparison against MatConvNet itself. |
| V2 demo sequence | `tools/eval_sequence.py ../demo-sequences/vot15_bag` (196 frames, 480x360). Gray net on gray frames: mean IoU 0.796, IoU>0.5 on 99.5 % of frames, mean centre error 5.9 px. RGB net on RGB frames: IoU 0.773, 6.4 px. CPU and CUDA give identical results. |
| V5 latency, RTX 4060 Laptop (fp32, uint8 crop transfer, 2026-11-06) | IDS-sized 2592x1944 Mono8 frames (`tools/bench_siamfc.py`, zLR on): 16-150 px targets take 6.9-7.0 ms mean and 8.0-10.8 ms p99; 400 px targets take 8.9 ms mean and 11.7 ms p99 (15.3 / 17.7 ms before the uint8 transfer). Stage breakdown at 48 px: backbone x3 4.4 ms, template update 1.1 ms, crop+H2D+resize 0.45 ms, xcorr+upsample+peak 0.5 ms. Demo sequence (480x360): 7.9-8.1 ms mean. CPU i7-13700H: ~98 ms. |
| V5 latency, RTX 5060 Laptop, training PC (fp32, uint8 crop transfer, torch 2.14.1+cu130, 2026-11-06) | IDS-sized 2592x1944 Mono8 frames (`tools/bench_siamfc.py`, zLR on): 16-150 px targets take 5.9-6.4 ms mean and 7.1-8.7 ms p99; 400 px targets take 8.1 ms mean and 9.2 ms p99. With zLR = 0: 4.7-5.1 ms mean (16-150 px), 6.5 ms (400 px). Stage breakdown at 48 px: backbone x3 3.4 ms, template update 1.2 ms, crop+H2D+resize 0.42 ms, xcorr+upsample+peak 0.47 ms. Demo sequence (480x360, gray): 6.4 ms mean, 8.6 ms p99, with the same accuracy as on the laptop (IoU 0.796, 5.9 px). |
| V4 baseline, IDS recordings (2026-11-18) | 29 labelled recordings (`../labels/ids/`, reviewed by the user), `tools/replay_siamfc_source.py --gt`, current settings, real time at 62.6 fps; summary with `tools/v4_report.py` (`../runs/v4_baseline/report.json`).<br>Coverage 85.3 %; centre error 0.40 mrad median, 1.21 mrad p90; re-acquisition after re-entry 0.43 s median; 0 false alarms in 16 min of empty scene.<br>Missed frames: mostly before the lock (detector scores 0.2-0.43, below `CV_SIAMFC_ACQUIRE_MIN_SCORE`).<br>Dangerous frames: 29 wrong target and 26 drone absent, all from spontaneous recoveries out of "lost" just above `CV_SIAMFC_LOST_CONF`. Hence `CV_SIAMFC_RECOVER_CONF` / `CV_SIAMFC_RECOVER_EDGE_PX` (proposal P6 in `../PLAN.md`), implemented with defaults that keep this behaviour. Values to be chosen after the V4 replays. |
| V5 latency, training PC, IDS camera size, detector on the same GPU (2026-11-16) | **SiamFC alone**, 2040x1536 frames (`tools/bench_siamfc.py`, new default size): 6.0-7.8 ms mean, p99 ≤ 8.6 ms (16-400 px targets, zLR on).<br>**Full `ids_siamfc` source**: `read_candidate` with the YOLO detector asynchronous on the same GPU (`tools/replay_siamfc_source.py --resize 2040 1536 --fps 62.6 --preload`, DUT video01/05/10/15, 3450 frames): median 6.3 ms, p95 12.4 ms, p99 17.6 ms, max 33 ms.<br>Budget at 62.6 fps: 16.0 ms. 2.3 % of the frames exceed it, never two in a row. They come about every 13 frames (about 5 Hz), i.e. when the tracking verification runs the detector: GPU/CPU contention.<br>Effect at runtime: the next frame is taken a few ms late (rarely one frame is skipped; latest-only slot). Motor commands are capped at 50 Hz (`CV_CUAS_MOTOR_COMMAND_HZ`) and samples stay fresh for 250 ms, so this has no practical effect. Known remedy if needed: call the YOLO model directly instead of Ultralytics `predict` (about 8 ms of overhead; PLAN, Phase 3). |

## Layout

```
control/suncubes/siamfc/      new: SiamFC port (convert_matconvnet, net, tracker, imresize)
                              and drone detector for init/re-acquisition (detector)
tools/eval_sequence.py        new: VOT-format sequence evaluation (accuracy + latency)
tools/bench_siamfc.py         new: latency benchmark on IDS-sized Mono8 frames
tools/prepare_yolo_dataset.py new: builds the grayscale YOLO detector dataset
tools/train_detector.py       new: detector training (Ultralytics YOLO), with clean stop/resume
tools/eval_detector.py        new: detector test breakdown by source and drone size
tools/replay_siamfc_source.py new: replays a recorded sequence (images or video, optional GT) through
                              the full ids_siamfc source in real time: safety metrics, per-frame CSV
tools/record_ids.py           new: records every IDS frame with its timestamp, using the runtime camera
                              settings (procedure: ../RECORDING_PROTOCOL.md); --fake for dry runs
tools/prelabel_ids.py         new: automatic per-frame labels of IDS recordings (detector + SiamFC
                              forward/backward), uncertain frames flagged -> ../labels/ids/<rec>.csv
tools/review_labels.py        new: OpenCV review/correction of the labels (play, draw, absent,
                              interpolate, re-track, undo); keys in its docstring or with h
tools/ids_labels.py           new: label format and pre-label fusion shared by the two tools
tools/v4_report.py            new: V4 summary of replays against the labels (safety, coverage with and
                              without border frames, clipped IoU, mrad error, lock/re-lock/exit times)
../labels/ids/                IDS labels (edge coordinates; read by replay_siamfc_source.py --gt)
control/suncubes/siamfc_source.py  new: ids_siamfc camera correction source (SiamFC + detector)
control/cuasp.py, suncubes/   copies of C-UASP (integration changes go here)
calibration/                  copies of the C-UASP calibration YAMLs (read-only)
tests/                        copied C-UASP tests + new SiamFC tests
../pretrained/                SiamFC weights (gitignored)
../datasets/                  datasets + MANIFEST.md (data gitignored)
../runs/                      training outputs (gitignored)
```
