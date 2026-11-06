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
| Detector training (YOLO11s) | Script ready (`tools/train_detector.py`); first run stopped on the laptop (out of virtual memory), to be run on the training PC (RTX 5060 Laptop, estimated ~7–7.5 h for 40 epochs, see §7) |
| Camera adapter `ids_siamfc` and its wiring into `cuasp.py` / `settings.py` | **Not implemented yet**: `control/cuasp.py` is still the unchanged C-UASP copy and runs the **ArUco** pipeline |
| Validation on drone data, hyperparameter search, bench tests (V3, V4, V4b, V6–V8) | Not started |

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
| Live camera only | IDS peak SDK with its Python bindings (`ids_peak`, `ids_peak_ipl`), from the IDS installer (not on PyPI). Not needed for training or offline tests. |

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
- Duration: measure the first epoch before planning. References at 1280 px, batch 3,
  `--workers 2`:
  - development laptop (RTX 4060 Laptop): one epoch ~16 min of training plus validation;
  - training PC (RTX 5060 Laptop), measured on 2026-11-06 on a synthetic stand-in of
    `yolo_uav_gray` (600 train / 300 val grayscale JPEGs, 1920x1080 and 1280x720, same
    training settings as `train_detector.py`): 24 train images/s and 78 val images/s in
    steady state. That gives ~9 min of training plus ~1–2 min of validation per epoch on
    the real set, so **~7–7.5 h for 40 epochs**. The synthetic JPEGs are about 5 times
    larger than the real ones, so this estimate is on the pessimistic side; check it
    against the first real epoch. Keep the laptop on mains power, in a performance power
    mode, with sleep disabled.
- The run can be stopped at any time and resumed from the last completed epoch:

  ```powershell
  .\.venv\Scripts\python -c "from ultralytics import YOLO; YOLO(r'..\runs\detector\yolo11s_uav_gray_1280_e40\weights\last.pt').train(resume=True)"
  ```

  `last.pt` is written after each epoch's validation, so only the epoch in progress is
  lost. Resuming keeps the original settings: epochs, image size and batch cannot be
  changed. Do not let the PC sleep during training: stop the run and resume it instead.
- Test evaluation of a resumed run (the resume command does not run it):

  ```powershell
  .\.venv\Scripts\python -c "from ultralytics import YOLO; m = YOLO(r'..\runs\detector\yolo11s_uav_gray_1280_e40\weights\best.pt'); r = m.val(data=r'..\datasets\yolo_uav_gray\data.yaml', split='test', imgsz=1280); print(r.box.map50, r.box.map)"
  ```

### 8. Live runtime with the IDS camera

The SiamFC camera adapter is **not implemented yet** (see "Project status"), so
running `control\cuasp.py` today runs the unchanged C-UASP **ArUco** pipeline on the
copied files. Before any live run on the new PC:

- install the IDS peak SDK and its Python bindings into the venv, following the IDS
  instructions. Set `CV_IDS_GENTL_CTI_PATH` in `control\suncubes\settings.py` if the
  SDK needs an explicit GenTL producer path;
- check the network settings of the motor controller: `IP` / `PORT` in
  `control\suncubes\settings.py` (currently `192.168.0.109:8123`);
- start with motor commands **disabled**: `CV_CUAS_ENABLE_MOTOR_COMMANDS = 0` in
  `settings.py` (currently 1), as required by the C-UASP safety rules;
- known open point, inherited from C-UASP: the calibration YAMLs are computed at
  2040x1536, while `settings.py` acquires 2592x1944 and `camera.py` rescales the
  intrinsics (fx ≈ 21060 → ≈ 26760 px). The comment in the YAML itself says the real
  acquisition resolution is unclear. This affects every pixel-to-angle conversion and
  must be resolved before the angular validation (V7);
- `settings.py` sets an exposure of 20 ms (`CV_IDS_EXPOSURE_TIME_US = 20000`) with a
  70 Hz frame-rate target. A 20 ms exposure caps the camera at about 50 fps; check the
  real frame rate on the bench.

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
| V5 latency, RTX 5060 Laptop, training PC (fp32, uint8 crop transfer, torch 2.14.1+cu130, 2026-11-06) | IDS-sized 2592x1944 Mono8 frames (`tools/bench_siamfc.py`, zLR on): 16-150 px targets take 5.9-6.4 ms mean and 7.1-8.7 ms p99; 400 px targets take 8.1 ms mean and 9.2 ms p99. With zLR = 0: 4.7-5.1 ms mean (16-150 px), 6.5 ms (400 px). Stage breakdown at 48 px: backbone x3 3.4 ms, template update 1.2 ms, crop+H2D+resize 0.42 ms, xcorr+upsample+peak 0.47 ms. Demo sequence (480x360, gray): 6.4 ms mean, 8.6 ms p99, with the same accuracy as on the laptop (IoU 0.796, 5.9 px). Not yet measured with the detector running on the same GPU. |

## Layout

```
control/suncubes/siamfc/      new: SiamFC port (convert_matconvnet, net, tracker, imresize)
tools/eval_sequence.py        new: VOT-format sequence evaluation (accuracy + latency)
tools/bench_siamfc.py         new: latency benchmark on IDS-sized Mono8 frames
tools/prepare_yolo_dataset.py new: builds the grayscale YOLO detector dataset
tools/train_detector.py       new: detector training (Ultralytics YOLO)
control/cuasp.py, suncubes/   copies of C-UASP (integration changes go here)
calibration/                  copies of the C-UASP calibration YAMLs (read-only)
tests/                        copied C-UASP tests + new SiamFC tests
../pretrained/                SiamFC weights (gitignored)
../datasets/                  datasets + MANIFEST.md (data gitignored)
../runs/                      training outputs (gitignored)
```
