# cuasp_siamfc — SiamFC target tracking for the C-UASP pointing system

Python port of the SiamFC tracker, integrated as a camera correction source for
a copy of the C-UASP runtime (`control/cuasp.py` + `control/suncubes/`). The
goal is to replace the ArUco/PnP target measurement with visual tracking of the
drone itself, keeping the `CameraCorrection` -> PBVS -> PCMM/KAS motor
dispatch contract unchanged.

The C-UASP repository is **never modified**: the files it contributes were
copied here verbatim first (see [PROVENANCE.md](PROVENANCE.md)) and all
integration changes are made on these copies.

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
- hyper-parameters from CFNet `run_baseline5_evaluation.m`: scaleStep 1.0470,
  scalePenalty 0.9825, scaleLR 0.68, wInfluence 0.175, zLR 0.0102,
  responseUp 8.

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

## Validation status

| Level | Result |
|---|---|
| V1 conversion | `tests/test_siamfc_convert.py`: the folded PyTorch net matches the unfolded MatConvNet semantics (synthetic DagNN). Gray input == replicated RGB. No MATLAB available, so there is no tensor-level comparison against MatConvNet itself. |
| V2 demo sequence | `tools/eval_sequence.py ../demo-sequences/vot15_bag` (196 frames, 480x360). Gray net on gray frames: mean IoU 0.796, IoU>0.5 on 99.5 % of frames, mean centre error 5.9 px. RGB net on RGB frames: IoU 0.773, 6.4 px. CPU and CUDA give identical results. |
| V5 latency (unoptimised, fp32) | RTX 4060 Laptop: gray 8.7 ms mean / 13.0 ms p99 per update (~115 Hz). CPU i7-13700H: ~98 ms. |

## Layout

```
control/suncubes/siamfc/      new: SiamFC port (convert_matconvnet, net, tracker, imresize)
tools/eval_sequence.py        new: VOT-format sequence evaluation (accuracy + latency)
control/cuasp.py, suncubes/   copies of C-UASP (integration changes go here)
calibration/                  copies of the C-UASP calibration YAMLs (read-only)
tests/                        copied C-UASP tests + new SiamFC tests
```

## Setup

```
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install numpy opencv-contrib-python PyYAML scipy pytest pytest-asyncio
.venv/Scripts/python -m suncubes.siamfc.convert_matconvnet ../pretrained/networks/baseline-conv5_gray_e100.mat
```
