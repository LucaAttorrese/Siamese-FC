# SiamMask feasibility study (proposal P7, step 0)

Not used by the runtime: `control/` does not import anything from here. SiamFC stays the tracker
of the `ids_siamfc` source. See `../../../PLAN.md`, "Proposta P7".

## Provenance

| Item | Source | Size | SHA256 |
|---|---|---|---|
| SiamMask source (reference only) | `https://github.com/foolwood/SiamMask/archive/a2b3ce3dca23b3a911d96b791e007eec6d39a625.zip` (commit of 2025-02-14), unpacked in `../../../reference/siammask_dl/` | 7 354 131 B | `2a2a1dd359dc0fc974f93fe1e41b1d9238e6e1db14db539f2b7e8149e233775c` |
| Weights `SiamMask_VOT.pth` (siammask_sharp, refine) | `https://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth`, in `../../../pretrained/siammask/` (gitignored) | 101 533 952 B | `3fea9718e56af5c5c42f817ab194d487e66f816398f34a640a962333dd4c0e94` |

License: MIT, Copyright (c) 2019 Qiang Wang. `net.py` and `tracker.py` are ports of
`experiments/siammask_sharp/{custom,resnet}.py`, `models/{siammask_sharp,rpn}.py` and
`tools/test.py`, and keep the copyright notice in their docstrings.

The checkpoint is a 2019 legacy pickle. It is loaded with `torch.load(weights_only=True)`; the
only non-tensor globals it references (checked with `pickletools`) are `OrderedDict`, numpy
`dtype` / scalar and `_codecs.encode`, which `net.py` allow-lists. All 356 parameters load, none
missing or unexpected. The stored `anchor_cfg` matches `config_vot.json`.

## Files

- `net.py`: the network, same module names as the original so that the state dict loads as is.
- `tracker.py`: `SiamMaskTracker` with the `init` / `update` interface of `siamfc.tracker`
  (centre, size, peak, APCE), plus mask outputs. Faithful path (`crop="cv2"`, `mask="full"`)
  and fast path (`crop="torch"`, `mask="window"`, optional FP16).
- `bench.py`: latency on 2040x1536 frames with a stage breakdown; `fold_bn()` folds the batch
  norms into the convolutions (needed for FP16 to be faster than FP32).
- `compare_vot.py`: demo sequence, faithful against fast path, and accuracy against SiamFC.
- `compare_ids.py`: tracker-only comparison with SiamFC on the labelled IDS recordings.

## Commands (from `cuasp_siamfc/`)

```powershell
.\.venv\Scripts\python experiments\siammask\compare_vot.py ..\demo-sequences\vot15_bag --weights ..\pretrained\siammask\SiamMask_VOT.pth
.\.venv\Scripts\python experiments\siammask\bench.py --weights ..\pretrained\siammask\SiamMask_VOT.pth --frame ..\recordings\20261117_102552_B6_sfondi_6db_r3\frames\000000.jpg --box 575 790 287 69
.\.venv\Scripts\python experiments\siammask\compare_ids.py --out ..\runs\siammask_step0 <recording> [...]
```
