# Datasets manifest

Contents of this folder are gitignored (except this file). Downloaded 2026-10-06 / 2026-10-07.
The zip archives were deleted after extraction; sizes and hashes below refer to the original
downloads, so a new download can be checked against them (`sha256sum <file>`).

## Download links

| Dataset | Google Drive | Baidu mirror | Size |
|---|---|---|---|
| DUT-Anti-UAV detection `train.zip` | [1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj](https://drive.google.com/open?id=1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj) | [link](https://pan.baidu.com/s/1-ogC7P_K6lwYAqIS8bgIUQ), code `u955` | 710 MB |
| DUT-Anti-UAV detection `val.zip` | [1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n](https://drive.google.com/open?id=1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n) | [link](https://pan.baidu.com/s/15sekmPn0hYNQS05Makbmtw), code `wkzs` | 355 MB |
| DUT-Anti-UAV detection `test.zip` | [1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0](https://drive.google.com/open?id=1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0) | [link](https://pan.baidu.com/s/1GiA-bKlvMSBkzUwYvo-RiA), code `ik4d` | 259 MB |
| DUT-Anti-UAV tracking images `Anti-UAV-Tracking-V0.zip` | [1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9](https://drive.google.com/open?id=1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9) | [link](https://pan.baidu.com/s/1OTExqKgvUnqpENtTDu_gGQ), code `oine` | 3.4 GB |
| DUT-Anti-UAV tracking groundtruth | [16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq](https://drive.google.com/open?id=16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq) | [link](https://pan.baidu.com/s/1nkGNERDVgmYIAiwFTdj2xA), code `e8mr` | 90 KB |
| Anti-UAV300 RGB+IR `Anti-UAV-RGBT.zip` | [1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr](https://drive.google.com/file/d/1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr/view) | [link](https://pan.baidu.com/s/1dJR0VKyLyiXBNB_qfa2ZrA), password `sagx` | 5.6 GB |

Sources: the dataset READMEs, <https://github.com/wangdongdut/DUT-Anti-UAV> and
<https://github.com/ZhaoJ9014/Anti-UAV>.

Pretrained SiamFC weights (not a dataset, but needed on every machine): CFNet
`cfnet-networks.zip`, `ftp://ftp.robots.ox.ac.uk/pub/outgoing/tvg/cfnet/cfnet-networks.zip`
(62.8 MB, SHA256 `440e9125a0c377c29523954cf61d07342cd5384ceb13792b3dcdb342a2743579`),
unzipped into `../pretrained/` (see `cuasp_siamfc/README.md`).

### Rebuilding this folder on another machine

Large Google Drive files show a virus-scan confirmation page. The command-line download
below skips it. If Drive answers with a ~2 KB HTML page saying "Quota exceeded", the file
has been downloaded too often: retry later (up to 24 h), or use the Baidu mirror.

```bash
# from Siamese-FC/datasets
dl() { curl -L --fail --retry 3 -C - -o "$1" "https://drive.usercontent.google.com/download?id=$2&export=download&confirm=t"; }
mkdir -p DUT-Anti-UAV/detection DUT-Anti-UAV/tracking Anti-UAV300
dl DUT-Anti-UAV/detection/train.zip 1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj
dl DUT-Anti-UAV/detection/val.zip   1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n
dl DUT-Anti-UAV/detection/test.zip  1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0
dl DUT-Anti-UAV/tracking/gt.zip     16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq
dl DUT-Anti-UAV/tracking/Anti-UAV-Tracking-V0.zip 1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9
dl Anti-UAV300/Anti-UAV-RGBT.zip    1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr
sha256sum DUT-Anti-UAV/*/*.zip Anti-UAV300/*.zip   # compare with the tables below
```

Extraction (Python, so it works the same on Windows and Linux):

```python
import zipfile
for z in ["train", "val", "test"]:
    zipfile.ZipFile(f"DUT-Anti-UAV/detection/{z}.zip").extractall("DUT-Anti-UAV/detection")
for z in ["gt", "Anti-UAV-Tracking-V0"]:
    zipfile.ZipFile(f"DUT-Anti-UAV/tracking/{z}.zip").extractall("DUT-Anti-UAV/tracking")
# Anti-UAV300: visible-light part only (the infrared videos are not used)
zf = zipfile.ZipFile("Anti-UAV300/Anti-UAV-RGBT.zip")
zf.extractall("Anti-UAV300/Anti-UAV-RGBT",
              [n for n in zf.namelist() if not n.endswith("/") and ("/visible." in n or n.startswith("label_new/") or n == "framecut.py")])
```

Then rebuild the detector training set (from `cuasp_siamfc/`, about 1 h on a laptop):

```bash
.venv/Scripts/python tools/prepare_yolo_dataset.py --out ../datasets/yolo_uav_gray --anti-uav300 --anti-uav-stride 20
```

Expected result (`yolo_uav_gray/counts.json`): dut 5200/2600/2200 and anti_uav300
7499/3109/4280 images (train/val/test).

## DUT-Anti-UAV

Visible-light (RGB) drone images. Used for the Phase 2 detector training (detection
split) and for SiamFC validation on drones (V4, tracking split). Extracted size ~4.9 GB.

| File | Bytes | SHA256 | Content |
|---|---|---|---|
| `DUT-Anti-UAV/detection/train.zip` | 744616230 | `14f927290556df60e23cedfa80dffc10dc21e4a3b6843e150cfc49644376eece` | 5200 jpg + VOC xml |
| `DUT-Anti-UAV/detection/val.zip` | 372283691 | `238be0ceb3e7c5be6711ee3247e49df2750d52f91f54f5366c68bebac112ebf8` | 2600 jpg + VOC xml |
| `DUT-Anti-UAV/detection/test.zip` | 271153425 | `a671989a01cff98c684aeb084e59b86f4152c50499d86152eb970a9fc7fb1cbe` | 2200 jpg + VOC xml |
| `DUT-Anti-UAV/tracking/Anti-UAV-Tracking-V0.zip` | 3637903273 | `5f3bb7150d38e613243397555405258674776b795b90b1e8c98c5259fe1bd72f` | 20 videos, 24804 jpg |
| `DUT-Anti-UAV/tracking/gt.zip` | 90836 | `8722a642e2f0e0981faf728efff4bd19e0e0e1b65df723e0a4ed1f11c972648f` | 20 `videoNN_gt.txt` (x y w h) |

Detection statistics: 10000 images; 79 % 1920x1080 and 18 % 1280x720; 99 % have a single
object (class `UAV`). Box size (sqrt(area) relative to sqrt(image area)): 5th / 50th / 95th
percentile = 0.011 / 0.023 / 0.244, so most targets are small.

## Anti-UAV300 (RGB + IR)

`Anti-UAV-RGBT.zip`, 6037566331 bytes, SHA256
`ed7d80bbd8ca8e01ea784c64eb0782eae8b3a1572177535c0fada9beb53fdca8` (the first download
attempt on 2026-10-06 hit the Google Drive quota; it worked on 2026-10-07).

Only the **visible-light** part was extracted, into `Anti-UAV300/Anti-UAV-RGBT/` (5.3 GB):
- `<split>/<video>/visible.mp4` + `visible.json` (`exist[i]` flag, `gt_rect[i] = [x, y, w, h]`);
- 318 videos: train 160, val 67, test 91. 296901 frames at 1920x1080 and 20 fps; target
  present in 94.4 % of frames;
- `label_new/<split>.json`: per-video attribute tags only (for example `TC`, `TC-EASY`), no boxes;
- `framecut.py`: the authors' frame-extraction helper (not used).
- The videos contain camera overlay graphics (white/black reticle lines), which the IDS
  camera does not have.

Excluded on purpose: Anti-UAV410 (Baidu only, password `wfds`) and Anti-UAV600
(ModelScope): infrared only, not useful for the visible-light IDS camera.

## Derived: `yolo_uav_gray/` (detector training set)

Built by `cuasp_siamfc/tools/prepare_yolo_dataset.py --anti-uav300 --anti-uav-stride 20`:
grayscale JPEGs (quality 95) with YOLO labels (single class `UAV`); 3.6 GB. Image name
prefixes `dut_*` and `auav_*` identify the source, so metrics can be computed per source.
Anti-UAV300 is sampled at 1 frame every 20 (1 frame/s); frames with the target absent are
kept as negatives (empty label file).

| Split | DUT | Anti-UAV300 | Total | Negatives |
|---|---|---|---|---|
| train | 5200 | 7499 | 12699 | 374 |
| val | 2600 | 3109 | 5709 | 190 |
| test | 2200 | 4280 | 6480 | 276 |
