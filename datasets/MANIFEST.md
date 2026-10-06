# Datasets manifest

Contents of this folder are gitignored (except this file). Downloaded 2026-10-06.

## DUT-Anti-UAV — <https://github.com/wangdongdut/DUT-Anti-UAV>

The zip archives were deleted after extraction (2026-10-06); sizes and hashes below refer
to the original downloads. Only the extracted folders remain (~4.9 GB).

Visible-light (RGB) drone images. Used for the Phase 2 detector training (detection
split) and for SiamFC validation on drones (V4, tracking split).

| File (Google Drive) | Bytes | SHA256 | Content |
|---|---|---|---|
| `DUT-Anti-UAV/detection/train.zip` | 744616230 | `14f927290556df60e23cedfa80dffc10dc21e4a3b6843e150cfc49644376eece` | 5200 jpg + VOC xml |
| `DUT-Anti-UAV/detection/val.zip` | 372283691 | `238be0ceb3e7c5be6711ee3247e49df2750d52f91f54f5366c68bebac112ebf8` | 2600 jpg + VOC xml |
| `DUT-Anti-UAV/detection/test.zip` | 271153425 | `a671989a01cff98c684aeb084e59b86f4152c50499d86152eb970a9fc7fb1cbe` | 2200 jpg + VOC xml |
| `DUT-Anti-UAV/tracking/Anti-UAV-Tracking-V0.zip` | 3637903273 | `5f3bb7150d38e613243397555405258674776b795b90b1e8c98c5259fe1bd72f` | 20 videos, 24804 jpg |
| `DUT-Anti-UAV/tracking/gt.zip` | 90836 | `8722a642e2f0e0981faf728efff4bd19e0e0e1b65df723e0a4ed1f11c972648f` | 20 `videoNN_gt.txt` (x y w h) |

Detection statistics: 10000 images; 79 % 1920x1080 and 18 % 1280x720; 99 % have a single
object (class `UAV`). Box size (sqrt(area) relative to sqrt(image area)): 5th / 50th / 95th
percentile = 0.011 / 0.023 / 0.244, so most targets are small.

## Anti-UAV300 (RGB + IR) — <https://github.com/ZhaoJ9014/Anti-UAV>

**NOT downloaded yet.** Google Drive file `1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr`
(`Anti-UAV-RGBT.zip`, 5.6 GB) returned "Quota exceeded — too many users have downloaded
this file recently" on 2026-10-06. Retry later. The Baidu mirror (password `sagx`) needs
a Baidu account.

Excluded on purpose: Anti-UAV410 / Anti-UAV600 (infrared only, not useful for the
visible-light IDS camera).
