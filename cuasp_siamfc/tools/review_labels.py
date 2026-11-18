"""Review and correct the pre-labels of an IDS recording (validation V4).

Plays the recording with the label box drawn on it. You stop only where the box is wrong
and fix it: draw a new box with the mouse, mark frames absent, interpolate, or re-track
with SiamFC from a corrected box. Flagged frames (uncertain pre-labels) are red on the
timeline; ``n`` jumps to the next flagged segment you have not seen yet.

Usage (from cuasp_siamfc/)::

    .venv/Scripts/python tools/review_labels.py ../recordings/20261117_093557_B3_rapido_in_volo_r3

The labels are read from and saved to ``labels/ids/<recording>.csv`` (created by
``tools/prelabel_ids.py``). Saving is atomic; a backup of the file as it was at the start
of the session goes to ``runs/v4_labels/backups/``. Autosave every 2 minutes and on exit.

Keys (press h in the window for this list)::

    space        play / pause             + / -        faster / slower (0.25x .. 16x)
    right / .    next frame               left / ,     previous frame
    up / k       +1 s                     down / j     -1 s
    PgUp / PgDn  +10 s / -10 s            Home / End   first / last frame
    n / p        next / previous flagged segment not yet reviewed (N: any flagged)
    mouse drag   draw the box of this frame (manual)
    a            this frame: drone absent
    m            set / clear the mark (start of a range)
    c            confirm the range mark..here as correct (or this frame without a mark)
    x            range mark..here: drone absent
    i            range mark..here: interpolate between the two end boxes
    t / T        re-track forward / backward with SiamFC from this frame's box
    u            undo the last edit
    z            zoom 2x around the box      g      brightening on / off (display only)
    d            raw detections on / off     s      save
    q / Esc      save and quit

Frames played at 4x or slower, or stepped through, are marked reviewed.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import datetime as dt
from pathlib import Path
import shutil
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ids_labels as L  # noqa: E402

TOOL_VERSION = "2026-11-17-review-labels-v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO / "labels" / "ids"
DEFAULT_BACKUPS = REPO / "runs" / "v4_labels" / "backups"
DEFAULT_SIAMFC = REPO / "pretrained" / "networks" / "baseline-conv5_gray_e100.pth"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
AUTOSAVE_S = 120.0
STATUS_H = 46
TIMELINE_H = 26

# cv2.waitKeyEx codes on Windows
K_LEFT, K_UP, K_RIGHT, K_DOWN = 2424832, 2490368, 2555904, 2621440
K_PGUP, K_PGDN, K_END, K_HOME = 2162688, 2228224, 2293760, 2359296


def caps_lock_on() -> bool:
    """Caps Lock state on Windows (False elsewhere or when it cannot be read)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.user32.GetKeyState(0x14) & 1)  # VK_CAPITAL, toggle bit
    except Exception:  # noqa: BLE001 - a missing state only disables the correction
        return False


def key_char(k: int, caps_lock: bool) -> str:
    """Character of a waitKeyEx code, with Caps Lock undone: 'm' means m with or without it.

    With Caps Lock on, OpenCV delivers 'M' for m, so the lower-case commands were ignored
    and t / n turned into T / N (user report 2026-11-18). Shift + letter keeps its meaning.
    """
    ch = chr(k) if 0 < k < 256 else "\0"  # never "" : "" is `in` every string
    return ch.swapcase() if caps_lock and ch.isalpha() else ch

# BGR colours
GREEN, YELLOW, RED, MAGENTA, CYAN, GREY, WHITE, BLUE = (60, 220, 60), (0, 220, 255), (40, 40, 255), (255, 60, 255), (255, 255, 0), (110, 110, 110), (255, 255, 255), (255, 140, 40)


def box_colour(source: str, flag: int) -> tuple[int, int, int]:
    if flag:
        return RED
    if source in L.MANUAL_SOURCES:
        return MAGENTA
    if source in L.TRACKER_SOURCES:
        return YELLOW
    return GREEN


def timeline_codes(track: L.LabelTrack) -> np.ndarray:
    """Per-frame colour priority: 4 flagged and not reviewed, 3 manual, 2 tracker only, 1 absent, 0 detector."""
    code = np.zeros(track.n, np.int8)
    code[track.present == 0] = 1
    code[[s in L.TRACKER_SOURCES for s in track.source]] = 2
    code[[s in L.MANUAL_SOURCES for s in track.source]] = 3
    code[track.flag.astype(bool) & ~track.reviewed.astype(bool)] = 4
    return code


class Reviewer:
    def __init__(self, rec: Path, frames: list[Path], track: L.LabelTrack, meta: dict, labels_path: Path,
                 dets: Optional[list], display_h: int, review_max_speed: float, siamfc_weights: Path, fps: float) -> None:
        import cv2  # noqa: PLC0415

        self.cv2 = cv2
        self.rec, self.frames, self.track, self.meta, self.path = rec, frames, track, meta, labels_path
        self.dets = dets
        self.fps = fps
        self.review_max_speed = review_max_speed
        self.siamfc_weights = siamfc_weights
        first = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
        self.H, self.W = first.shape
        self.disp_h = int(display_h)
        self.disp_w = int(round(self.W * self.disp_h / self.H))
        self.f = 0
        self.playing = False
        self.speed_i = SPEEDS.index(1.0)
        self.mark: Optional[int] = None
        self.zoom = False
        self.enhance = True
        self.show_dets = False
        self.show_help = False
        self.drag: Optional[list[float]] = None  # x0, y0, x1, y1 in frame coords
        self.view = (0.0, 0.0, float(self.W), float(self.H))  # crop shown: x0, y0, w, h
        self.cache: OrderedDict = OrderedDict()
        self.message = ""
        self.message_t = 0.0
        self.tracker = None
        self.last_save = time.monotonic()
        self.quit = False
        self.win = f"review {rec.name}"

    # ------------------------------------------------------------------ frames

    def frame(self, f: int, full: bool) -> np.ndarray:
        key = (f, full)
        img = self.cache.get(key)
        if img is None:
            flags = self.cv2.IMREAD_GRAYSCALE if full else self.cv2.IMREAD_REDUCED_GRAYSCALE_2
            img = self.cv2.imread(str(self.frames[f]), flags)
            if img is None:
                raise RuntimeError(f"cannot read {self.frames[f]}")
            self.cache[key] = img
            if len(self.cache) > 12:
                self.cache.popitem(last=False)
        return img

    # ------------------------------------------------------------------ geometry

    def update_view(self) -> None:
        if not self.zoom:
            self.view = (0.0, 0.0, float(self.W), float(self.H))
            return
        cw, ch = self.W / 2.0, self.H / 2.0
        if self.track.present[self.f]:
            cx, cy = L.box_centre(self.track.box[self.f])
        else:
            x0, y0, w, h = self.view
            cx, cy = (x0 + w / 2.0, y0 + h / 2.0) if w < self.W else (self.W / 2.0, self.H / 2.0)
        x0 = min(max(cx - cw / 2.0, 0.0), self.W - cw)
        y0 = min(max(cy - ch / 2.0, 0.0), self.H - ch)
        self.view = (x0, y0, cw, ch)

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, w, h = self.view
        return (x0 + x * w / self.disp_w, y0 + y * h / self.disp_h)

    def to_disp(self, x: float, y: float) -> tuple[int, int]:
        x0, y0, w, h = self.view
        return (int(round((x - x0) * self.disp_w / w)), int(round((y - y0) * self.disp_h / h)))

    # ------------------------------------------------------------------ actions

    def say(self, text: str) -> None:
        self.message, self.message_t = text, time.monotonic()
        print(text, flush=True)

    def seek(self, f: int) -> None:
        self.f = int(min(max(f, 0), self.track.n - 1))

    def step(self, df: int) -> None:
        self.playing = False
        before = self.f
        self.seek(self.f + df)
        if abs(df) == 1:
            self.track.mark_reviewed(self.f, self.f)
        elif before != self.f:
            pass  # jumps do not mark frames reviewed

    def range_ends(self) -> tuple[int, int]:
        return (self.mark, self.f) if self.mark is not None else (self.f, self.f)

    def save(self, reason: str = "saved") -> None:
        meta = dict(self.meta)
        review = dict(meta.get("review") or {})
        review.update({"tool_version": TOOL_VERSION, "last_saved": dt.datetime.now().isoformat(timespec="seconds")})
        meta["review"] = review
        L.save_labels(self.track, self.path, meta)
        self.meta = meta
        self.last_save = time.monotonic()
        c = self.track.counts()
        self.say(f"{reason}: {self.path.name} (reviewed {c['reviewed']}/{c['frames']}, flagged not reviewed {c['flagged_not_reviewed']})")

    def ensure_tracker(self):
        if self.tracker is None:
            self.say("loading SiamFC ...")
            self.redraw()
            from suncubes.siamfc.net import SiamFCNet  # noqa: PLC0415
            from suncubes.siamfc.tracker import SiamFCTracker, default_device  # noqa: PLC0415

            self.tracker = SiamFCTracker(SiamFCNet.from_file(self.siamfc_weights), device=default_device())
        return self.tracker

    def retrack(self, forward: bool) -> None:
        f0 = self.f
        if not self.track.present[f0]:
            self.say("re-track: this frame has no box (draw one first)")
            return
        import torch  # noqa: PLC0415

        tracker = self.ensure_tracker()
        p = L.PrelabelParams()
        stop = self.track.n if forward else -1
        order = []
        f = f0 + (1 if forward else -1)
        while f != stop and len(order) < p.max_track_frames and self.track.source[f] not in L.MANUAL_SOURCES:
            order.append(f)
            f += 1 if forward else -1
        if not order:
            self.say("re-track: nothing to do (next frame already manual, or end of the recording)")
            return
        self.say(f"re-tracking {'forward' if forward else 'backward'} from frame {f0} ...")
        self.redraw()
        init = tuple(float(v) for v in self.track.box[f0])
        x, y, w, h = L.box_to_pixel_centre(init)

        def outputs():
            tracker.init(self.frame(f0, True), (x + w / 2.0, y + h / 2.0), (w, h))
            for g in order:
                res = tracker.update(self.frame(g, True))
                (cx, cy), (rw, rh) = res.center_xy, res.size_wh
                yield L.box_from_pixel_centre((cx - rw / 2.0, cy - rh / 2.0, rw, rh)), float(res.peak), float(res.apce)

        with torch.no_grad():
            steps = L.track_run(init, outputs(), (self.W, self.H), p)
        frames, boxes, scores, flags = [], [], [], []
        agree = 0
        for g, s in zip(order, steps):
            old_ok = self.track.present[g] and not self.track.flag[g] and self.track.source[g] in ("det", "det_weak", "blend")
            if s.box is not None and old_ok and L.iou(s.box, self.track.box[g]) >= 0.7:
                agree += 1
                if agree >= 10:
                    break  # back on a trusted pre-label: keep the rest as it is
            else:
                agree = 0
            if s.box is None:
                continue  # weak tracker frame: leave the old label, flag it below
            frames.append(g)
            boxes.append(s.box)
            scores.append(s.conf)
            flags.append(0)
        if frames:
            self.track.write_run(frames, boxes, "retrack", scores, flags)
            covered = set(frames)
            weak = [g for g, s in zip(order, steps) if s.box is None and g not in covered]
            for g in weak:
                self.track.flag[g] = 1
            self.say(f"re-tracked {len(frames)} frames ({min(frames)}..{max(frames)}); weak frames flagged: {len(weak)}; u undoes")
        else:
            self.say("re-track: the tracker lost the target at once")

    # ------------------------------------------------------------------ input

    def on_mouse(self, event, x, y, flags, param) -> None:
        cv2 = self.cv2
        if y >= self.disp_h + STATUS_H:  # timeline
            if event == cv2.EVENT_LBUTTONDOWN:
                self.playing = False
                self.seek(int(x * self.track.n / self.disp_w))
            return
        if y >= self.disp_h:
            return
        fx, fy = self.to_frame(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.playing = False
            self.drag = [fx, fy, fx, fy]
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.drag[2], self.drag[3] = fx, fy
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            x0, y0, x1, y1 = self.drag
            self.drag = None
            bx, by = min(x0, x1), min(y0, y1)
            bw, bh = abs(x1 - x0), abs(y1 - y0)
            if bw >= 4 and bh >= 4:
                self.track.set_box(self.f, (bx, by, bw, bh), "manual", None, 0)
                self.track.reviewed[self.f] = 1
                self.say(f"frame {self.f}: manual box {bw:.0f}x{bh:.0f} (t re-tracks forward from it)")

    def on_key(self, k: int) -> None:
        tr = self.track
        ch = key_char(k, caps_lock_on())
        if k in (27,) or ch == "q":
            self.quit = True
        elif ch == " ":
            self.playing = not self.playing
            self.play_t = time.monotonic()
            self.play_acc = 0.0
        elif ch in "+=":
            self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
        elif ch == "-":
            self.speed_i = max(self.speed_i - 1, 0)
        elif k == K_RIGHT or ch == ".":
            self.step(1)
        elif k == K_LEFT or ch == ",":
            self.step(-1)
        elif k == K_UP or ch == "k":
            self.step(int(round(self.fps)))
        elif k == K_DOWN or ch == "j":
            self.step(-int(round(self.fps)))
        elif k == K_PGUP:
            self.step(int(round(10 * self.fps)))
        elif k == K_PGDN:
            self.step(-int(round(10 * self.fps)))
        elif k == K_HOME:
            self.step(-tr.n)
        elif k == K_END:
            self.step(tr.n)
        elif ch in "nNp":
            nxt = tr.next_flagged(self.f, forward=ch != "p", unreviewed_only=ch != "N")
            if nxt is None:
                self.say("no more flagged segments in that direction" + (" (not yet reviewed)" if ch != "N" else ""))
            else:
                self.playing = False
                self.seek(max(nxt - int(round(self.fps / 2)), 0))  # start half a second before
                self.say(f"flagged segment at frame {nxt}")
        elif ch == "a":
            tr.set_absent(self.f, "manual_absent", 0)
            tr.reviewed[self.f] = 1
            self.say(f"frame {self.f}: absent")
        elif ch == "m":
            self.mark = None if self.mark == self.f else self.f
            self.say("mark cleared" if self.mark is None else f"mark at frame {self.f}")
        elif ch == "c" or k == 13:
            f0, f1 = self.range_ends()
            tr.confirm(f0, f1)
            self.mark = None
            self.say(f"confirmed frames {min(f0, f1)}..{max(f0, f1)}")
        elif ch == "x":
            if self.mark is None:
                self.say("x needs a mark (m) at the other end of the range; a marks a single frame")
            else:
                f0, f1 = self.range_ends()
                tr.set_range_absent(f0, f1)
                self.mark = None
                self.say(f"frames {min(f0, f1)}..{max(f0, f1)}: absent")
        elif ch == "i":
            if self.mark is None:
                self.say("i needs a mark (m) at the other end of the range")
            else:
                try:
                    n = tr.interpolate(*self.range_ends())
                    self.say(f"interpolated {n} frames")
                    self.mark = None
                except ValueError as exc:
                    self.say(f"interpolation: {exc}")
        elif ch in "tT":
            self.playing = False
            self.retrack(forward=ch == "t")
        elif ch == "u" or k == 26:
            r = tr.undo()
            self.say("nothing to undo" if r is None else f"undone (frames {r[0]}..{r[1]})")
        elif ch == "z":
            self.zoom = not self.zoom
        elif ch == "g":
            self.enhance = not self.enhance
        elif ch == "d":
            self.show_dets = not self.show_dets
            if self.show_dets and self.dets is None:
                self.say("no detection cache for this recording")
        elif ch == "s":
            self.save()
        elif ch == "h":
            self.show_help = not self.show_help

    # ------------------------------------------------------------------ drawing

    def redraw(self) -> None:
        self.cv2.imshow(self.win, self.render())
        self.cv2.waitKey(1)

    def render(self) -> np.ndarray:
        cv2, tr, f = self.cv2, self.track, self.f
        self.update_view()
        x0, y0, w, h = self.view
        full = self.zoom
        img = self.frame(f, full)
        k = 1.0 if full else img.shape[1] / self.W  # reduced decode: half size
        crop = img[int(y0 * k):int((y0 + h) * k), int(x0 * k):int((x0 + w) * k)]
        if self.enhance:
            crop = L.apply_gain(crop, L.auto_gain(crop, step=4))
        disp = cv2.cvtColor(cv2.resize(crop, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA if w / self.disp_w > 1 else cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
        if self.show_dets and self.dets is not None:
            for (bx, by, bw, bh), s in self.dets[f]:
                p0, p1 = self.to_disp(bx, by), self.to_disp(bx + bw, by + bh)
                cv2.rectangle(disp, p0, p1, BLUE, 1)
                cv2.putText(disp, f"{s:.2f}", (p0[0], p1[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE, 1)
        if tr.present[f]:
            bx, by, bw, bh = tr.box[f]
            col = box_colour(tr.source[f], tr.flag[f])
            p0, p1 = self.to_disp(bx, by), self.to_disp(bx + bw, by + bh)
            cv2.rectangle(disp, p0, p1, col, 2)
            label = tr.source[f] + ("" if not np.isfinite(tr.score[f]) else f" {tr.score[f]:.2f}")
            cv2.putText(disp, label, (p0[0], max(p0[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        else:
            col = RED if tr.flag[f] else GREY
            cv2.putText(disp, f"ABSENT ({tr.source[f]})", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        if self.drag is not None:
            dx0, dy0, dx1, dy1 = self.drag
            cv2.rectangle(disp, self.to_disp(min(dx0, dx1), min(dy0, dy1)), self.to_disp(max(dx0, dx1), max(dy0, dy1)), CYAN, 1)
        if self.show_help:
            y1 = min(46 + 18 * len(HELP_LINES) + 8, self.disp_h)
            x1 = min(640, self.disp_w)
            disp[40:y1, 0:x1] = (disp[40:y1, 0:x1] * 0.25).astype(np.uint8)  # dark panel behind the text
            for i, line in enumerate(HELP_LINES):
                cv2.putText(disp, line, (12, 60 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        status = np.zeros((STATUS_H, self.disp_w, 3), np.uint8)
        c = tr.counts()
        line1 = (f"{f}/{tr.n - 1}  {f / self.fps:6.2f} s  {'PLAY' if self.playing else 'pause'} x{SPEEDS[self.speed_i]:g}"
                 f"  {'flag ' if tr.flag[f] else ''}{'reviewed' if tr.reviewed[f] else 'not reviewed'}"
                 f"  mark {self.mark if self.mark is not None else '-'}  {'zoom ' if self.zoom else ''}{'' if self.enhance else 'raw '}"
                 f"{'*unsaved*' if tr.dirty else ''}")
        line2 = (f"reviewed {100 * c['reviewed'] / c['frames']:.0f}%  flagged not reviewed {c['flagged_not_reviewed']}"
                 f"  |  {self.message if time.monotonic() - self.message_t < 6 else 'h: help'}")
        cv2.putText(status, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.putText(status, line2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        timeline = np.zeros((TIMELINE_H, self.disp_w, 3), np.uint8)
        codes = timeline_codes(tr)
        cols = (np.arange(tr.n) * self.disp_w // tr.n).astype(int)
        col_code = np.zeros(self.disp_w, np.int8)
        np.maximum.at(col_code, cols, codes)
        rev = np.ones(self.disp_w, bool)
        np.logical_and.at(rev, cols, tr.reviewed.astype(bool))
        palette = np.array([GREEN, GREY, YELLOW, MAGENTA, RED], np.uint8)
        timeline[4:TIMELINE_H - 6] = palette[col_code][None, :, :]
        timeline[TIMELINE_H - 4:] = np.where(rev[:, None], np.array(WHITE, np.uint8), np.array((40, 40, 40), np.uint8))[None, :, :]
        cx = int(f * self.disp_w / tr.n)
        timeline[:, max(cx - 1, 0):cx + 2] = CYAN
        if self.mark is not None:
            mx = int(self.mark * self.disp_w / tr.n)
            timeline[:, max(mx - 1, 0):mx + 2] = WHITE
        return np.vstack([disp, status, timeline])

    # ------------------------------------------------------------------ loop

    def run(self) -> None:
        cv2 = self.cv2
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.win, self.disp_w, self.disp_h + STATUS_H + TIMELINE_H)
        cv2.setMouseCallback(self.win, self.on_mouse)
        self.play_t = time.monotonic()
        self.play_acc = 0.0
        try:
            while not self.quit:
                cv2.imshow(self.win, self.render())
                k = cv2.waitKeyEx(1 if self.playing else 20)
                if k != -1:
                    self.on_key(k)
                if self.playing:
                    now = time.monotonic()
                    self.play_acc += (now - self.play_t) * self.fps * SPEEDS[self.speed_i]
                    self.play_t = now
                    adv = int(self.play_acc)
                    if adv:
                        self.play_acc -= adv
                        before = self.f
                        self.seek(self.f + adv)
                        if SPEEDS[self.speed_i] <= self.review_max_speed:
                            self.track.mark_reviewed(before, self.f)
                        if self.f >= self.track.n - 1:
                            self.playing = False
                            self.say("end of the recording")
                if self.track.dirty and time.monotonic() - self.last_save > AUTOSAVE_S:
                    self.save("autosaved")
                if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if self.track.dirty:
                self.save("saved on exit")
            cv2.destroyAllWindows()


HELP_LINES = [line.strip() for line in __doc__.split("Keys (press h in the window for this list)::")[1].split("Frames played")[0].strip().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", type=Path, help="recording folder (or its frames/ folder)")
    ap.add_argument("--labels", type=Path, help=f"label file (default {DEFAULT_LABELS}/<recording>.csv)")
    ap.add_argument("--display-height", type=int, default=860, help="height of the image area in screen px")
    ap.add_argument("--review-max-speed", type=float, default=4.0, help="frames played at this speed or slower count as reviewed")
    ap.add_argument("--fps", type=float, default=62.6, help="recording frame rate (playback and time display)")
    ap.add_argument("--siamfc-weights", type=Path, default=DEFAULT_SIAMFC)
    ap.add_argument("--start", type=int, default=None, help="start at this frame (default: first flagged frame not reviewed)")
    args = ap.parse_args()

    rec = args.recording.parent if args.recording.name == "frames" else args.recording
    frames = sorted(p for p in (rec / "frames").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    path = args.labels or DEFAULT_LABELS / f"{rec.name}.csv"
    if not path.is_file():
        print(f"no label file {path}: run tools/prelabel_ids.py on the recording first")
        return 2
    track = L.load_labels(path)
    if track.n != len(frames):
        print(f"{path.name} has {track.n} frames, the recording {len(frames)}: refusing to review")
        return 2
    meta = L.load_meta(path)
    DEFAULT_BACKUPS.mkdir(parents=True, exist_ok=True)
    backup = DEFAULT_BACKUPS / f"{path.stem}.{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    shutil.copy2(path, backup)
    dets = None
    cache = Path(meta.get("detections_cache", "")) if meta.get("detections_cache") else None
    if cache is not None and cache.is_file():
        dets, _ = L.load_detections(cache, len(frames))
    r = Reviewer(rec, frames, track, meta, path, dets, args.display_height, args.review_max_speed, args.siamfc_weights, args.fps)
    if args.start is not None:
        r.seek(args.start)
    else:
        nxt = track.next_flagged(-1, unreviewed_only=True)
        r.seek(0 if nxt is None else max(nxt - int(round(args.fps / 2)), 0))
    c = track.counts()
    print(f"{rec.name}: {c['frames']} frames, flagged {c['flagged']}, reviewed {c['reviewed']}; backup {backup}")
    print("press h in the window for the keys")
    r.run()
    c = track.counts()
    print(f"done: reviewed {c['reviewed']}/{c['frames']}, flagged not reviewed {c['flagged_not_reviewed']}, sources {c['sources']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
