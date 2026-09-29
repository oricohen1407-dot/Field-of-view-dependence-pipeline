"""Gradio web interface for DeepSTORM3D PSF characterization."""
import io
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import tifffile
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from config.config import Config, UserConfig, AdvancedConfig, TrainingDataConfig, TrainingRunConfig
from config.emitter_centers import (
    PROJECT_DIR as DATA_ROOT_DIR, ZSTACK_FILES_PATH,
    ZSTACK_FILE, CENTRAL_BEAD_COORDINATES_PIXEL, OFFAXIS_ZSTACK_FILES, OFFAXIS_COORDS_PIXEL,
)
from func_utils import characterize_PSF
import app_utils

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_PATH = str(PROJECT_DIR / "config" / "config.json")
MICROSCOPES_PATH = str(PROJECT_DIR / "config" / "microscopes.json")
# Where Calibration Setup saves cropped emitters — NOT the Gradio upload's own temp copy path
# (that lands in the OS temp dir, e.g. AppData\Local\Temp\gradio\<hash>\..., which isn't a
# sensible permanent home for real output data).
CALIBRATION_EMITTERS_DIR = PROJECT_DIR / "calibration_setup_emitters"
# phase_retrieval() (app_utils.py) hardcodes this exact path for its per-bead exp/sim outputs —
# not configurable via pr_dict/param_dict, so this constant must track that literal default.
RESULTS_DIR = PROJECT_DIR / "phase_retrieval_outputs"
# Same Gradio-temp-path gotcha as CALIBRATION_EMITTERS_DIR above, hitting "Calibration folder"
# mode too: gr.File(file_count="directory") uploads each file into its OWN per-file hash
# subdirectory (AppData\Local\Temp\gradio\<hash>\<filename>), not one shared folder — so
# os.path.commonpath() across the uploaded paths collapses to their shared ancestor (just
# ...\Temp\gradio), one level above where any actual file lives. Resolving project_dir/filename
# against that then 404s. Fix: copy the uploaded files into one real, stable folder here instead
# of trying to reuse Gradio's own scattered temp layout as project_dir.
CALIBRATION_FOLDER_IMPORTS_DIR = PROJECT_DIR / "calibration_folder_imports"

MICROSCOPE_FIELDS = ["M", "NA", "n_immersion", "f_4f", "ps_camera", "ps_BFP", "n_sample", "bitdepth"]


def _make_emitters_out_dir(raw_stem: str) -> str:
    """A fresh, uniquely-timestamped Calibration Setup output folder — one per upload/Restart
    All, never reused — so separate picking sessions on the same raw file (or a restart within
    one) never pile near-duplicate crops (e.g. the same emitter re-picked a pixel or two off)
    into a shared folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(CALIBRATION_EMITTERS_DIR / f"{raw_stem}_{timestamp}_emitters")


def _import_calib_folder(tif_paths: list) -> str:
    """Copies each uploaded calibration TIFF (each sitting in its own Gradio temp-upload
    subdirectory) into one fresh, real, flat folder under CALIBRATION_FOLDER_IMPORTS_DIR, and
    returns that folder's path. This is the "Calibration folder" analog of
    _make_emitters_out_dir — one real place the rest of the app can treat as project_dir,
    instead of trying to reconstruct a shared folder out of Gradio's own scattered per-file
    temp layout."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = CALIBRATION_FOLDER_IMPORTS_DIR / timestamp
    os.makedirs(out_dir, exist_ok=True)
    for p in tif_paths:
        shutil.copy2(p, out_dir / os.path.basename(p))
    return str(out_dir)


CRITICAL_CSS = """
.critical-config {
    border: 2px solid #d9534f;
    border-radius: 10px;
    padding: 14px;
    background: rgba(217, 83, 79, 0.06);
}
.confirm-btn {
    background: rgba(92, 184, 92, 0.18) !important;
    border: 1px solid rgba(92, 184, 92, 0.55) !important;
    color: #3c763d !important;
}
.confirm-btn:hover {
    background: rgba(92, 184, 92, 0.30) !important;
}
.clear-btn {
    background: rgba(217, 83, 79, 0.12) !important;
    border: 1px solid rgba(217, 83, 79, 0.45) !important;
    color: #a94442 !important;
}
.clear-btn:hover {
    background: rgba(217, 83, 79, 0.22) !important;
}
.restart-btn {
    background: #1a1a1a !important;
    border: 1px solid #1a1a1a !important;
    color: #fff !important;
}
.restart-btn:hover {
    background: #3a3a3a !important;
}
"""

# Draws a square/rectangle "cursor" that tracks the mouse over an image and resizes live to
# match the currently-typed crop dimensions (in the image's own pixel space, not CSS pixels --
# scaled by the displayed <img>'s naturalWidth/naturalHeight vs. its on-screen size). Runs once
# at page load via demo.load(..., js=...); a real <script> tag inside gr.HTML would NOT execute
# (browsers don't run scripts inserted via innerHTML), so this is Gradio's actual supported
# mechanism for custom page-load JS. Uses elem_id + a generic `#id img` / `#id input` descendant
# selector rather than guessing at Gradio's internal component class names, and re-queries the
# DOM on every mousemove rather than caching the <img> node, so it keeps working even if Gradio
# replaces that node on a later image update (e.g. after marking a region).
SETUP_CROP_CURSOR_JS = """
() => {
    function readNumberInput(elemId) {
        const el = document.getElementById(elemId);
        const inp = el ? el.querySelector('input') : null;
        const v = inp ? parseFloat(inp.value) : NaN;
        return (isNaN(v) || v <= 0) ? 0 : v;
    }
    function readRadioColor(elemId, colorMap, fallback) {
        const el = document.getElementById(elemId);
        const checked = el ? el.querySelector('input[type=radio]:checked') : null;
        if (checked && colorMap[checked.value]) return colorMap[checked.value];
        return fallback;
    }
    function setupCropCursor(imageId, widthId, heightId, colorOrFn) {
        const overlay = document.createElement('div');
        overlay.style.position = 'fixed';
        overlay.style.pointerEvents = 'none';
        overlay.style.display = 'none';
        overlay.style.zIndex = '9999';
        overlay.style.boxSizing = 'border-box';
        overlay.style.borderWidth = '2px';
        overlay.style.borderStyle = 'solid';
        document.body.appendChild(overlay);

        document.addEventListener('mousemove', (e) => {
            const container = document.getElementById(imageId);
            const img = container ? container.querySelector('img') : null;
            if (!img) { overlay.style.display = 'none'; return; }
            const rect = img.getBoundingClientRect();
            const inside = e.clientX >= rect.left && e.clientX <= rect.right &&
                           e.clientY >= rect.top && e.clientY <= rect.bottom;
            if (!inside) { overlay.style.display = 'none'; return; }
            img.style.cursor = 'none';
            const wPx = readNumberInput(widthId);
            const hPx = readNumberInput(heightId);
            if (!wPx || !hPx || !img.naturalWidth || !img.naturalHeight) {
                overlay.style.display = 'none';
                return;
            }
            const scaleX = rect.width / img.naturalWidth;
            const scaleY = rect.height / img.naturalHeight;
            const wCss = wPx * scaleX;
            const hCss = hPx * scaleY;
            const color = typeof colorOrFn === 'function' ? colorOrFn() : colorOrFn;
            overlay.style.borderColor = color;
            overlay.style.background = color + '26';
            overlay.style.width = wCss + 'px';
            overlay.style.height = hCss + 'px';
            overlay.style.left = (e.clientX - wCss / 2) + 'px';
            overlay.style.top = (e.clientY - hCss / 2) + 'px';
            overlay.style.display = 'block';
        });
    }

    setupCropCursor('calib_raw_image', 'calib_crop_size', 'calib_crop_size', '#ff2828');
    setupCropCursor('td_frame_image', 'td_noise_w', 'td_noise_h', () => readRadioColor(
        'td_mark_mode',
        {'No-emitter region (baseline)': '#ff2828', 'Bright emitter (peak signal)': '#00dcff'},
        '#ff2828'
    ));
}
"""


def _default_config() -> Config:
    """Same experiment defaults main.py uses, for when no saved config.json exists yet.

    project_dir points directly at the folder holding the z-stack files (folded together with
    ZSTACK_FILES_PATH) rather than relying on zstack_folder, since the GUI no longer exposes
    that field — it always constructs UserConfig with zstack_folder defaulted to "".
    """
    return Config(
        user=UserConfig(
            project_dir=str(DATA_ROOT_DIR / ZSTACK_FILES_PATH),
            zstack_file=ZSTACK_FILE,
            central_bead_coordinates_pixel=CENTRAL_BEAD_COORDINATES_PIXEL,
            offaxis_zstack_files=OFFAXIS_ZSTACK_FILES,
            offaxis_coords_pixel=OFFAXIS_COORDS_PIXEL,
            external_mask=None,
        )
    )


def _load_microscopes() -> dict:
    """Read the named-microscope-preset library, tolerating a missing/corrupt file."""
    try:
        with open(MICROSCOPES_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_microscopes(microscopes: dict):
    with open(MICROSCOPES_PATH, "w") as f:
        json.dump(microscopes, f, indent=2)


class _StreamToQueue(io.TextIOBase):
    """Redirect stdout writes into a thread-safe queue for GUI streaming."""

    def __init__(self, q: queue.SimpleQueue):
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self):
        pass


def _build_live_figure(live_box: dict):
    """Build the training-progress panel from the latest app_utils._update_live_panel snapshot:
    mask-plane phase (bead-shifted) + effective BFP phase, stacked on the left, next to a rotating
    bead's multi-slice PSF grid (calculated vs. experimental) on top, loss/parameter history
    graphs on the bottom.

    Uses the matplotlib object-oriented API + an explicit Agg canvas (no pyplot global state),
    since this runs on the GUI polling thread while the training worker thread makes its own
    bare plt.* calls (phase_retrieval_with_displacement_iteration/iteration_<epoch>.jpg) — sharing
    pyplot's global figure stack across threads would race.
    """
    phase = live_box.get('phase')
    mask_phase = live_box.get('mask_phase')
    mask_shift_px = live_box.get('mask_shift_px', (0, 0))
    pred_slices = live_box.get('pred_slices')
    target_slices = live_box.get('target_slices')
    if phase is None or mask_phase is None or pred_slices is None or target_slices is None:
        return None

    slice_zi = live_box.get('slice_zi', [])
    slice_nfp = live_box.get('slice_nfp_um', [])
    bead_name = live_box.get('bead_name', '?')
    meta = live_box.get('meta', {})
    loss_hist = live_box.get('loss_history', [])
    d_hist = live_box.get('d_history', [])
    nfp_hist = live_box.get('nfp_offset_history', [])
    g_hist = live_box.get('g_sigma_history', [])

    n_slices = pred_slices.shape[0]
    fig = Figure(figsize=(2.5 + 2.0 * n_slices, 8), constrained_layout=True)
    FigureCanvasAgg(fig)
    subfig_top, subfig_bottom = fig.subfigures(2, 1, height_ratios=[2.2, 1])

    # ---- top: mask-plane phase (row 0) + effective BFP phase (row 1) on the left,
    top_gs = subfig_top.add_gridspec(2, 2 + n_slices, width_ratios=[1.3, 0.08] + [1] * n_slices)

    ax_mask_phase = subfig_top.add_subplot(top_gs[0, 0])
    im_mask_phase = ax_mask_phase.imshow(mask_phase, cmap="twilight")
    dx_px, dy_px = mask_shift_px
    ax_mask_phase.set_title(f"mask-plane phase\n(bead-shifted, Δ=({dx_px},{dy_px})px)", fontsize=9)
    ax_mask_phase.axis("off")
    subfig_top.colorbar(im_mask_phase, ax=ax_mask_phase, fraction=0.046, pad=0.04)

    ax_phase = subfig_top.add_subplot(top_gs[1, 0])
    im_phase = ax_phase.imshow(phase, cmap="twilight")
    ax_phase.set_title("effective BFP phase", fontsize=9)
    ax_phase.axis("off")
    subfig_top.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)

    ax_sep = subfig_top.add_subplot(top_gs[:, 1])
    ax_sep.set_xlim(0, 1)
    ax_sep.axvline(0.5, color="black", linewidth=3, alpha=0.8)
    ax_sep.axis("off")

    for col in range(n_slices):
        zi = slice_zi[col] if col < len(slice_zi) else col
        nfp = slice_nfp[col] if col < len(slice_nfp) else float('nan')

        ax_pred = subfig_top.add_subplot(top_gs[0, col + 2])
        ax_pred.imshow(pred_slices[col], cmap="gray")
        ax_pred.set_title(f"z{zi}\nNFP={nfp:.2f}um", fontsize=8)
        ax_pred.set_xticks([]); ax_pred.set_yticks([])
        if col == 0:
            ax_pred.set_ylabel("calculated", fontsize=9)

        ax_tgt = subfig_top.add_subplot(top_gs[1, col + 2])
        ax_tgt.imshow(target_slices[col], cmap="gray")
        ax_tgt.set_xticks([]); ax_tgt.set_yticks([])
        if col == 0:
            ax_tgt.set_ylabel("experimental", fontsize=9)

    subfig_top.suptitle(f"bead: {bead_name}", fontsize=10)

    # ---- bottom: loss + learned parameters over epochs ----
    ax_loss, ax_d, ax_nfp, ax_g = subfig_bottom.subplots(1, 4)
    x = list(range(len(loss_hist)))

    ax_loss.plot(x, loss_hist)
    ax_loss.set_yscale("log")
    ax_loss.set_title(f"loss={meta.get('loss', float('nan')):.4g}", fontsize=9)
    ax_loss.set_xlabel("epoch")

    ax_d.plot(x, d_hist, color="tab:orange")
    ax_d.set_title(f"d={meta.get('d', float('nan')):.1f}um", fontsize=9)
    ax_d.set_xlabel("epoch")

    ax_nfp.plot(x, nfp_hist, color="tab:green")
    ax_nfp.set_title(f"nfp_offset={meta.get('nfp_offset', float('nan')):.2f}um", fontsize=9)
    ax_nfp.set_xlabel("epoch")

    ax_g.plot(x, g_hist, color="tab:red")
    ax_g.set_title(f"g_sigma={meta.get('g_sigma', float('nan')):.3f}", fontsize=9)
    ax_g.set_xlabel("epoch")

    fig.suptitle(f"epoch {meta.get('step', '?')}")
    return fig


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_train_live_figure(live_box: dict):
    """Build the Train Model tab's live monitor from app_utils._make_post_epoch_fn's latest
    snapshot: train/val loss curves + LR on top, epoch/ETA/early-stopping status as text, and a
    fixed validation tile's predicted-vs-ground-truth max-projection (+ Jaccard/RMSE, if the
    best-effort Volume2XYZ decode succeeded) on the bottom. Same object-oriented Figure +
    FigureCanvasAgg convention as _build_live_figure (no bare pyplot — this runs on the GUI
    polling thread while the training worker thread runs concurrently)."""
    train_hist = live_box.get('train_loss_history')
    if train_hist is None:
        return None
    test_hist = live_box.get('test_loss_history', [])
    lr_hist = live_box.get('lr_history', [])
    epoch = live_box.get('epoch', 0)
    total_epochs = live_box.get('total_epochs', 0)
    best_metric = live_box.get('best_metric')
    epochs_without_improvement = live_box.get('epochs_without_improvement', 0)

    fig = Figure(figsize=(11, 7), constrained_layout=True)
    FigureCanvasAgg(fig)
    subfig_top, subfig_bottom = fig.subfigures(2, 1, height_ratios=[1, 1.3])

    ax_loss, ax_lr, ax_info = subfig_top.subplots(1, 3)
    x = list(range(1, len(train_hist) + 1))
    ax_loss.plot(x, train_hist, label="train")
    ax_loss.plot(x, test_hist, label="val")
    if best_metric is not None and test_hist:
        best_epoch = int(np.argmin(test_hist)) + 1
        ax_loss.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("epoch")
    cur_test = test_hist[-1] if test_hist else float('nan')
    ax_loss.set_title(f"best={best_metric:.4g} | current={cur_test:.4g}" if best_metric is not None
                       else f"current={cur_test:.4g}", fontsize=9)
    ax_loss.legend(fontsize=8)

    ax_lr.plot(x, lr_hist, color="tab:purple", drawstyle="steps-post")
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("epoch")
    ax_lr.set_title(f"lr={lr_hist[-1]:.2g}" if lr_hist else "lr", fontsize=9)

    ax_info.axis("off")
    early_stop = live_box.get('early_stopping_patience')
    info_lines = [
        f"epoch {epoch} / {total_epochs}",
        f"elapsed {_fmt_hms(live_box.get('elapsed_s', 0))}",
        f"ETA {_fmt_hms(live_box.get('eta_s', 0))}",
    ]
    if early_stop:
        info_lines.append(f"no improvement: {epochs_without_improvement} / {early_stop}")
    else:
        info_lines.append(f"no improvement: {epochs_without_improvement}")
    ax_info.text(0.0, 0.5, "\n".join(info_lines), fontsize=10, va="center", family="monospace")

    ax_pred, ax_tgt = subfig_bottom.subplots(1, 2)
    pred_proj = live_box.get('pred_proj')
    target_proj = live_box.get('target_proj')
    if pred_proj is not None and target_proj is not None:
        ax_pred.imshow(pred_proj, cmap="gray")
        ax_tgt.imshow(target_proj, cmap="gray")
    ax_pred.set_title(f"predicted (max-z proj, epoch {live_box.get('viz_epoch', '?')})", fontsize=9)
    ax_tgt.set_title("ground truth (max-z proj)", fontsize=9)
    ax_pred.set_xticks([]); ax_pred.set_yticks([])
    ax_tgt.set_xticks([]); ax_tgt.set_yticks([])

    jacc = live_box.get('sample_jaccard')
    if jacc is not None:
        rmse_xy = live_box.get('sample_rmse_xy')
        rmse_z = live_box.get('sample_rmse_z')
        parts = [f"jaccard={jacc:.2f}"]
        if rmse_xy is not None:
            parts.append(f"RMSE_xy={rmse_xy * 1000:.1f}nm")
        if rmse_z is not None:
            parts.append(f"RMSE_z={rmse_z * 1000:.1f}nm")
        subfig_bottom.suptitle(" | ".join(parts), fontsize=9)

    return fig


# ── Calibration Setup helpers (click-to-crop emitter picker) ──────────────────

_COORD_RE = re.compile(r"_x(\d+)_y(\d+)")


def _parse_coords_from_filename(name: str):
    """Extract (row, col) from a '..._x{col}_y{row}...' filename, or None if absent."""
    m = _COORD_RE.search(str(name))
    if not m:
        return None
    col, row = int(m.group(1)), int(m.group(2))
    return row, col


def _compass_tag(row_offset: int, col_offset: int) -> str:
    """8-way compass tag for a pixel offset from the on-axis bead. 'center' only for (0, 0)."""
    if row_offset == 0 and col_offset == 0:
        return "center"
    # -row_offset so "up" (smaller row index) reads as a positive angle, matching compass intuition
    angle = math.degrees(math.atan2(-row_offset, col_offset)) % 360
    sectors = [
        (22.5, "right"), (67.5, "topRight"), (112.5, "top"), (157.5, "topLeft"),
        (202.5, "left"), (247.5, "bottomLeft"), (292.5, "bottom"), (337.5, "bottomRight"),
        (360.0, "right"),
    ]
    for boundary, tag in sectors:
        if angle < boundary:
            return tag
    return "right"  # unreachable, angle < 360 always matches above


def _crop_window(row: int, col: int, size: int, H: int, W: int):
    """Bounding box (r0, r1, c0, c1) of a size×size window centered at (row, col), shifted to
    stay in-bounds (shrunk only if size exceeds the image dimension)."""
    size_h, size_w = min(size, H), min(size, W)
    r0 = max(0, min(row - size_h // 2, H - size_h))
    c0 = max(0, min(col - size_w // 2, W - size_w))
    return r0, r0 + size_h, c0, c0 + size_w


def _crop_window_rect(row: int, col: int, w: int, h: int, H: int, W: int):
    """Same idea as _crop_window but with independent width/height, for a plain rectangle mark
    (used by the Generate Training Data tab's noise-patch picker, which has no reason to be
    square)."""
    h, w = min(h, H), min(w, W)
    r0 = max(0, min(row - h // 2, H - h))
    c0 = max(0, min(col - w // 2, W - w))
    return r0, r0 + h, c0, c0 + w


def _read_tiff_with_retry(path: str, attempts: int = 8, base_delay: float = 0.25) -> np.ndarray:
    """tifffile.imread with retry-on-PermissionError. On Windows a just-uploaded large file can
    stay briefly locked (antivirus scanning it, or the upload's own write handle not released
    yet) — retrying with backoff rides that out instead of failing the very first attempt."""
    last_exc = None
    for i in range(attempts):
        try:
            return tifffile.imread(path)
        except PermissionError as exc:
            last_exc = exc
            time.sleep(base_delay * (i + 1))
    raise last_exc


def _normalize_slice(slice2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Scale a 2D slice to uint8 [0,255] using fixed vmin/vmax (stack-wide, not per-slice) so
    contrast stays consistent while scrubbing through Z."""
    rng = (vmax - vmin) or 1.0
    norm = np.clip((slice2d.astype(np.float32) - vmin) / rng, 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


_PENDING_BOX_COLOR = (255, 40, 40)     # red — current unconfirmed click
_CONFIRMED_BOX_COLOR = (255, 230, 0)   # yellow — already-saved OFF-axis emitter crops
_ONAXIS_BOX_COLOR = (60, 220, 90)      # green — the on-axis (central) emitter crop, distinct from off-axis
_TD_EMITTER_BOX_COLOR = (0, 220, 255)  # cyan — Generate Training Data tab's marked bright-emitter region
_BOX_THICKNESS = 2


def _draw_box(rgb: np.ndarray, bbox, color, thickness: int = _BOX_THICKNESS) -> None:
    """Draw a hollow rectangle border in-place on an RGB uint8 image — outline only, interior
    left untouched, so it reads as a transparent square mark rather than a filled overlay."""
    r0, r1, c0, c1 = bbox
    H, W = rgb.shape[:2]
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, H), min(c1, W)
    color_arr = np.array(color, dtype=np.uint8)
    rgb[r0:min(r0 + thickness, r1), c0:c1] = color_arr
    rgb[max(r1 - thickness, r0):r1, c0:c1] = color_arr
    rgb[r0:r1, c0:min(c0 + thickness, c1)] = color_arr
    rgb[r0:r1, max(c1 - thickness, c0):c1] = color_arr


def _render_raw_frame(stack_state: dict, z, emitters: list, pending: dict | None = None) -> np.ndarray:
    """Grayscale Z-slice rendered as RGB with hollow-square markers: green for the on-axis
    (central) emitter's saved crop, yellow for every other confirmed OFF-axis emitter crop
    (accumulates as more are picked), red for the current unconfirmed pending crop (if any) —
    same crop-sized square used as the click "crosshair"."""
    z = max(0, min(int(z), stack_state["Z"] - 1))
    gray = _normalize_slice(stack_state["array"][z], stack_state["vmin"], stack_state["vmax"])
    rgb = np.stack([gray, gray, gray], axis=-1).copy()
    for e in emitters:
        color = _ONAXIS_BOX_COLOR if e.get("is_onaxis") else _CONFIRMED_BOX_COLOR
        _draw_box(rgb, e["bbox"], color)
    if pending is not None:
        _draw_box(rgb, pending["bbox"], _PENDING_BOX_COLOR)
    return rgb


_STRIP_SEPARATOR_COLOR = (90, 90, 90)
_STRIP_SEPARATOR_WIDTH = 2


def _multi_slice_strip(crop: np.ndarray, vmin: float, vmax: float, max_slices: int = 7) -> np.ndarray:
    """Render up to max_slices evenly-spaced Z-slices of a crop side by side (thin gray
    separator between each), so the emitter's shape across depth is visible in the preview
    before confirming — not just whichever single slice the main Z-slider happens to be on."""
    Z = crop.shape[0]
    n = min(max_slices, Z)
    idxs = np.linspace(0, Z - 1, n).round().astype(int)
    panels = []
    for i, zi in enumerate(idxs):
        gray = _normalize_slice(crop[zi], vmin, vmax)
        panels.append(np.stack([gray, gray, gray], axis=-1))
        if i < n - 1:
            H = panels[-1].shape[0]
            sep = np.empty((H, _STRIP_SEPARATOR_WIDTH, 3), dtype=np.uint8)
            sep[:] = _STRIP_SEPARATOR_COLOR
            panels.append(sep)
    return np.concatenate(panels, axis=1)


# ── Per-emitter results viewer (reads phase_retrieval()'s existing saved outputs — no
# training-loop changes needed, this is purely reading files it already writes) ──────────────

_RESULT_STACK_RE = re.compile(r"^exp_stack_(\d+)_(.+)\.tif$")


def _discover_run_beads(results_dir: Path, expected_names: set | None = None) -> list:
    """List every bead with a completed exp+sim stack pair in results_dir, sorted by its
    training-time index (cnt=0 is always the on-axis bead, matching phase_retrieval()'s own
    stacks-list order). phase_retrieval() never clears results_dir between runs — it accumulates
    output from every run ever made — so without expected_names this would list stale beads
    from unrelated past runs (old data, old crops) right alongside the current run's. Pass the
    current run's actual bead names (see _expected_bead_names) to filter those out."""
    if not results_dir.is_dir():
        return []
    beads = []
    for exp_path in results_dir.glob("exp_stack_*.tif"):
        m = _RESULT_STACK_RE.match(exp_path.name)
        if not m:
            continue
        cnt, name = int(m.group(1)), m.group(2)
        if expected_names is not None and name not in expected_names:
            continue
        sim_path = results_dir / f"sim_stack_{cnt}_{name}.tif"
        if sim_path.is_file():
            beads.append({"cnt": cnt, "name": name, "exp_path": str(exp_path), "sim_path": str(sim_path)})
    beads.sort(key=lambda b: b["cnt"])
    return beads


def _expected_bead_names(cfg: Config) -> set:
    """Bead names phase_retrieval() actually produces for this run's config — the exact same
    os.path.splitext(basename)[0] transform it applies when building each bead's 'name'."""
    names = {Path(cfg.user.zstack_file).stem}
    names.update(Path(f).stem for f in cfg.user.offaxis_zstack_files)
    return names


def _render_bead_comparison(exp_path: str, sim_path: str, max_slices: int = 7) -> np.ndarray:
    """Two-row grid — calculated (top) vs experimental (bottom), matching the live debug
    panel's row order (_build_live_figure) — of up to max_slices evenly-spaced Z-slices from
    this bead's saved result stacks. Those stacks are already per-slice-max-normalized uint16
    (by phase_retrieval()'s own _to_u16), so this only needs a 16-to-8-bit rescale for display,
    not the vmin/vmax handling raw crops need."""
    exp = tifffile.imread(exp_path)
    sim = tifffile.imread(sim_path)
    Z = min(exp.shape[0], sim.shape[0])
    n = min(max_slices, Z)
    idxs = np.linspace(0, Z - 1, n).round().astype(int)

    def _row(stack: np.ndarray) -> np.ndarray:
        panels = []
        for i, zi in enumerate(idxs):
            gray = (stack[zi].astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
            panels.append(np.stack([gray, gray, gray], axis=-1))
            if i < n - 1:
                H = panels[-1].shape[0]
                sep = np.empty((H, _STRIP_SEPARATOR_WIDTH, 3), dtype=np.uint8)
                sep[:] = _STRIP_SEPARATOR_COLOR
                panels.append(sep)
        return np.concatenate(panels, axis=1)

    exp_row, sim_row = _row(exp), _row(sim)
    row_sep = np.empty((_STRIP_SEPARATOR_WIDTH, exp_row.shape[1], 3), dtype=np.uint8)
    row_sep[:] = _STRIP_SEPARATOR_COLOR
    return np.concatenate([sim_row, row_sep, exp_row], axis=0)


# ── Config ↔ field helpers ────────────────────────────────────────────────────

def _opt_float(v):
    s = str(v).strip() if v is not None else ""
    return None if s == "" else float(s)

def _opt_int(v):
    s = str(v).strip() if v is not None else ""
    return None if s == "" else int(float(s))

def _opt_str(v):
    s = str(v).strip() if v is not None else ""
    return None if s == "" else s


def config_to_fields(cfg: Config) -> list:
    """Flatten a Config into the ordered list of Gradio field values (78 items)."""
    u, a, t, tr = cfg.user, cfg.advanced, cfg.training, cfg.training_run
    sig_lo, sig_hi = (float(x) for x in t.signal_range.split(','))
    bg_lo, bg_hi = (float(x) for x in t.background_range.split(','))
    dens_lo, dens_hi = (float(x) for x in t.density_range.split(','))
    z_source = t.zrange_um if t.zrange_um.strip() else u.zrange
    z_lo, z_hi = (float(x) for x in z_source.split(','))
    noff_lo, noff_hi = (float(x) for x in t.noise_offset_range.split(','))
    return [
        # ── Microscope preset fields, part 1 (7 of 8 — bitdepth is with AdvancedConfig below) ──
        u.M, u.NA, u.n_immersion, u.lamda, u.n_sample,
        u.f_4f, u.ps_camera, u.ps_BFP,
        # ── UserConfig geometry / data (8) ─────────────────────────────────
        u.nfp_range_um, u.zrange,
        u.project_dir,
        u.zstack_file,
        json.dumps(u.central_bead_coordinates_pixel),
        "\n".join(u.offaxis_zstack_files),
        json.dumps(u.offaxis_coords_pixel),
        u.external_mask or "",
        # ── AdvancedConfig (30) ──────────────────────────────────────────────
        a.epochs, a.learning_rate, a.loss_label, a.r_bead,
        json.dumps(list(a.adam_betas)),
        a.lr_phase_mult, a.lr_sigma_mult, a.lr_d_mult,
        a.fine_defocus_range_um, a.fine_defocus_step_um, a.max_shift_px,
        a.g_sigma, a.g_size, a.circ_scale,
        a.d_min_um, a.d_max_um,
        "" if a.d_init_um is None else str(a.d_init_um),
        a.bitdepth,
        "" if a.baseline is None else str(a.baseline),
        "" if a.read_std is None else str(a.read_std),
        "" if a.bg is None else str(a.bg),
        a.non_uniform_noise_flag,
        a.mask_fit_save_dir or "",
        a.debug_bfp,
        a.debug_every_num_epoch,
        "" if a.debug_max_emitters is None else str(a.debug_max_emitters),
        # ── NFP center offset (learned) — appended, keeps every index above stable ───
        a.lr_nfp_mult,
        "" if a.nfp_offset_init_um is None else str(a.nfp_offset_init_um),
        a.nfp_offset_min_um,
        a.nfp_offset_max_um,
        a.mask_warmup_epochs,
        # ── Training-data generation (Generate Training Data tab) — appended, keeps every index above stable ──
        sig_lo, sig_hi,
        bg_lo, bg_hi,
        dens_lo, dens_hi,
        z_lo, z_hi,
        t.canvas_size_px,
        t.num_z_voxel, t.us_factor,
        t.blob_r, t.blob_sigma, t.blob_maxv,
        noff_lo, noff_hi,
        # ── Train Model — appended, keeps every index above stable ──────────────
        tr.training_data_dir, tr.checkpoint_dir, tr.device,
        tr.resume_checkpoint or "",
        tr.num_epochs,
        tr.batch_size, tr.learning_rate, tr.early_stopping_patience,
        tr.train_val_split, tr.shuffle_train_val_split, tr.num_workers,
        tr.numpy_seed, tr.torch_seed, tr.sample_viz_every_epochs, tr.viz_threshold,
    ]


def fields_to_config(
    # Microscope preset fields, part 1 (7 of 8)
    M, NA, n_immersion, lamda, n_sample,
    f_4f, ps_camera, ps_BFP,
    # UserConfig geometry / data (8)
    nfp_range_um, zrange,
    project_dir,
    zstack_file,
    central_bead_json, offaxis_files_text, offaxis_coords_json,
    external_mask,
    # AdvancedConfig (30)
    epochs, learning_rate, loss_label, r_bead,
    adam_betas_json,
    lr_phase_mult, lr_sigma_mult, lr_d_mult,
    fine_defocus_range_um, fine_defocus_step_um, max_shift_px,
    g_sigma, g_size, circ_scale,
    d_min_um, d_max_um, d_init_um,
    bitdepth,
    baseline, read_std, bg,
    non_uniform_noise_flag,
    mask_fit_save_dir,
    debug_bfp, debug_every_num_epoch, debug_max_emitters,
    lr_nfp_mult, nfp_offset_init_um, nfp_offset_min_um, nfp_offset_max_um,
    mask_warmup_epochs,
    # Training-data generation (16)
    td_sig_min, td_sig_max,
    td_bg_min, td_bg_max,
    td_density_min, td_density_max,
    td_zmin, td_zmax,
    td_canvas_size,
    td_num_z_voxel, td_us_factor,
    td_blob_r, td_blob_sigma, td_blob_maxv,
    td_noise_off_min, td_noise_off_max,
    # Train Model (15)
    train_data_dir, train_ckpt_dir, train_device, train_resume_ckpt, train_num_epochs,
    train_batch_size, train_lr, train_early_stopping, train_val_split, train_shuffle_split,
    train_num_workers, train_numpy_seed, train_torch_seed, train_sample_viz_every, train_viz_threshold,
) -> Config:
    """Parse ordered Gradio field values back into a Config object."""
    offaxis_files = [
        ln.strip()
        for ln in str(offaxis_files_text).strip().split("\n")
        if ln.strip()
    ]
    return Config(
        user=UserConfig(
            M=float(M), NA=float(NA), n_immersion=float(n_immersion),
            lamda=float(lamda), n_sample=float(n_sample),
            f_4f=float(f_4f), ps_camera=float(ps_camera), ps_BFP=float(ps_BFP),
            nfp_range_um=float(nfp_range_um), zrange=str(zrange),
            project_dir=str(project_dir).strip(),
            zstack_file=str(zstack_file).strip(),
            central_bead_coordinates_pixel=json.loads(str(central_bead_json)),
            offaxis_zstack_files=offaxis_files,
            offaxis_coords_pixel=json.loads(str(offaxis_coords_json)),
            external_mask=_opt_str(external_mask),
        ),
        advanced=AdvancedConfig(
            epochs=int(float(epochs)),
            learning_rate=float(learning_rate),
            loss_label=int(float(loss_label)),
            r_bead=float(r_bead),
            adam_betas=tuple(json.loads(str(adam_betas_json))),
            lr_phase_mult=float(lr_phase_mult),
            lr_sigma_mult=float(lr_sigma_mult),
            lr_d_mult=float(lr_d_mult),
            fine_defocus_range_um=float(fine_defocus_range_um),
            fine_defocus_step_um=float(fine_defocus_step_um),
            max_shift_px=int(float(max_shift_px)),
            g_sigma=float(g_sigma),
            g_size=int(float(g_size)),
            circ_scale=float(circ_scale),
            d_min_um=float(d_min_um),
            d_max_um=float(d_max_um),
            d_init_um=_opt_float(d_init_um),
            bitdepth=int(float(bitdepth)),
            baseline=_opt_float(baseline),
            read_std=_opt_float(read_std),
            bg=_opt_float(bg),
            non_uniform_noise_flag=bool(non_uniform_noise_flag),
            mask_fit_save_dir=_opt_str(mask_fit_save_dir),
            debug_bfp=bool(debug_bfp),
            debug_every_num_epoch=int(float(debug_every_num_epoch)),
            debug_max_emitters=_opt_int(debug_max_emitters),
            lr_nfp_mult=float(lr_nfp_mult),
            nfp_offset_init_um=_opt_float(nfp_offset_init_um),
            nfp_offset_min_um=float(nfp_offset_min_um),
            nfp_offset_max_um=float(nfp_offset_max_um),
            mask_warmup_epochs=int(float(mask_warmup_epochs)),
        ),
        training=TrainingDataConfig(
            signal_range=f"{float(td_sig_min)}, {float(td_sig_max)}",
            background_range=f"{float(td_bg_min)}, {float(td_bg_max)}",
            density_range=f"{int(float(td_density_min))}, {int(float(td_density_max))}",
            zrange_um=f"{float(td_zmin)}, {float(td_zmax)}",
            canvas_size_px=int(float(td_canvas_size)),
            num_z_voxel=int(float(td_num_z_voxel)),
            us_factor=int(float(td_us_factor)),
            blob_r=int(float(td_blob_r)),
            blob_sigma=float(td_blob_sigma),
            blob_maxv=int(float(td_blob_maxv)),
            noise_offset_range=f"{float(td_noise_off_min)}, {float(td_noise_off_max)}",
        ),
        training_run=TrainingRunConfig(
            training_data_dir=str(train_data_dir).strip(),
            checkpoint_dir=str(train_ckpt_dir).strip(),
            device=str(train_device),
            resume_checkpoint=_opt_str(train_resume_ckpt),
            num_epochs=int(float(train_num_epochs)),
            batch_size=int(float(train_batch_size)),
            learning_rate=float(train_lr),
            early_stopping_patience=int(float(train_early_stopping)),
            train_val_split=float(train_val_split),
            shuffle_train_val_split=bool(train_shuffle_split),
            num_workers=int(float(train_num_workers)),
            numpy_seed=int(float(train_numpy_seed)),
            torch_seed=int(float(train_torch_seed)),
            sample_viz_every_epochs=int(float(train_sample_viz_every)),
            viz_threshold=float(train_viz_threshold),
        ),
    )


def _load_pr_results_and_status():
    """Loads Phase Retrieval's saved results (RESULTS_DIR/results.json + phase_mask.npy) if
    present -- shared by GUI startup (auto-load the latest finished calibration into the
    Generate Training Data tab) and the manual "Load Phase Retrieval Results" button (refresh
    after running a new calibration in the same session)."""
    # results.json/phase_mask.npy are written non-atomically (plain json.dump/np.save, no
    # temp-file+rename) -- a load landing mid-write could hit a truncated/locked file.
    try:
        results = app_utils.load_phase_retrieval_results(str(RESULTS_DIR))
    except Exception as exc:
        return None, f"[ERROR] Could not load Phase Retrieval results: {exc}"
    if results is None:
        return None, "No Phase Retrieval results found yet — run Phase Retrieval first (Configure + Run tabs)."
    status = (f"Loaded: d={results['d_um']:.1f} um, g_sigma={results['g_sigma']:.3f}, "
              f"nfp_offset={results['nfp_offset_um']:.3f} um, nfp_range={results['nfp_range_um']:.2f} um.")
    return results, status


# ── UI ────────────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    defaults = config_to_fields(_default_config())
    if Path(DEFAULT_SAVE_PATH).exists():
        try:
            defaults = config_to_fields(Config.load(DEFAULT_SAVE_PATH))
        except Exception:
            pass
    # Auto-load whatever Phase Retrieval results already exist on disk (RESULTS_DIR), so the
    # Generate Training Data tab starts pre-loaded with the latest finished calibration instead
    # of requiring a manual "Load Phase Retrieval Results" click every time the app is opened.
    initial_pr_results, initial_pr_status = _load_pr_results_and_status()

    with gr.Blocks(title="DeepSTORM3D — FOV-dependance") as demo:
        gr.HTML(f"<style>{CRITICAL_CSS}</style>")
        gr.Markdown("# DeepSTORM3D — FOV-dependance PSF Characterization")

        with gr.Tabs():
            with gr.Tab("Configure"):
                # ── Load / Save ──────────────────────────────────────────────────────
                with gr.Row(equal_height=True):
                    load_file = gr.File(
                        label="Load Config from JSON",
                        file_types=[".json"],
                        type="filepath",
                    )
                    with gr.Column():
                        save_btn = gr.Button("Save Config to Disk")
                        save_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Save status appears here",
                        )

                # ── Critical config ──────────────────────────────────────────────────
                with gr.Group(elem_classes=["critical-config"]):
                    gr.Markdown("## ⚠️ Critical — configure before running")
                    gr.Markdown("These vary per experiment — double-check before every run.")

                    calib_mode = gr.Radio(
                        ["Calibration folder", "Calibration setup"],
                        value="Calibration folder", label="Calibration data source",
                    )

                    with gr.Group(visible=True) as calib_folder_group:
                        gr.Markdown(
                            "**Calibration folder** — browse to auto-fill the files below, "
                            "pick the on-axis file, then add coordinates."
                        )
                        with gr.Row(equal_height=True):
                            folder_upload = gr.File(
                                label="Browse for calibration data folder",
                                file_count="directory",
                            )
                            with gr.Column():
                                onaxis_picker = gr.Dropdown(
                                    label="Which file is the on-axis (central) bead?", choices=[],
                                )
                                move_onaxis_btn = gr.Button("Move to Central Bead field")
                        scan_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Folder scan status appears here",
                        )

                    with gr.Group(visible=False) as calib_setup_group:
                        gr.Markdown(
                            "**Calibration setup** — browse a raw .tif, click an emitter to crop "
                            "it, Confirm or Clear. First pick is on-axis. Set min/max below to "
                            "trim the Z-range used for every emitter's crop."
                        )
                        raw_stack_state = gr.State(None)
                        pending_crop_state = gr.State(None)
                        emitters_state = gr.State([])
                        with gr.Row(equal_height=True):
                            raw_file_upload = gr.File(
                                label="Browse for raw calibration .tif", file_count="single",
                            )
                            crop_size_input = gr.Number(
                                label="Crop size (px)", value=70, precision=0, elem_id="calib_crop_size",
                            )
                        z_slider = gr.Slider(label="Z-slice (browse)", minimum=0, maximum=1, step=1, value=0)
                        with gr.Row(equal_height=True):
                            set_min_btn = gr.Button("Set min z-slice")
                            z_min_display = gr.Number(label="Min z-slice", value=0, precision=0, interactive=False)
                            set_max_btn = gr.Button("Set max z-slice")
                            z_max_display = gr.Number(label="Max z-slice", value=1, precision=0, interactive=False)
                        z_range_display = gr.Markdown("")
                        raw_image = gr.Image(
                            label="Raw calibration data — click an emitter", interactive=False,
                            elem_id="calib_raw_image",
                        )
                        setup_instruction = gr.Markdown("Browse a raw .tif file to begin.")
                        cropped_image = gr.Image(
                            label="Cropped preview (Z-slices left→right) — Confirm or Clear",
                            interactive=False,
                        )
                        with gr.Row(equal_height=True):
                            confirm_btn = gr.Button("Confirm", elem_classes=["confirm-btn"])
                            clear_btn = gr.Button("Clear / Retry", elem_classes=["clear-btn"])
                            restart_btn = gr.Button("Restart All", elem_classes=["restart-btn"])
                        setup_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Emitter setup status appears here",
                        )

                    with gr.Row(equal_height=True):
                        u_nfp_range = gr.Number(label="NFP z-range (µm)", value=defaults[8])
                        u_lamda    = gr.Number(label="λ emission (µm)",   value=defaults[3])
                    with gr.Row(equal_height=True):
                        a_d_min    = gr.Number(label="d_min (µm)", value=defaults[30])
                        a_d_max    = gr.Number(label="d_max (µm)", value=defaults[31])
                    u_zstack      = gr.Textbox(label="Central bead file (filename only)", value=defaults[11])
                    u_central     = gr.Textbox(label="Central bead coords [row, col] (JSON)", value=defaults[12])
                    u_offax_files = gr.Textbox(
                        label="Off-axis files (one per line)", value=defaults[13], lines=5,
                    )
                    u_offax_coord = gr.Textbox(
                        label="Off-axis coords [[row, col], ...] (JSON)", value=defaults[14], lines=3,
                    )

                # ── Microscope setup (named presets) ────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Microscope Setup")
                    gr.Markdown("Fixed for a given physical setup — save/load as a named preset.")
                    microscopes = _load_microscopes()
                    with gr.Row(equal_height=True):
                        m_dropdown = gr.Dropdown(
                            label="Microscope preset",
                            choices=list(microscopes.keys()),
                            value="Default" if "Default" in microscopes else None,
                        )
                    with gr.Row(equal_height=True):
                        m_M        = gr.Number(label="Magnification (M)",      value=defaults[0])
                        m_NA       = gr.Number(label="NA",                      value=defaults[1])
                        m_n_imm    = gr.Number(label="n_immersion",             value=defaults[2])
                        m_n_sample = gr.Number(label="n_sample",                value=defaults[4])
                    with gr.Row(equal_height=True):
                        m_f4f      = gr.Number(label="f_4f (µm)",               value=defaults[5])
                        m_ps_cam   = gr.Number(label="Camera pixel size (µm)",  value=defaults[6])
                        m_ps_BFP   = gr.Number(label="BFP pixel size (µm)",     value=defaults[7])
                        m_bitdepth = gr.Number(label="Bit depth",               value=defaults[33], precision=0)
                    with gr.Row(equal_height=True):
                        m_name     = gr.Textbox(label="Save current values as new microscope named:")
                        m_save_btn = gr.Button("Save as Microscope")
                    m_status = gr.Textbox(show_label=False, interactive=False, placeholder="Microscope save status appears here")

                # ── Other settings ───────────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Other settings")
                    u_zrange   = gr.Textbox(label='zrange ("min, max" µm, display only)', value=defaults[9])
                    u_calib_root_dir = gr.Textbox(
                        label="Calibration root dir",
                        value=defaults[10],
                    )
                    u_ext_mask = gr.Textbox(
                        label="Starting-guess mask for phase retrieval (.npy/.mat path, optional)",
                        value=defaults[15],
                    )

                # ── Advanced Config ──────────────────────────────────────────────────
                with gr.Accordion("Advanced Config", open=False):
                    gr.Markdown("**Phase retrieval optimisation**")
                    with gr.Row(equal_height=True):
                        a_epochs   = gr.Number(label="Epochs",               value=defaults[16], precision=0)
                        a_lr       = gr.Number(label="Learning rate",         value=defaults[17])
                        a_loss     = gr.Number(label="Loss (1=Gauss, 2=L2)", value=defaults[18], precision=0)
                        a_r_bead   = gr.Number(label="Bead radius (µm)",      value=defaults[19])
                        a_mask_warmup = gr.Number(label="Mask warmup epochs (on-axis only, d/NFP frozen)", value=defaults[46], precision=0)
                    with gr.Row(equal_height=True):
                        a_betas    = gr.Textbox(label="Adam betas [β1, β2] (JSON)", value=defaults[20])
                        a_lr_phase = gr.Number(label="lr_phase_mult",         value=defaults[21])
                        a_lr_sigma = gr.Number(label="lr_sigma_mult",         value=defaults[22])
                        a_lr_d     = gr.Number(label="lr_d_mult",             value=defaults[23])
                        a_lr_nfp   = gr.Number(label="lr_nfp_mult",           value=defaults[42])

                    gr.Markdown("**Per-bead fine alignment**")
                    with gr.Row(equal_height=True):
                        a_fd_range = gr.Number(label="Defocus range (µm)",   value=defaults[24])
                        a_fd_step  = gr.Number(label="Defocus step (µm)",     value=defaults[25])
                        a_max_sh   = gr.Number(label="Max shift (px)",         value=defaults[26], precision=0)

                    gr.Markdown("**Forward model**")
                    with gr.Row(equal_height=True):
                        a_g_sigma  = gr.Number(label="g_sigma (µm)",          value=defaults[27])
                        a_g_size   = gr.Number(label="g_size (px)",            value=defaults[28], precision=0)
                        a_circ     = gr.Number(label="circ_scale",             value=defaults[29])
                    a_d_init   = gr.Textbox(label="d_init (µm, empty=midpoint of [d_min, d_max] above)", value=defaults[32])
                    gr.Markdown("*Only the NFP window's CENTER OFFSET is learned (the range length above "
                                "is fixed). These bounds/init are sanity limits, not critical:*")
                    with gr.Row(equal_height=True):
                        a_nfp_offset_init = gr.Textbox(label="nfp_offset init (µm, empty=midpoint of bounds)", value=defaults[43])
                        a_nfp_offset_min  = gr.Number(label="nfp_offset min (µm)", value=defaults[44])
                        a_nfp_offset_max  = gr.Number(label="nfp_offset max (µm)", value=defaults[45])

                    gr.Markdown("**Camera / noise**")
                    with gr.Row(equal_height=True):
                        a_baseline = gr.Textbox(label="Baseline (empty=None)", value=defaults[34])
                        a_read_std = gr.Textbox(label="Read std (empty=None)", value=defaults[35])
                        a_bg       = gr.Textbox(label="BG (empty=None)",       value=defaults[36])
                    a_noisy        = gr.Checkbox(label="Non-uniform noise",    value=defaults[37])

                    gr.Markdown("**Runtime / debug**")
                    a_save_dir = gr.Textbox(label="mask_fit_save_dir (empty=auto)",  value=defaults[38])
                    with gr.Row(equal_height=True):
                        a_dbg_ev   = gr.Number(label="Debug every N epochs",            value=defaults[40], precision=0)
                        a_dbg_max  = gr.Textbox(label="debug_max_emitters (empty=auto)", value=defaults[41])

            with gr.Tab("Run"):
                a_dbg_bfp = gr.Checkbox(label="Save debug images to disk", value=defaults[39])
                with gr.Row(equal_height=True):
                    run_btn = gr.Button("Run Characterize PSF", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop", interactive=False)
                gr.Markdown("**Latest debug snapshot (live)**")
                live_plot = gr.Plot(show_label=False)
                log_out = gr.Textbox(label="Output Log", lines=20, interactive=False)

                gr.Markdown(
                    "### Per-emitter results (populated once the run finishes)\n"
                    "Click an emitter on the raw image (if you used Calibration Setup), or "
                    "just pick one from the dropdown."
                )
                results_beads_state = gr.State([])
                with gr.Row(equal_height=True):
                    results_image = gr.Image(
                        label="Raw calibration data — click an emitter", interactive=False, visible=False,
                    )
                    with gr.Column():
                        results_dropdown = gr.Dropdown(label="Pick an emitter", choices=[])
                        results_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Run status / selected emitter appears here",
                        )
                results_plot = gr.Image(
                    label="Calculated (top row) vs Experimental (bottom row) — Z-slices left→right",
                    interactive=False,
                )

            with gr.Tab("Generate Training Data"):
                gr.Markdown(
                    "### 1. Load fitted PSF parameters from Phase Retrieval\n"
                    "Run Phase Retrieval first (Configure + Run tabs), then load its results here."
                )
                with gr.Row(equal_height=True):
                    td_load_pr_btn = gr.Button("Load Phase Retrieval Results")
                    td_pr_status = gr.Textbox(
                        show_label=False, interactive=False, value=initial_pr_status,
                        placeholder="Click to load the latest fitted d / g_sigma / NFP offset / phase mask.",
                    )
                td_pr_results_state = gr.State(initial_pr_results)

                gr.Markdown(
                    "### 2. Upload an experimental frame and mark two reference regions\n"
                    "Mark a **no-emitter** patch (baseline/noise) and a **bright emitter** patch "
                    "(peak signal) — together they calibrate Background, Noise offset, and Signal "
                    "against your real data, mirroring how the root pipeline's SNR-characterization "
                    "step works."
                )
                with gr.Row(equal_height=True):
                    td_frame_upload = gr.File(label="Experimental frame (.tif)", file_count="single")
                    with gr.Column():
                        td_mark_mode = gr.Radio(
                            ["No-emitter region (baseline)", "Bright emitter (peak signal)"],
                            value="No-emitter region (baseline)", label="Click marks",
                            elem_id="td_mark_mode",
                        )
                        td_noise_w = gr.Number(
                            label="Marked-patch width (px)", value=40, precision=0, elem_id="td_noise_w",
                        )
                        td_noise_h = gr.Number(
                            label="Marked-patch height (px)", value=40, precision=0, elem_id="td_noise_h",
                        )
                td_z_slider = gr.Slider(label="Z-slice (browse)", minimum=0, maximum=1, step=1, value=0)
                td_frame_state = gr.State(None)
                td_noise_bbox_state = gr.State(None)
                td_emitter_bbox_state = gr.State(None)
                td_frame_image = gr.Image(
                    label="Click to mark the selected region type", interactive=False,
                    elem_id="td_frame_image",
                )
                td_noise_status = gr.Textbox(
                    show_label=False, interactive=False,
                    placeholder="Upload a frame, then mark both a no-emitter and a bright-emitter region.",
                )

                gr.Markdown("### 3. Parameters — adjust, then Update Preview")
                with gr.Row(equal_height=True):
                    td_sig_min = gr.Number(label="Signal min (photons)", value=defaults[47])
                    td_sig_max = gr.Number(label="Signal max (photons)", value=defaults[48])
                with gr.Row(equal_height=True):
                    td_bg_min = gr.Number(label="Background noise variance min (counts²)", value=defaults[49])
                    td_bg_max = gr.Number(label="Background noise variance max (counts²)", value=defaults[50])
                with gr.Row(equal_height=True):
                    td_density_min = gr.Number(label="Emitters/frame min", value=defaults[51], precision=0)
                    td_density_max = gr.Number(label="Emitters/frame max", value=defaults[52], precision=0)
                with gr.Row(equal_height=True):
                    td_zmin = gr.Number(label="Z min (µm)", value=defaults[53])
                    td_zmax = gr.Number(label="Z max (µm)", value=defaults[54])
                td_canvas_size = gr.Number(label="Training-frame canvas size (px)", value=defaults[55], precision=0)

                with gr.Accordion("Advanced", open=False):
                    with gr.Row(equal_height=True):
                        td_num_z_voxel = gr.Number(label="Z voxels (D)", value=defaults[56], precision=0)
                        td_us_factor = gr.Number(label="Up-sampling factor", value=defaults[57], precision=0)
                    with gr.Row(equal_height=True):
                        td_blob_r = gr.Number(label="Blob radius (voxels)", value=defaults[58], precision=0)
                        td_blob_sigma = gr.Number(label="Blob sigma", value=defaults[59])
                        td_blob_maxv = gr.Number(label="Blob max value", value=defaults[60])
                    with gr.Row(equal_height=True):
                        td_noise_off_min = gr.Number(label="Baseline / noise offset min (counts)", value=defaults[61])
                        td_noise_off_max = gr.Number(label="Baseline / noise offset max (counts)", value=defaults[62])

                td_update_preview_btn = gr.Button("Update Preview")
                td_preview_plot = gr.Plot(show_label=False)
                td_preview_status = gr.Textbox(
                    show_label=False, interactive=False, placeholder="Preview status appears here.",
                )

                gr.Markdown("### 4. Simulate Training Data")
                with gr.Row(equal_height=True):
                    td_out_dir = gr.Textbox(label="Output folder", value=str(PROJECT_DIR / "training_data"))
                    td_n_ims = gr.Number(label="Number of frames", value=10000, precision=0)
                with gr.Row(equal_height=True):
                    td_simulate_btn = gr.Button("Simulate Training Data", variant="primary")
                    td_stop_btn = gr.Button("Stop", variant="stop", interactive=False)
                td_log_out = gr.Textbox(label="Output Log", lines=15, interactive=False)

            with gr.Tab("Train Model"):
                gr.Markdown("### 1. Load Training Data")
                with gr.Row(equal_height=True):
                    train_load_td_btn = gr.Button("Load Training Data")
                    train_td_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Click to load x/, y.pickle, param.pickle from the training-data folder below.",
                    )
                train_td_meta_state = gr.State(None)

                gr.Markdown("### 2. Configure")
                with gr.Group(elem_classes=["critical-config"]):
                    with gr.Row(equal_height=True):
                        train_data_dir = gr.Textbox(
                            label="Training data folder",
                            value=defaults[63] or str(PROJECT_DIR / "training_data"),
                        )
                        train_ckpt_dir = gr.Textbox(
                            label="Checkpoint output folder",
                            value=defaults[64] or str(PROJECT_DIR / "training_results"),
                        )
                    with gr.Row(equal_height=True):
                        train_device = gr.Dropdown(
                            label="Device", choices=["auto", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
                            value=defaults[65],
                        )
                        train_resume_ckpt = gr.Textbox(
                            label="Resume from checkpoint (filename, empty = fresh run)", value=defaults[66],
                        )
                        train_num_epochs = gr.Number(label="Number of epochs", value=defaults[67], precision=0)

                with gr.Accordion("Advanced", open=False):
                    with gr.Row(equal_height=True):
                        train_batch_size = gr.Number(label="Batch size", value=defaults[68], precision=0)
                        train_lr = gr.Number(label="Learning rate", value=defaults[69])
                        train_early_stopping = gr.Number(label="Early stopping patience (epochs)", value=defaults[70], precision=0)
                    with gr.Row(equal_height=True):
                        train_val_split = gr.Number(label="Train/val split (train fraction)", value=defaults[71])
                        train_shuffle_split = gr.Checkbox(label="Shuffle before train/val split", value=defaults[72])
                        train_num_workers = gr.Number(label="DataLoader num_workers", value=defaults[73], precision=0)
                    with gr.Row(equal_height=True):
                        train_numpy_seed = gr.Number(label="NumPy seed", value=defaults[74], precision=0)
                        train_torch_seed = gr.Number(label="Torch seed", value=defaults[75], precision=0)
                    with gr.Row(equal_height=True):
                        train_sample_viz_every = gr.Number(label="Sample-viz cadence (epochs)", value=defaults[76], precision=0)
                        train_viz_threshold = gr.Number(label="Viz decode threshold", value=defaults[77])

                with gr.Row(equal_height=True):
                    train_start_btn = gr.Button("Train Model", variant="primary")
                    train_stop_btn = gr.Button("Stop", variant="stop", interactive=False)

                gr.Markdown("**Live training monitor**")
                train_live_plot = gr.Plot(show_label=False)
                train_summary = gr.Textbox(
                    show_label=False, interactive=False,
                    placeholder="Best/last checkpoint paths appear here once training finishes.",
                )
                train_log_out = gr.Textbox(label="Output Log", lines=15, interactive=False)

        # component list — order MUST match config_to_fields / fields_to_config
        all_fields = [
            m_M, m_NA, m_n_imm, u_lamda, m_n_sample,
            m_f4f, m_ps_cam, m_ps_BFP,
            u_nfp_range, u_zrange,
            u_calib_root_dir,
            u_zstack, u_central, u_offax_files, u_offax_coord, u_ext_mask,
            a_epochs, a_lr, a_loss, a_r_bead,
            a_betas, a_lr_phase, a_lr_sigma, a_lr_d,
            a_fd_range, a_fd_step, a_max_sh,
            a_g_sigma, a_g_size, a_circ,
            a_d_min, a_d_max, a_d_init,
            m_bitdepth, a_baseline, a_read_std, a_bg,
            a_noisy, a_save_dir,
            a_dbg_bfp, a_dbg_ev, a_dbg_max,
            a_lr_nfp, a_nfp_offset_init, a_nfp_offset_min, a_nfp_offset_max,
            a_mask_warmup,
            td_sig_min, td_sig_max,
            td_bg_min, td_bg_max,
            td_density_min, td_density_max,
            td_zmin, td_zmax,
            td_canvas_size,
            td_num_z_voxel, td_us_factor,
            td_blob_r, td_blob_sigma, td_blob_maxv,
            td_noise_off_min, td_noise_off_max,
            train_data_dir, train_ckpt_dir, train_device, train_resume_ckpt, train_num_epochs,
            train_batch_size, train_lr, train_early_stopping, train_val_split, train_shuffle_split,
            train_num_workers, train_numpy_seed, train_torch_seed, train_sample_viz_every, train_viz_threshold,
        ]

        # microscope preset fields, in the fixed order used by microscopes.json entries
        microscope_fields = [m_M, m_NA, m_n_imm, m_f4f, m_ps_cam, m_ps_BFP, m_n_sample, m_bitdepth]

        # runtime-only state shared between run_handler and stop_handler — not part of Config,
        # never persisted. "busy" is an explicit one-run-at-a-time guard, kept even though
        # demo.queue()'s default concurrency_limit=1 already serializes Run clicks process-wide.
        _run_state = {"stop_event": None, "busy": False}
        # separate from _run_state above so the Run tab's PSF-characterization run and this
        # tab's training-data generation run never share a stop button / busy flag.
        _td_run_state = {"stop_event": None, "busy": False}
        # separate again for Train Model — all three worker threads redirect the process-global
        # sys.stdout, so all three busy flags must be cross-checked by every one of the three
        # run-starting handlers (see the guards in run_handler, on_td_simulate, on_train_start).
        _train_run_state = {"stop_event": None, "busy": False}

        # ── Handlers ─────────────────────────────────────────────────────────

        def load_handler(filepath):
            if not filepath:
                return [gr.update()] * len(all_fields)
            cfg = Config.load(filepath)
            return config_to_fields(cfg)

        def save_handler(*vals):
            try:
                cfg = fields_to_config(*vals)
                cfg.save(DEFAULT_SAVE_PATH)
                return f"Saved to {DEFAULT_SAVE_PATH}"
            except Exception as exc:
                return f"[ERROR] {exc}"

        def scan_folder_handler(file_paths):
            if not file_paths:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip(), "No folder selected."
            tif_paths = [f for f in file_paths if str(f).lower().endswith((".tif", ".tiff"))]
            if not tif_paths:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip(), "No .tif files found in the selected folder."
            # Copy into one real folder rather than os.path.commonpath(tif_paths) -- Gradio
            # uploads each file into its OWN per-file temp subdirectory for file_count="directory",
            # so the common path is just their shared temp-root ancestor, not a folder any of the
            # files actually live in (see CALIBRATION_FOLDER_IMPORTS_DIR's comment above).
            folder = _import_calib_folder(tif_paths)
            names = sorted(os.path.basename(p) for p in tif_paths)
            parsed = [_parse_coords_from_filename(n) for n in names]
            if all(p is not None for p in parsed):
                coord_update = json.dumps([[r, c] for r, c in parsed])
                warn = ""
            else:
                # Previously this silently left u_offax_coord untouched (gr.skip()) whenever any
                # filename didn't match "_x###_y###" -- if that textbox had stale content from an
                # earlier scan, the files list would update but the coordinates wouldn't, with no
                # indication anything was wrong. Now it always reflects exactly what was just
                # scanned: cleared here, with an explicit warning about which files couldn't be
                # parsed, rather than silently leaving mismatched old data in place.
                coord_update = "[]"
                bad = [n for n, p in zip(names, parsed) if p is None]
                warn = f" WARNING: could not parse coordinates from: {', '.join(bad)} -- fix these filenames or set coordinates manually."
            return (
                folder, "\n".join(names),
                gr.update(choices=names, value=None),
                coord_update,
                f"Found {len(names)} .tif file(s) in {folder}.{warn}",
            )

        def move_onaxis_handler(selected, offaxis_text, offaxis_coord_text):
            # offaxis_coord_text is intentionally unused: the coordinates are always re-derived
            # from the filenames below (see the comment on coord_update), never taken from
            # whatever this textbox currently holds -- see the fix note just below for why.
            if not selected:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), "Pick a file from the dropdown first."
            lines = [ln.strip() for ln in str(offaxis_text).strip().split("\n") if ln.strip()]
            if selected not in lines:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), f"'{selected}' is not in the off-axis list."
            lines.remove(selected)

            # Re-derive the off-axis coordinate list straight from the remaining filenames,
            # rather than index-popping a separately-maintained JSON array to match. The old
            # approach could silently desync files from coordinates (e.g. if a prior scan's
            # coordinate list didn't line up 1:1 with the files list for any reason) with no
            # warning to the user. Since every filename already embeds its own "_x###_y###"
            # coordinate by construction (both from "Calibration folder" scans and from the
            # interactive Calibration Setup picker), re-parsing is strictly more robust than
            # trying to keep two parallel lists in sync through every edit -- it cannot desync.
            reparsed = [_parse_coords_from_filename(n) for n in lines]
            if not lines:
                coord_update = "[]"
                warn = ""
            elif all(p is not None for p in reparsed):
                coord_update = json.dumps([[r, c] for r, c in reparsed])
                warn = ""
            else:
                coord_update = gr.skip()
                bad = [n for n, p in zip(lines, reparsed) if p is None]
                warn = f" WARNING: could not parse coordinates from: {', '.join(bad)} -- off-axis coordinates NOT updated, check these filenames."

            parsed = _parse_coords_from_filename(selected)
            central_update = json.dumps([parsed[0], parsed[1]]) if parsed is not None else gr.skip()

            return (
                selected, "\n".join(lines), gr.update(choices=lines, value=None),
                central_update, coord_update,
                f"Moved '{selected}' to the central-bead field.{warn}",
            )

        # ── Calibration Setup handlers (click-to-crop emitter picker) ──────────

        def _range_display_text(z0: int, z1: int, Z: int) -> str:
            return f"Z-range: {z0}–{z1} ({z1 - z0 + 1}/{Z} slices)"

        def on_raw_file_uploaded(file_path):
            # a new raw file means a new picking session — wipe any emitter data left over
            # from whatever was previously loaded/picked, regardless of how this load goes, and
            # unlock the Z-range controls (they lock again once a new on-axis emitter is saved).
            cleared_fields = ("", "", "[]", "", "[]")
            unlock = (gr.update(interactive=True),) * 3
            # z_slider, z_min_display, z_max_display, z_range_display, raw_image, cropped_image
            no_change = (gr.skip(),) * 6
            if not file_path:
                return (None, [], None, *no_change,
                        "Browse a raw .tif file to begin.", "No file selected.", *cleared_fields, *unlock)
            try:
                arr = _read_tiff_with_retry(str(file_path))
            except Exception as exc:
                return (None, [], None, *no_change,
                        "Browse a raw .tif file to begin.", f"[ERROR] Could not read file: {exc}",
                        *cleared_fields, *unlock)
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim != 3:
                return (None, [], None, *no_change,
                        "Browse a raw .tif file to begin.",
                        f"[ERROR] Expected a 2D or 3D TIFF, got shape {arr.shape}.", *cleared_fields, *unlock)
            Z, H, W = arr.shape
            vmin, vmax = float(arr.min()), float(arr.max())
            # Use the ORIGINAL filename (Gradio preserves it) but save under our own dedicated,
            # freshly-timestamped folder — not next to Gradio's temp upload copy of the raw
            # file, and not reused across sessions (avoids piling up near-duplicate crops, e.g.
            # the same emitter re-picked a pixel or two off, across separate times this same raw
            # file gets browsed).
            raw_stem = Path(str(file_path)).stem
            out_dir = _make_emitters_out_dir(raw_stem)
            stack_state = {
                "array": arr, "vmin": vmin, "vmax": vmax, "out_dir": out_dir,
                "H": H, "W": W, "Z": Z, "raw_stem": raw_stem,
            }
            mid_z = Z // 2
            z_last = max(Z - 1, 0)
            img = _render_raw_frame(stack_state, mid_z, [])
            return (
                stack_state, [], None,
                # gr.Slider requires minimum < maximum strictly -- a single-frame (Z=1, non-
                # stack) upload would otherwise crash with maximum=z_last=0=minimum. Keep the
                # slider's own maximum at least 1, and disable it when there's truly only one
                # frame to browse (z_last/z_min_display/z_max_display below stay at their real
                # values -- those are plain Numbers with no such constraint).
                gr.update(minimum=0, maximum=max(z_last, 1), value=mid_z, step=1, interactive=Z > 1),
                0, z_last,
                _range_display_text(0, z_last, Z),
                img, None,
                "Click the **on-axis (central)** emitter.",
                f"Loaded {Z}x{H}x{W} stack from {file_path}.",
                *cleared_fields, *unlock,
            )

        def on_z_slider_change(z, stack_state, emitters, pending):
            if not stack_state:
                return gr.skip()
            return _render_raw_frame(stack_state, z, emitters or [], pending)

        def _make_pending(stack_state, row, col, crop_size, z_min, z_max):
            """Build the pending-crop dict (+ a multi-Z-slice preview strip) for a given
            center/size, cropping only the user-selected Z sub-range (not the full stack) — shared
            by the click handler and by reacting to a crop-size or Z-range change afterward. The
            preview spans the crop's whole (trimmed) depth, not just one Z."""
            size = max(1, int(crop_size))
            H, W = stack_state["H"], stack_state["W"]
            bbox = _crop_window(row, col, size, H, W)
            r0, r1, c0, c1 = bbox
            z0, z1 = int(z_min), int(z_max)
            crop = stack_state["array"][z0:z1 + 1, r0:r1, c0:c1]
            pending = {"row": row, "col": col, "bbox": bbox, "crop": crop}
            preview = _multi_slice_strip(crop, stack_state["vmin"], stack_state["vmax"])
            return pending, preview

        def on_raw_image_click(evt: gr.SelectData, stack_state, crop_size, z, emitters, z_min, z_max):
            if not stack_state:
                return gr.skip(), None, None, "Browse a raw .tif file first."
            if crop_size is None:
                # gr.Number reports None while its field is momentarily empty mid-edit
                return gr.skip(), gr.skip(), gr.skip(), "Enter a crop size before clicking an emitter."
            col, row = int(evt.index[0]), int(evt.index[1])
            pending, preview = _make_pending(stack_state, row, col, crop_size, z_min, z_max)
            marked = _render_raw_frame(stack_state, z, emitters or [], pending)
            return pending, marked, preview, f"Clicked (row={row}, col={col}). Review the crop, then Confirm or Clear."

        def on_crop_size_change(crop_size, stack_state, pending, z, emitters, z_min, z_max):
            # only resizes a still-pending (unconfirmed) selection — already-saved crops keep
            # whatever size they were actually cropped+saved at. crop_size is None while the
            # field is momentarily empty mid-edit (e.g. clearing "70" before typing "50") — wait
            # for the next change event with a real number instead of erroring on it.
            if not stack_state or not pending or crop_size is None:
                return gr.skip(), gr.skip(), gr.skip()
            new_pending, preview = _make_pending(stack_state, pending["row"], pending["col"], crop_size, z_min, z_max)
            marked = _render_raw_frame(stack_state, z, emitters or [], new_pending)
            return new_pending, marked, preview

        def _apply_z_bound(new_bound, other_bound, is_min, stack_state, pending, crop_size):
            """Shared logic for the Set-min/Set-max buttons: capture the raw-image slider's
            current position as that bound, push the other bound along if crossed (so min can
            never exceed max), update the range display, and — if a crop is still pending —
            re-crop it to the new range too (same "only touches the pending selection" rule the
            crop-size field already follows)."""
            if not stack_state:
                return (gr.skip(),) * 5
            Z = stack_state["Z"]
            new_bound = max(0, min(int(new_bound), Z - 1))
            other_bound = int(other_bound)
            if is_min:
                z_min, z_max = new_bound, max(new_bound, other_bound)
            else:
                z_min, z_max = min(new_bound, other_bound), new_bound
            display = _range_display_text(z_min, z_max, Z)
            # crop_size can be None while its field is momentarily empty mid-edit — skip
            # re-cropping the pending selection in that case rather than erroring on it.
            if pending and crop_size is not None:
                new_pending, preview = _make_pending(
                    stack_state, pending["row"], pending["col"], crop_size, z_min, z_max
                )
            else:
                new_pending, preview = pending, gr.skip()
            return z_min, z_max, display, new_pending, preview

        def on_set_min_click(z, z_max, stack_state, pending, crop_size):
            return _apply_z_bound(z, z_max, True, stack_state, pending, crop_size)

        def on_set_max_click(z, z_min, stack_state, pending, crop_size):
            return _apply_z_bound(z, z_min, False, stack_state, pending, crop_size)

        def on_confirm(pending, stack_state, emitters, z, offaxis_files_text, offaxis_coord_text):
            no_field_change = (gr.skip(),) * 5
            no_lock_change = (gr.skip(), gr.skip(), gr.skip())
            if not pending or not stack_state:
                return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                        "Nothing to confirm — click an emitter first.", *no_field_change, *no_lock_change)
            if len(emitters) >= 15:
                return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                        "Already have the max of 15 emitters.", *no_field_change, *no_lock_change)

            row, col, bbox, crop = pending["row"], pending["col"], pending["bbox"], pending["crop"]
            is_onaxis = len(emitters) == 0

            if is_onaxis:
                tag = "center"
            else:
                onaxis = next(e for e in emitters if e["is_onaxis"])
                tag = _compass_tag(row - onaxis["row"], col - onaxis["col"])

            filename = f"{tag}_x{col}_y{row}.tif"
            out_dir = stack_state["out_dir"]
            os.makedirs(out_dir, exist_ok=True)
            tifffile.imwrite(os.path.join(out_dir, filename), crop)

            new_emitters = emitters + [
                {"is_onaxis": is_onaxis, "row": row, "col": col, "bbox": bbox, "filename": filename}
            ]
            count = len(new_emitters)

            if is_onaxis:
                field_updates = (out_dir, filename, json.dumps([row, col]), gr.skip(), gr.skip())
            else:
                lines = [ln.strip() for ln in str(offaxis_files_text).strip().split("\n") if ln.strip()]
                lines.append(filename)
                try:
                    coords = json.loads(str(offaxis_coord_text)) if str(offaxis_coord_text).strip() else []
                    if not isinstance(coords, list):
                        coords = []
                except (ValueError, TypeError):
                    coords = []
                coords.append([row, col])
                field_updates = (gr.skip(), gr.skip(), gr.skip(), "\n".join(lines), json.dumps(coords))

            if count >= 15:
                instruction = f"Done — 15 emitters saved to {out_dir}."
            elif is_onaxis:
                instruction = ("On-axis emitter confirmed — crop size and Z-range are now locked "
                               "so every emitter matches. Click off-axis emitter #1 (up to 14 more).")
            else:
                instruction = f"Click off-axis emitter #{count} (up to 15 total, {count} confirmed)."

            # lock crop size + Z-range in place after the on-axis emitter, so every later crop is
            # the same X/Y size and Z depth — re-picking any of these between emitters would
            # otherwise leave some crops a different shape than others, which downstream code
            # (fixed-shape stacking across beads) can't handle. "Restart All" is the escape hatch.
            lock_update = (gr.update(interactive=False),) * 3 if is_onaxis else no_lock_change

            status = f"Saved {filename} ({crop.shape[0]}x{crop.shape[1]}x{crop.shape[2]})."
            marked = _render_raw_frame(stack_state, z, new_emitters)
            return (new_emitters, None, marked, None, instruction, status, *field_updates, *lock_update)

        def on_clear(stack_state, z, emitters):
            if not stack_state:
                return None, gr.skip(), None, "Cleared — click a new emitter location."
            marked = _render_raw_frame(stack_state, z, emitters or [])
            return None, marked, None, "Cleared — click a new emitter location."

        def on_restart_all(stack_state, z):
            """Wipe every confirmed emitter for the current raw file, unlock crop size +
            Z-range, and empty out this session's own output folder on disk — deletes it
            (recreated fresh by the next Confirm's os.makedirs) rather than spawning yet another
            timestamped folder, so repeated restarts on one raw file don't leave a trail of
            near-empty folders behind. Safe to rmtree here: out_dir is this session's own
            uniquely-timestamped subfolder, never the shared CALIBRATION_EMITTERS_DIR constant."""
            if not stack_state:
                return (gr.skip(),) * 8 + ("Nothing to restart — no file loaded.",) + (gr.skip(),) * 8
            Z = stack_state["Z"]
            z_last = max(Z - 1, 0)
            out_dir = stack_state["out_dir"]
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            marked = _render_raw_frame(stack_state, z, [])
            cleared_fields = ("", "", "[]", "", "[]")
            reset_locks = (
                gr.update(interactive=True, value=70), gr.update(interactive=True), gr.update(interactive=True),
            )
            return (
                [], None, 0, z_last, _range_display_text(0, z_last, Z), marked, None,
                "Click the **on-axis (central)** emitter.",
                "Restarted — all emitters cleared and this session's output folder emptied.",
                *cleared_fields, *reset_locks,
            )

        def microscope_load_handler(name):
            data = _load_microscopes()
            preset = data.get(name)
            if preset is None:
                return [gr.update()] * len(MICROSCOPE_FIELDS)
            return [preset.get(k, gr.update()) for k in MICROSCOPE_FIELDS]

        def microscope_save_handler(name, M, NA, n_immersion, f_4f, ps_camera, ps_BFP, n_sample, bitdepth):
            name = str(name).strip()
            if not name:
                return gr.update(), "[ERROR] Enter a microscope name first."
            data = _load_microscopes()
            data[name] = {
                "M": float(M), "NA": float(NA), "n_immersion": float(n_immersion),
                "f_4f": float(f_4f), "ps_camera": float(ps_camera), "ps_BFP": float(ps_BFP),
                "n_sample": float(n_sample), "bitdepth": int(float(bitdepth)),
            }
            _save_microscopes(data)
            return gr.update(choices=list(data.keys()), value=name), f"Saved microscope '{name}'."

        # ── Per-emitter results viewer handlers ─────────────────────────────────

        def _find_emitter_by_name(emitters, name):
            return next((e for e in emitters if Path(e["filename"]).stem == name), None)

        def on_results_bead_select(bead_name, beads, stack_state, emitters):
            bead = next((b for b in beads if b["name"] == bead_name), None)
            if bead is None:
                return gr.skip(), None, "Pick an emitter to view its results."
            img = _render_bead_comparison(bead["exp_path"], bead["sim_path"])
            status = f"Showing '{bead_name}' — calculated (top) vs experimental (bottom)."
            if not stack_state or not emitters:
                return gr.skip(), img, status
            selected = _find_emitter_by_name(emitters, bead_name)
            marked = _render_raw_frame(stack_state, stack_state["Z"] // 2, emitters, selected)
            return marked, img, status

        def on_results_image_click(evt: gr.SelectData, emitters, beads, stack_state):
            if not emitters or not stack_state:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip()
            col, row = int(evt.index[0]), int(evt.index[1])
            # require the click to actually land inside an emitter's own crop box — clicking
            # empty background shouldn't silently jump to "nearest" bead.
            hits = [e for e in emitters
                    if e["bbox"][0] <= row < e["bbox"][1] and e["bbox"][2] <= col < e["bbox"][3]]
            if not hits:
                return gr.skip(), gr.skip(), gr.skip(), "Click inside an emitter's box to select it."
            closest = min(hits, key=lambda e: (e["row"] - row) ** 2 + (e["col"] - col) ** 2)
            name = Path(closest["filename"]).stem
            bead = next((b for b in beads if b["name"] == name), None)
            if bead is None:
                return gr.update(value=None), gr.skip(), None, f"'{name}' wasn't part of this run's results."
            img = _render_bead_comparison(bead["exp_path"], bead["sim_path"])
            marked = _render_raw_frame(stack_state, stack_state["Z"] // 2, emitters, closest)
            return (gr.update(value=name), marked, img,
                    f"Showing '{name}' — calculated (top) vs experimental (bottom).")

        def stop_handler():
            if _run_state["stop_event"] is not None:
                _run_state["stop_event"].set()
                gr.Info(
                    "Stop requested — finishing the current epoch and saving results. "
                    "This can take a little while.",
                    duration=6,
                )
            return gr.update(value="⏳ Stopping…", interactive=False)

        def run_handler(raw_stack_state, emitters_state_val, *vals):
            # results_beads_state, results_dropdown, results_image, results_plot, results_status
            no_results_change = (gr.skip(),) * 5
            # also blocks against a concurrent Generate Training Data run: both worker threads
            # redirect the process-global sys.stdout, which corrupts each other's log streams
            # (and each other's redirection) if they ever run at the same time.
            if _run_state["busy"] or _td_run_state["busy"] or _train_run_state["busy"]:
                yield ("[ERROR] Another run (Phase Retrieval, Generate Training Data, or Train Model) "
                       "is already in progress.",
                       gr.skip(), gr.update(interactive=False), gr.update(interactive=True), *no_results_change)
                return

            try:
                cfg = fields_to_config(*vals)
            except Exception as exc:
                yield (f"[CONFIG ERROR] {exc}", gr.skip(),
                       gr.update(interactive=True), gr.update(interactive=False), *no_results_change)
                return

            q: queue.SimpleQueue = queue.SimpleQueue()
            old_stdout = sys.stdout
            sys.stdout = _StreamToQueue(q)
            done_evt = threading.Event()
            run_error: list = [None]
            live_box: dict = {}
            stop_event = threading.Event()
            _run_state["stop_event"] = stop_event
            _run_state["busy"] = True

            def _worker():
                try:
                    characterize_PSF(cfg, live_box=live_box, stop_event=stop_event)
                except Exception as exc:
                    q.put(f"\n[EXCEPTION] {exc}\n")
                    run_error[0] = exc
                finally:
                    sys.stdout = old_stdout
                    _run_state["busy"] = False
                    done_evt.set()

            threading.Thread(target=_worker, daemon=True).start()

            log = ""
            last_seen_version = 0
            while True:
                try:
                    chunk = q.get(timeout=0.2)
                    log += chunk
                except queue.Empty:
                    if done_evt.is_set():
                        break
                    # heartbeat keeps the WebSocket alive — fall through to yield below

                version = live_box.get("version", 0)
                if version != last_seen_version:
                    last_seen_version = version
                    plot_update = _build_live_figure(live_box)
                else:
                    plot_update = gr.skip()
                # once Stop has been clicked, stop_handler already set the "Stopping…" label —
                # keep the button disabled (don't touch its value) instead of re-enabling it
                stop_btn_update = gr.skip() if stop_event.is_set() else gr.update(interactive=True)
                yield log, plot_update, gr.update(interactive=False), stop_btn_update, *no_results_change

            while not q.empty():
                log += q.get_nowait()

            version = live_box.get("version", 0)
            if version != last_seen_version:
                plot_update = _build_live_figure(live_box)
            else:
                plot_update = gr.skip()

            _run_state["stop_event"] = None
            log += "\n\n--- DONE ---" if run_error[0] is None else f"\n\n--- FAILED: {run_error[0]} ---"

            # Populate the per-emitter results viewer from phase_retrieval()'s own saved
            # outputs — only on a clean finish (full run or user Stop); on an exception, leave
            # it untouched rather than showing possibly-stale results from an earlier run.
            if run_error[0] is None:
                beads = _discover_run_beads(RESULTS_DIR, _expected_bead_names(cfg))
                bead_names = [b["name"] for b in beads]
                default_bead = next((b for b in beads if b["cnt"] == 0), beads[0] if beads else None)
                if default_bead is not None:
                    results_plot_update = _render_bead_comparison(default_bead["exp_path"], default_bead["sim_path"])
                    results_status_update = (
                        f"Showing '{default_bead['name']}' — calculated (top) vs experimental (bottom)."
                    )
                    dropdown_update = gr.update(choices=bead_names, value=default_bead["name"])
                else:
                    results_plot_update = None
                    results_status_update = "Run finished but no per-emitter result files were found."
                    dropdown_update = gr.update(choices=[], value=None)

                if emitters_state_val and raw_stack_state:
                    selected = (
                        _find_emitter_by_name(emitters_state_val, default_bead["name"])
                        if default_bead is not None else None
                    )
                    marked = _render_raw_frame(
                        raw_stack_state, raw_stack_state["Z"] // 2, emitters_state_val, selected
                    )
                    image_update = gr.update(value=marked, visible=True)
                else:
                    image_update = gr.update(visible=False)

                results_outputs = (beads, dropdown_update, image_update, results_plot_update, results_status_update)
            else:
                results_outputs = no_results_change

            yield (log, plot_update, gr.update(interactive=True),
                   gr.update(value="Stop", interactive=False), *results_outputs)

        # ── Generate Training Data tab handlers ─────────────────────────────────

        def on_td_load_pr():
            return _load_pr_results_and_status()

        def on_td_frame_uploaded(file_path):
            no_z = gr.update(minimum=0, maximum=1, value=0)
            if not file_path:
                return None, None, None, None, "No file selected.", no_z
            try:
                arr = _read_tiff_with_retry(file_path)
            except Exception as exc:
                return None, None, None, None, f"[ERROR] Could not read {file_path}: {exc}", no_z
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim != 3:
                return (None, None, None, None,
                        f"[ERROR] Expected a 2D (or 3D Z-stack) TIFF, got shape {arr.shape}.", no_z)
            Z = arr.shape[0]
            vmin, vmax = float(arr.min()), float(arr.max())
            frame_state = {"array": arr, "vmin": vmin, "vmax": vmax, "Z": Z}
            mid_z = Z // 2
            rgb = _render_td_marks(frame_state, mid_z, None, None)
            # gr.Slider requires minimum < maximum strictly -- a single-frame (Z=1) upload,
            # which is a normal/expected input here (a plain experimental frame, not
            # necessarily a stack), would otherwise crash with maximum=0=minimum.
            z_update = gr.update(minimum=0, maximum=max(Z - 1, 1), value=mid_z, step=1, interactive=Z > 1)
            return (frame_state, None, None, rgb,
                    "Frame loaded — mark a no-emitter region and a bright-emitter region. "
                    "Use Z-slice to browse other frames of the stack (e.g. to catch a blinking emitter).",
                    z_update)

        def _render_td_marks(frame_state, z, noise_bbox, emitter_bbox):
            z = max(0, min(int(z), frame_state["Z"] - 1))
            gray = _normalize_slice(frame_state["array"][z], frame_state["vmin"], frame_state["vmax"])
            rgb = np.stack([gray, gray, gray], axis=-1).copy()
            if noise_bbox is not None:
                _draw_box(rgb, noise_bbox, _PENDING_BOX_COLOR)
            if emitter_bbox is not None:
                _draw_box(rgb, emitter_bbox, _TD_EMITTER_BOX_COLOR)
            return rgb

        def on_td_z_slider_change(z, frame_state, noise_bbox, emitter_bbox):
            if not frame_state:
                return gr.skip()
            return _render_td_marks(frame_state, z, noise_bbox, emitter_bbox)

        def on_td_frame_click(evt: gr.SelectData, frame_state, w, h, mode, z,
                               noise_bbox, emitter_bbox, pr_results, *vals):
            no_seed = (gr.skip(),) * 6
            if not frame_state:
                return gr.skip(), gr.skip(), gr.skip(), "Upload an experimental frame first.", *no_seed
            if w is None or h is None or float(w) <= 0 or float(h) <= 0:
                return (gr.skip(), gr.skip(), gr.skip(),
                        "Enter a positive marked-patch width/height first.", *no_seed)
            z = max(0, min(int(z), frame_state["Z"] - 1))
            arr = frame_state["array"][z]
            col, row = int(evt.index[0]), int(evt.index[1])
            H, W = arr.shape
            bbox = _crop_window_rect(row, col, int(w), int(h), H, W)
            is_noise_mode = mode.startswith("No-emitter")
            noise_bbox = bbox if is_noise_mode else noise_bbox
            emitter_bbox = bbox if not is_noise_mode else emitter_bbox
            rgb = _render_td_marks(frame_state, z, noise_bbox, emitter_bbox)

            if noise_bbox is None or emitter_bbox is None:
                missing = "a bright-emitter region" if noise_bbox is not None else "a no-emitter region"
                status = f"Marked. Now also mark {missing} to calibrate Background/Noise offset/Signal."
                return noise_bbox, emitter_bbox, rgb, status, *no_seed

            # _simulate_one_frame does poisson(canvas + background) - background + offset:
            # `background` is subtracted back out after sampling, so it only ever contributes
            # NOISE VARIANCE to the output, never a mean shift; `offset` is the only thing that
            # sets the actual output baseline. So the no-emitter patch's mean (the real camera
            # baseline) seeds Noise offset, and its std^2 (the real noise variance) seeds
            # Background -- not the other way around, which was the original bug: it injected
            # the raw baseline as if it were noise variance (way too much noise) while leaving
            # the simulated background sitting at ~0 instead of the real baseline.
            mean, std = app_utils.noise_patch_stats(arr, noise_bbox)
            variance = std ** 2
            bg_min, bg_max = max(0.0, 0.7 * variance), 1.3 * variance
            off_min = off_max = mean
            er0, er1, ec0, ec1 = emitter_bbox
            exp_maxv = float(arr[er0:er1, ec0:ec1].max())

            if pr_results is None:
                status = (f"Baseline mean={mean:.1f}, std={std:.1f}; emitter peak={exp_maxv:.1f}. "
                          f"Seeded Background≈{variance:.1f}, Noise offset≈{mean:.1f}. "
                          f"Load Phase Retrieval Results to also calibrate Signal.")
                return (noise_bbox, emitter_bbox, rgb, status,
                        bg_min, bg_max, off_min, off_max, gr.skip(), gr.skip())

            try:
                cfg = fields_to_config(*vals)
                param_dict = cfg.generate_training_param_dict(pr_results)
                sig_min, sig_max = app_utils.estimate_signal_range(param_dict, mean, exp_maxv)
                status = (f"Baseline mean={mean:.1f}, std={std:.1f}; emitter peak={exp_maxv:.1f}. "
                          f"Seeded Background≈{variance:.1f}, Noise offset≈{mean:.1f}, "
                          f"Signal≈({sig_min:.0f},{sig_max:.0f}) photons.")
            except Exception as exc:
                sig_min = sig_max = gr.skip()
                status = (f"Baseline mean={mean:.1f}, std={std:.1f}; emitter peak={exp_maxv:.1f}. "
                          f"Seeded Background/Noise offset, but Signal calibration failed: {exc}")

            return (noise_bbox, emitter_bbox, rgb, status,
                    bg_min, bg_max, off_min, off_max, sig_min, sig_max)

        def on_td_update_preview(pr_results, frame_state, z, *vals):
            if pr_results is None:
                return None, "Load Phase Retrieval Results first."
            try:
                cfg = fields_to_config(*vals)
                param_dict = cfg.generate_training_param_dict(pr_results)
                sim = app_utils.generate_training_frame(param_dict)
            except Exception as exc:
                return None, f"[ERROR] {exc}"

            n_panels = 2 if frame_state else 1
            fig = Figure(figsize=(5 * n_panels, 5), constrained_layout=True)
            FigureCanvasAgg(fig)
            axes = fig.subplots(1, n_panels)
            axes = [axes] if n_panels == 1 else list(axes)
            idx = 0
            if frame_state:
                zc = max(0, min(int(z), frame_state["Z"] - 1))
                arr = frame_state["array"][zc]
                axes[idx].imshow(arr, cmap="gray", vmin=frame_state["vmin"], vmax=frame_state["vmax"])
                axes[idx].set_title(f"experimental frame (z={zc})")
                axes[idx].axis("off")
                idx += 1
            axes[idx].imshow(sim, cmap="gray")
            axes[idx].set_title("simulated frame")
            axes[idx].axis("off")
            return fig, "Preview updated."

        def on_td_simulate(pr_results, out_dir, n_ims, *vals):
            # also blocks against a concurrent Run-tab phase retrieval — see the matching guard
            # in run_handler for why (shared sys.stdout redirection).
            if _td_run_state["busy"] or _run_state["busy"] or _train_run_state["busy"]:
                yield ("[ERROR] Another run (Phase Retrieval, Generate Training Data, or Train Model) "
                       "is already in progress.",
                       gr.update(interactive=False), gr.update(interactive=True))
                return
            if pr_results is None:
                yield "[ERROR] Load Phase Retrieval Results first.", gr.update(interactive=True), gr.update(interactive=False)
                return
            try:
                cfg = fields_to_config(*vals)
                param_dict = cfg.generate_training_param_dict(pr_results)
            except Exception as exc:
                yield f"[CONFIG ERROR] {exc}", gr.update(interactive=True), gr.update(interactive=False)
                return

            q: queue.SimpleQueue = queue.SimpleQueue()
            old_stdout = sys.stdout
            sys.stdout = _StreamToQueue(q)
            done_evt = threading.Event()
            run_error: list = [None]
            stop_event = threading.Event()
            _td_run_state["stop_event"] = stop_event
            _td_run_state["busy"] = True

            def _worker():
                try:
                    app_utils.generate_training_data(param_dict, str(out_dir), int(n_ims), stop_event=stop_event)
                except Exception as exc:
                    q.put(f"\n[EXCEPTION] {exc}\n")
                    run_error[0] = exc
                finally:
                    sys.stdout = old_stdout
                    _td_run_state["busy"] = False
                    done_evt.set()

            threading.Thread(target=_worker, daemon=True).start()

            log = ""
            while True:
                try:
                    chunk = q.get(timeout=0.2)
                    log += chunk
                except queue.Empty:
                    if done_evt.is_set():
                        break
                stop_btn_update = gr.skip() if stop_event.is_set() else gr.update(interactive=True)
                yield log, gr.update(interactive=False), stop_btn_update

            while not q.empty():
                log += q.get_nowait()

            _td_run_state["stop_event"] = None
            log += "\n\n--- DONE ---" if run_error[0] is None else f"\n\n--- FAILED: {run_error[0]} ---"
            yield log, gr.update(interactive=True), gr.update(value="Stop", interactive=False)

        def on_td_stop():
            if _td_run_state["stop_event"] is not None:
                _td_run_state["stop_event"].set()
                gr.Info("Stop requested — finishing the current frame.", duration=6)
            return gr.update(value="⏳ Stopping…", interactive=False)

        # ── Train Model tab handlers ─────────────────────────────────────────────

        def on_train_load_td(training_data_dir):
            try:
                meta = app_utils.load_training_data_metadata(str(training_data_dir))
            except Exception as exc:
                return None, f"[ERROR] Could not load training data: {exc}"
            if meta is None:
                return None, ("No training data found yet at this path — point this at any "
                               "folder generate_training_data() has written to (a previous run's "
                               "output is fine, not just the one just generated in this session), "
                               "or run Generate Training Data first.")
            n_frames = len(meta["labels"]) - len(app_utils._TRAINING_DATA_LABEL_METADATA_KEYS)
            D, HH, WW = meta["labels"]["volume_size"]
            status = f"Loaded {max(n_frames, 0)} frame(s). Volume size (D,H,W) = ({D},{HH},{WW})."
            try:
                data_warnings = app_utils.check_training_data_folder(str(training_data_dir), meta)
            except Exception as exc:
                data_warnings = [f"(sanity checks themselves failed, ignoring: {exc})"]
            if data_warnings:
                status += " ⚠ " + " ".join(data_warnings)
            return meta, status

        def on_train_start(td_meta, *vals):
            if _train_run_state["busy"] or _run_state["busy"] or _td_run_state["busy"]:
                yield ("[ERROR] Another run (Phase Retrieval, Generate Training Data, or Train Model) "
                       "is already in progress.", gr.skip(), gr.skip(),
                       gr.update(interactive=False), gr.update(interactive=True))
                return
            if td_meta is None:
                yield ("[ERROR] Load Training Data first.", gr.skip(), gr.skip(),
                       gr.update(interactive=True), gr.update(interactive=False))
                return
            try:
                cfg = fields_to_config(*vals)
                param_dict, training_dict = cfg.generate_training_run_dict(td_meta)
            except Exception as exc:
                yield (f"[CONFIG ERROR] {exc}", gr.skip(), gr.skip(),
                       gr.update(interactive=True), gr.update(interactive=False))
                return

            q: queue.SimpleQueue = queue.SimpleQueue()
            old_stdout = sys.stdout
            sys.stdout = _StreamToQueue(q)
            done_evt = threading.Event()
            run_error: list = [None]
            result: list = [None]
            live_box: dict = {}
            stop_event = threading.Event()
            _train_run_state["stop_event"] = stop_event
            _train_run_state["busy"] = True

            def _worker():
                try:
                    result[0] = app_utils.train_model(param_dict, training_dict, live_box=live_box, stop_event=stop_event)
                except Exception as exc:
                    q.put(f"\n[EXCEPTION] {exc}\n")
                    run_error[0] = exc
                finally:
                    sys.stdout = old_stdout
                    _train_run_state["busy"] = False
                    done_evt.set()

            threading.Thread(target=_worker, daemon=True).start()

            log = ""
            last_seen_version = 0
            while True:
                try:
                    chunk = q.get(timeout=0.2)
                    log += chunk
                except queue.Empty:
                    if done_evt.is_set():
                        break

                version = live_box.get("version", 0)
                if version != last_seen_version:
                    last_seen_version = version
                    plot_update = _build_train_live_figure(live_box)
                else:
                    plot_update = gr.skip()
                stop_btn_update = gr.skip() if stop_event.is_set() else gr.update(interactive=True)
                yield log, plot_update, gr.skip(), gr.update(interactive=False), stop_btn_update

            while not q.empty():
                log += q.get_nowait()

            version = live_box.get("version", 0)
            plot_update = _build_train_live_figure(live_box) if version != last_seen_version else gr.skip()

            _train_run_state["stop_event"] = None
            log += "\n\n--- DONE ---" if run_error[0] is None else f"\n\n--- FAILED: {run_error[0]} ---"
            summary = f"net_file={result[0][0]}, fit_file={result[0][1]}" if result[0] else gr.skip()
            yield (log, plot_update, summary,
                   gr.update(interactive=True), gr.update(value="Stop", interactive=False))

        def on_train_stop():
            if _train_run_state["stop_event"] is not None:
                _train_run_state["stop_event"].set()
                gr.Info("Stop requested — finishing the current epoch and saving a resumable checkpoint.", duration=6)
            return gr.update(value="⏳ Stopping…", interactive=False)

        load_file.change(fn=load_handler, inputs=load_file, outputs=all_fields)
        save_btn.click(fn=save_handler, inputs=all_fields, outputs=save_status)
        folder_upload.upload(
            fn=scan_folder_handler, inputs=[folder_upload],
            outputs=[u_calib_root_dir, u_offax_files, onaxis_picker, u_offax_coord, scan_status],
        )
        move_onaxis_btn.click(
            fn=move_onaxis_handler, inputs=[onaxis_picker, u_offax_files, u_offax_coord],
            outputs=[u_zstack, u_offax_files, onaxis_picker, u_central, u_offax_coord, scan_status],
        )
        calib_mode.change(
            fn=lambda mode: (
                gr.update(visible=mode == "Calibration folder"),
                gr.update(visible=mode == "Calibration setup"),
            ),
            inputs=calib_mode, outputs=[calib_folder_group, calib_setup_group],
        )
        raw_file_upload.upload(
            fn=on_raw_file_uploaded, inputs=[raw_file_upload],
            outputs=[raw_stack_state, emitters_state, pending_crop_state,
                     z_slider, z_min_display, z_max_display, z_range_display,
                     raw_image, cropped_image, setup_instruction, setup_status,
                     u_calib_root_dir, u_zstack, u_central, u_offax_files, u_offax_coord,
                     crop_size_input, set_min_btn, set_max_btn],
        )
        z_slider.change(
            fn=on_z_slider_change,
            inputs=[z_slider, raw_stack_state, emitters_state, pending_crop_state],
            outputs=[raw_image],
        )
        set_min_btn.click(
            fn=on_set_min_click,
            inputs=[z_slider, z_max_display, raw_stack_state, pending_crop_state, crop_size_input],
            outputs=[z_min_display, z_max_display, z_range_display, pending_crop_state, cropped_image],
        )
        set_max_btn.click(
            fn=on_set_max_click,
            inputs=[z_slider, z_min_display, raw_stack_state, pending_crop_state, crop_size_input],
            outputs=[z_min_display, z_max_display, z_range_display, pending_crop_state, cropped_image],
        )
        raw_image.select(
            fn=on_raw_image_click,
            inputs=[raw_stack_state, crop_size_input, z_slider, emitters_state, z_min_display, z_max_display],
            outputs=[pending_crop_state, raw_image, cropped_image, setup_status],
        )
        crop_size_input.change(
            fn=on_crop_size_change,
            inputs=[crop_size_input, raw_stack_state, pending_crop_state, z_slider,
                    emitters_state, z_min_display, z_max_display],
            outputs=[pending_crop_state, raw_image, cropped_image],
        )
        confirm_btn.click(
            fn=on_confirm,
            inputs=[pending_crop_state, raw_stack_state, emitters_state, z_slider,
                    u_offax_files, u_offax_coord],
            outputs=[emitters_state, pending_crop_state, raw_image, cropped_image,
                     setup_instruction, setup_status,
                     u_calib_root_dir, u_zstack, u_central, u_offax_files, u_offax_coord,
                     crop_size_input, set_min_btn, set_max_btn],
        )
        clear_btn.click(
            fn=on_clear, inputs=[raw_stack_state, z_slider, emitters_state],
            outputs=[pending_crop_state, raw_image, cropped_image, setup_status],
        )
        restart_btn.click(
            fn=on_restart_all,
            inputs=[raw_stack_state, z_slider],
            outputs=[emitters_state, pending_crop_state, z_min_display, z_max_display, z_range_display,
                     raw_image, cropped_image, setup_instruction, setup_status,
                     u_calib_root_dir, u_zstack, u_central, u_offax_files, u_offax_coord,
                     crop_size_input, set_min_btn, set_max_btn],
        )
        m_dropdown.change(fn=microscope_load_handler, inputs=m_dropdown, outputs=microscope_fields)
        m_save_btn.click(fn=microscope_save_handler, inputs=[m_name] + microscope_fields, outputs=[m_dropdown, m_status])
        run_btn.click(
            fn=run_handler, inputs=[raw_stack_state, emitters_state] + all_fields,
            outputs=[log_out, live_plot, run_btn, stop_btn,
                     results_beads_state, results_dropdown, results_image, results_plot, results_status],
        )
        stop_btn.click(fn=stop_handler, outputs=stop_btn)
        results_dropdown.change(
            fn=on_results_bead_select,
            inputs=[results_dropdown, results_beads_state, raw_stack_state, emitters_state],
            outputs=[results_image, results_plot, results_status],
        )
        results_image.select(
            fn=on_results_image_click,
            inputs=[emitters_state, results_beads_state, raw_stack_state],
            outputs=[results_dropdown, results_image, results_plot, results_status],
        )
        td_load_pr_btn.click(fn=on_td_load_pr, outputs=[td_pr_results_state, td_pr_status])
        td_frame_upload.upload(
            fn=on_td_frame_uploaded, inputs=[td_frame_upload],
            outputs=[td_frame_state, td_noise_bbox_state, td_emitter_bbox_state,
                     td_frame_image, td_noise_status, td_z_slider],
        )
        td_z_slider.change(
            fn=on_td_z_slider_change,
            inputs=[td_z_slider, td_frame_state, td_noise_bbox_state, td_emitter_bbox_state],
            outputs=[td_frame_image],
        )
        td_frame_image.select(
            fn=on_td_frame_click,
            inputs=[td_frame_state, td_noise_w, td_noise_h, td_mark_mode, td_z_slider,
                    td_noise_bbox_state, td_emitter_bbox_state, td_pr_results_state] + all_fields,
            outputs=[td_noise_bbox_state, td_emitter_bbox_state, td_frame_image, td_noise_status,
                     td_bg_min, td_bg_max, td_noise_off_min, td_noise_off_max, td_sig_min, td_sig_max],
        )
        td_update_preview_btn.click(
            fn=on_td_update_preview,
            inputs=[td_pr_results_state, td_frame_state, td_z_slider] + all_fields,
            outputs=[td_preview_plot, td_preview_status],
        )
        td_simulate_btn.click(
            fn=on_td_simulate, inputs=[td_pr_results_state, td_out_dir, td_n_ims] + all_fields,
            outputs=[td_log_out, td_simulate_btn, td_stop_btn],
        )
        td_stop_btn.click(fn=on_td_stop, outputs=td_stop_btn)
        # Train Model's training-data-folder field auto-populates from Generate Training Data's
        # own output-folder field, so the path doesn't need retyping between tabs — still a plain
        # editable Textbox the user can override by hand at any time.
        td_out_dir.change(fn=lambda v: v, inputs=[td_out_dir], outputs=[train_data_dir])
        train_load_td_btn.click(fn=on_train_load_td, inputs=[train_data_dir], outputs=[train_td_meta_state, train_td_status])
        train_start_btn.click(
            fn=on_train_start, inputs=[train_td_meta_state] + all_fields,
            outputs=[train_log_out, train_live_plot, train_summary, train_start_btn, train_stop_btn],
        )
        train_stop_btn.click(fn=on_train_stop, outputs=train_stop_btn)

        demo.load(None, None, None, js=SETUP_CROP_CURSOR_JS)

    demo.queue()
    return demo
