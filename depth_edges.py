#!/usr/bin/env python
# Copyright (c) 2026, 7th software Ltd.
# All rights reserved.

"""
Depth Anything V2 depth-edge overlay (OpenCV).

This program draws outlines around depth discontinuities (object boundaries) and overlays them onto
the input video. Each outline is coloured by relative depth, so you can quickly see which objects are
nearer or farther without filling the whole frame with a depth heatmap.

Why this exists
- A full depth overlay is often visually noisy. Outlines are much easier to read.
- Directly colouring thick outlines from per-pixel depth tends to pick up the background, because the
  thickened edge mask spills onto the far side of the boundary.
- Outdoor scenes often have poor local depth contrast. Useful edges can be hard to detect unless you
  boost contrast in the depth signal used for edge finding.

Core pipeline (per frame)
1) Depth inference:
   A monocular depth model estimates a relative depth map for the frame. The output is not in metres.

2) Normalise and stabilise the depth signal:
   We work in "near score" space (nearer should be larger) and apply robust normalisation using
   percentiles. This reduces the impact of outliers and dodgy predictions.

   Optional knee remap (--knee-near-frac):
   When enabled, we remap the normalised values so that at least a chosen fraction of pixels occupies
   the nearest half of the value range. This stops very distant regions (for example the horizon) from
   monopolising the dynamic range and flattening the foreground.

3) Decide outline colours cheaply and robustly using a coarse grid:
   We downsample the normalised near-score image to a small grid using an averaging resize. We then
   apply a 3x3 neighbourhood max operation on that grid, and upsample back to full resolution using
   nearest neighbour (blocky by design).

   Why: this produces a near-biased, spatially coarse field that is cheap to compute and tends to
   assign an outline the depth of the nearby foreground surface, even when the edge mask is thick.

4) Detect edges using a depth signal that is allowed to be locally "enhanced":
   We generate the edge mask from gradients of the full-resolution near-score image.
   If --contrast-boost is enabled, we apply CLAHE to the near-score image before edge detection.

   Why: CLAHE improves local contrast in the depth signal, which makes meaningful boundaries easier
   to detect (especially outdoors). It is applied only after the colour grid is computed, so the colour
   mapping remains globally consistent and is not locally warped by CLAHE.

5) Compose:
   We colourise the blocky near-biased field with a colormap, mask it by the edge map, and alpha-blend
   the result over the original frame.

Important notes
- Depth is relative to the scene. Colours indicate nearer vs farther, not absolute distance.
- Robust normalisation helps with occasional bad depth values, but reflective/transparent objects can
  still cause artefacts.
- If you enable contrast boost, you may get more edges and some extra flicker. Temporal smoothing of
  the edge mask can help.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


@dataclass
class DeviceConfig:
    device: torch.device
    use_amp: bool


class EdgeTemporalEMA:
    """
    EMA smoother for a uint8 edge mask (0/255).

    Uses cv2.accumulateWeighted for speed. Stores a float32 buffer in [0,1].
    """
    def __init__(self) -> None:
        self.buf: Optional[np.ndarray] = None  # float32 [0,1]

    def apply(self, edges_u8: np.ndarray, alpha: float, thresh: float = 0.5) -> np.ndarray:
        """
        alpha: EMA update rate in (0,1]. Higher = follows new edges faster.
        thresh: threshold on EMA buffer to binarise.
        """
        if alpha <= 0.0:
            return edges_u8

        # Convert 0/255 -> 0/1 float
        cur = (edges_u8.astype(np.float32) / 255.0)

        if self.buf is None or self.buf.shape != cur.shape:
            self.buf = cur.copy()
        else:
            # buf = (1-alpha)*buf + alpha*cur
            cv2.accumulateWeighted(cur, self.buf, float(alpha))

        out = (self.buf >= float(thresh)).astype(np.uint8) * 255
        return out


def pick_device(device_arg: str) -> DeviceConfig:
    device_arg = device_arg.lower().strip()
    if device_arg == "cpu":
        return DeviceConfig(torch.device("cpu"), use_amp=False)

    if device_arg in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return DeviceConfig(torch.device("cuda"), use_amp=True)

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false.")
        return DeviceConfig(torch.device("mps"), use_amp=False)

    # auto
    if torch.cuda.is_available():
        return DeviceConfig(torch.device("cuda"), use_amp=True)
    if torch.backends.mps.is_available():
        return DeviceConfig(torch.device("mps"), use_amp=False)
    return DeviceConfig(torch.device("cpu"), use_amp=False)


def robust_normalise(x: np.ndarray, lo: float = 2.0, hi: float = 98.0, eps: float = 1e-6) -> np.ndarray:
    """Normalise x into [0, 1] using robust percentiles, ignoring NaNs."""
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)

    vmin = np.percentile(x[finite], lo)
    vmax = np.percentile(x[finite], hi)
    if vmax - vmin < eps:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - vmin) / (vmax - vmin)
    return np.clip(y, 0.0, 1.0)


def apply_knee_mapping_strategy_b(
    u: np.ndarray,
    near_frac: float,
    far_gamma: float = 2.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Strategy B: global, monotonic piecewise mapping with a knee chosen from a quantile.

    The mapping guarantees that at least `near_frac` of pixels (the nearest ones in `u`) land in the
    upper half of the output range [0.5, 1.0], without introducing spatial inconsistency.

    Parameters
    ----------
    u:
        Input values in [0, 1], where 0 is far and 1 is near (typically normalised inverse depth).
    near_frac:
        Fraction in (0, 1). If 0, the input is returned unchanged.
    far_gamma:
        Exponent (>1) applied to the far segment to compress it.
    eps:
        Numerical stability constant.

    Returns
    -------
    np.ndarray:
        Mapped values in [0, 1].
    """
    near_frac = float(near_frac)
    if near_frac <= 0.0:
        return u

    near_frac = min(max(near_frac, eps), 1.0 - eps)
    finite = np.isfinite(u)
    if not np.any(finite):
        return u

    # Knee t: the (1 - near_frac) quantile, so near_frac of pixels satisfy u >= t.
    t = float(np.quantile(u[finite], 1.0 - near_frac))
    t = min(max(t, eps), 1.0 - eps)

    # Far segment: u in [0, t] -> [0, 0.5] (compressive).
    x = np.clip(u / t, 0.0, 1.0)
    far = 0.5 * np.power(x, float(far_gamma))

    # Near segment: u in [t, 1] -> [0.5, 1.0] (linear to preserve near detail).
    y = np.clip((u - t) / (1.0 - t), 0.0, 1.0)
    near = 0.5 + 0.5 * y

    return np.where(u < t, far, near).astype(np.float32)


def compute_depth_anything_v2(
    bgr: np.ndarray,
    image_processor: AutoImageProcessor,
    model: AutoModelForDepthEstimation,
    devcfg: DeviceConfig,
    infer_max_side: int,
) -> np.ndarray:
    """Return a float32 depth map (relative) resized to the input frame size (H, W)."""
    h, w = bgr.shape[:2]

    # Resize for faster inference while keeping aspect ratio.
    if infer_max_side > 0:
        scale = infer_max_side / max(h, w)
        if scale < 1.0:
            new_w = max(16, int(round(w * scale)))
            new_h = max(16, int(round(h * scale)))
            bgr_small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            bgr_small = bgr
    else:
        bgr_small = bgr

    rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    inputs = image_processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(devcfg.device) for k, v in inputs.items()}

    with torch.no_grad():
        if devcfg.use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

        predicted_depth = outputs.predicted_depth  # [B, H, W]

        # Upsample to original frame size.
        pred = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    return pred[0].detach().float().cpu().numpy()


def overlay_depth_edges_grid(
    bgr: np.ndarray,
    depth: np.ndarray,
    edge_thresh: float,
    edge_dilate_px: int,
    alpha: float,
    colormap: int,
    blur_ksize: int,
    grid_w: int,
    grid_h: int,
    knee_near_frac: float = 0.5,
    contrast_boost: bool = False,
    edge_temporal_alpha: float = 0.0,
    edge_smoother: Optional[EdgeTemporalEMA] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce:
      - output BGR image with coloured depth edges overlaid
      - edge mask (uint8 0/255)
      - the full-resolution blocky "near-biased" depth field used for colouring (float32 [0,1])

    Recolouring logic
    -----------------
    - We build norm: full-res normalised depth in [0,1].
    - We downsample norm to a grid using mean pooling via INTER_AREA.
    - We apply a 3x3 max filter to that grid (dilate), so each cell becomes max(self and neighbours).
    - We upsample back to full resolution with nearest neighbour (blocky by design).
    - Edge pixels are coloured by this upsampled field, not by the local pixel depth.
      This is intended to reduce the tendency of thick outlines to inherit background depth.
    """
    h, w = bgr.shape[:2]

    # 0) Optional smoothing of the predicted depth map, mainly to reduce speckle-induced edges.
    depth_f = depth.astype(np.float32)
    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        depth_f = cv2.GaussianBlur(depth_f, (k, k), 0)

    # 1) Robust normalisation. norm is the common "depth signal" for:
    #    - edge extraction (gradients), and
    #    - grid recolouring (downsample -> max filter -> upsample).
    norm = robust_normalise(depth_f)

    # 2) Optional Strategy B remap: redistribute dynamic range so the far tail (horizon)
    #    cannot dominate colour resolution.
    #
    # norm_near remains monotonic and global, so "same norm_near => same colour anywhere in the frame".
    norm_near = apply_knee_mapping_strategy_b(norm, knee_near_frac)

    # 3) Build the coarse grid for colouring from the *unmodified* norm_near.
    #    This keeps colour self-consistent even if we later boost local contrast for edge detection.
    grid_w = max(2, int(grid_w))
    grid_h = max(2, int(grid_h))
    grid = cv2.resize(norm_near, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32)

    # 4) Neighbour max filter on the grid (dilate on float == max).
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_max = cv2.dilate(grid, kernel3, iterations=1)

    # 5) Upsample without interpolation so each pixel inherits its cell's value.
    blocky = cv2.resize(grid_max, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    blocky = np.clip(blocky, 0.0, 1.0)

    # 6) Optional CLAHE for edge detection only.
    #    CLAHE operates on 8-bit images. We convert norm_near -> uint8 -> CLAHE -> float32 in [0,1].
    edge_src = norm_near
    if contrast_boost:
        edge_u8 = (np.clip(norm_near, 0.0, 1.0) * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        edge_u8 = clahe.apply(edge_u8)
        edge_src = edge_u8.astype(np.float32) / 255.0

    # 7) Edge extraction from full-resolution (optionally contrast-boosted) depth.
    gx = cv2.Sobel(edge_src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(edge_src, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_n = robust_normalise(mag, lo=5.0, hi=99.5)
    edges = (mag_n >= float(edge_thresh)).astype(np.uint8) * 255

    if edge_dilate_px > 0:
        k = 2 * int(edge_dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.dilate(edges, kernel, iterations=1)

    if edge_smoother is not None and edge_temporal_alpha > 0.0:
        edges = edge_smoother.apply(edges, edge_temporal_alpha)

    # 8) Max filter the grid over a 3x3 neighbourhood.
    #    OpenCV dilation on float images is a max operation.
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_max = cv2.dilate(grid, kernel3, iterations=1)

    # 9) Upsample without interpolation, so each pixel inherits its cell's near-biased value.
    blocky = cv2.resize(grid_max, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    blocky = np.clip(blocky, 0.0, 1.0)

    # 10) Colourise the blocky depth field, then mask to edges only (blue = far, red/yellow = near)
    depth_u8 = (blocky * 255.0).astype(np.uint8)
    colour = cv2.applyColorMap(depth_u8, colormap)

    mask3 = cv2.merge([edges, edges, edges])
    edge_colour = cv2.bitwise_and(colour, mask3)

    # 11) Alpha blend onto the source frame only where edges exist.
    out = bgr.copy()
    if alpha > 0.0:
        edge_float = edge_colour.astype(np.float32) / 255.0
        base_float = out.astype(np.float32) / 255.0
        m = (edges.astype(np.float32) / 255.0)[:, :, None]
        blended = base_float * (1.0 - float(alpha) * m) + edge_float * (float(alpha) * m)
        out = (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)

    return out, edges, blocky


def apply_zoom(frame: np.ndarray, zoom: float) -> np.ndarray:
    """
    Resize a frame by `zoom`.

    - zoom == 1.0: returns the input frame unchanged.
    - zoom < 1.0: downscales using INTER_AREA (better for shrinking).
    - zoom > 1.0: upscales using INTER_LINEAR (reasonable default for enlarging).

    Parameters
    ----------
    frame:
        BGR image as a numpy array (H, W, 3).
    zoom:
        Scale factor (> 0). For example 0.5 halves both dimensions; 2.0 doubles them.

    Returns
    -------
    np.ndarray:
        The resized frame.
    """
    zoom = float(zoom)
    if zoom == 1.0:
        return frame

    h, w = frame.shape[:2]
    new_w = max(1, int(round(w * zoom)))
    new_h = max(1, int(round(h * zoom)))

    interp = cv2.INTER_AREA if zoom < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def main() -> int:
    ap = argparse.ArgumentParser(description="Depth Anything V2 depth-edge overlay with grid-based edge recolouring.")
    ap.add_argument("-a", "--alpha", type=float, default=0.85, help="Overlay alpha in [0,1].")
    ap.add_argument("-b", "--blur", type=int, default=5, help="Gaussian blur kernel size (odd). 0 or 1 disables.")
    ap.add_argument("-c", "--contrast-boost", action="store_true", help="Apply CLAHE to the normalised depth map for edge detection only.")
    ap.add_argument("-d", "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device.")
    ap.add_argument("-e", "--edge-thresh", type=float, default=0.25, help="Edge threshold in [0,1] after normalisation. Lower gives more edges.")
    ap.add_argument("-g", "--grid", type=int, default=16, help="Grid size for recolouring (N means NxN).")
    ap.add_argument("-gh", "--grid-h", type=int, default=0, help="Optional grid height override (cells).")
    ap.add_argument("-gw", "--grid-w", type=int, default=0, help="Optional grid width override (cells).")
    ap.add_argument("-i", "--infer-max-side", type=int, default=512, help="Resize for inference so max(H, W) <= this. 0 disables.")
    ap.add_argument("-k", "--knee-near-frac", type=float, default=0.5, help="Ensure at least this fraction of pixels maps into the nearest half of the colour range. 0 disables.")
    ap.add_argument("-l", "--limit-fps", type=float, default=0.0, help="If > 0, sleeps to avoid exceeding this FPS.")
    ap.add_argument("-m", "--model", default="depth-anything/Depth-Anything-V2-Small-hf", help="Hugging Face model id.")
    ap.add_argument("-o", "--out", default="", help="Optional output video path (e.g. out.mp4).")
    ap.add_argument("-r", "--dilate", type=int, default=2, help="Edge thickening radius in pixels.")
    ap.add_argument("-s", "--source", default="0", help="Webcam index (e.g. 0) or path to a video file.")
    ap.add_argument("-t", "--edge-temporal", type=float, default=0.0, help="Temporal smoothing for the edge mask using EMA (0 disables, typical 0.1..0.3). Higher is faster, less smoothing.")
    ap.add_argument("-w", "--show-debug", action="store_true", help="Show debug windows (edge mask and blocky depth field).")
    ap.add_argument("-z", "--zoom", type=float, default=1.0, help="Zoom factor applied to input frames after capture. 1.0 disables.")
    args = ap.parse_args()

    if not np.isfinite(args.zoom) or args.zoom <= 0.0:
        raise ValueError(f"--zoom must be a finite float > 0, got {args.zoom!r}")

    if not (0.0 <= args.edge_temporal <= 1.0):
        raise ValueError("--edge-temporal must be in [0,1]")

    devcfg = pick_device(args.device)

    print(f"[info] Loading model: {args.model}")
    image_processor = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(args.model)
    model.to(devcfg.device)
    model.eval()

    edge_smoother = EdgeTemporalEMA()

    # OpenCV video source.
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    # Output writer (optional).
    writer: Optional[cv2.VideoWriter] = None
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    if not out_fps or out_fps <= 0:
        out_fps = 30.0

    grid_w = int(args.grid_w) if int(args.grid_w) > 0 else int(args.grid)
    grid_h = int(args.grid_h) if int(args.grid_h) > 0 else int(args.grid)

    pause = False
    raw = None

    while True:
        if raw is None or not pause:
            ok, raw = cap.read()
            if not ok or raw is None:
                break

        t0 = time.time()

        frame = apply_zoom(raw, args.zoom)

        depth = compute_depth_anything_v2(
            frame,
            image_processor=image_processor,
            model=model,
            devcfg=devcfg,
            infer_max_side=int(args.infer_max_side),
        )

        out, edges, blocky = overlay_depth_edges_grid(
            frame,
            depth,
            edge_thresh=float(args.edge_thresh),
            edge_dilate_px=int(args.dilate),
            alpha=float(args.alpha),
            colormap=cv2.COLORMAP_TURBO,
            blur_ksize=int(args.blur),
            grid_w=grid_w,
            grid_h=grid_h,
            knee_near_frac=float(args.knee_near_frac),
            contrast_boost=bool(args.contrast_boost),
            edge_temporal_alpha=float(args.edge_temporal),
            edge_smoother=edge_smoother,
        )

        if args.out and writer is None:
            h, w = out.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, out_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open output writer: {args.out}")

        if writer is not None:
            writer.write(out)

        cv2.imshow("Depth edges (grid recolour)", out)

        if args.show_debug:
            cv2.imshow("Edge mask", edges)
            depth_u8 = (blocky * 255.0).astype(np.uint8)
            colour = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            cv2.imshow("Blocky near-biased inverse depth", colour)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            pause = not pause

        if args.limit_fps and float(args.limit_fps) > 0:
            dt = time.time() - t0
            target = 1.0 / float(args.limit_fps)
            if dt < target:
                time.sleep(target - dt)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
