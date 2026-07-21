"""Paired per-pixel/perceptual metrics for the Syn4D track: PSNR / SSIM / LPIPS.

Only applicable where a paired target-view GT video exists (Syn4D held-out).
Reuses the vendored `common_metrics/calculate_{psnr,ssim,lpips}.py` (each expects
two (B,T,C,H,W) tensors in [0,1] and returns {"value": [scalar]} with only_final).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

from .eval_video import _COMMON_METRICS, load_video


def _ensure_import() -> None:
    if str(_COMMON_METRICS) not in sys.path:
        sys.path.insert(0, str(_COMMON_METRICS))


def run_paired_eval(
    samples: List[dict],
    device: Optional[torch.device] = None,
    height: int = 288,
    width: int = 512,
    num_frames: int = 49,
    spatial_stride: int = 8,
    temporal_stride: int = 8,
) -> dict:
    """PSNR/SSIM/LPIPS over (pred, gt) pairs. Samples need a ``gt`` path; those
    without are skipped. For the official Syn4D NVS contract, both videos are
    standardized to ``height`` × ``width`` over their first ``num_frames``
    frames, area-downsampled by ``spatial_stride``, then temporally sampled
    every ``temporal_stride`` frames before comparison."""
    if height % spatial_stride or width % spatial_stride:
        raise ValueError(
            f"evaluation canvas {height}x{width} is not divisible by spatial_stride={spatial_stride}"
        )
    eval_height, eval_width = height // spatial_stride, width // spatial_stride
    expected_frames = len(range(0, num_frames, temporal_stride))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_import()
    from calculate_lpips import calculate_lpips
    from calculate_psnr import calculate_psnr
    from calculate_ssim import calculate_ssim

    preds, gts = [], []
    for s in samples:
        if not s.get("gt"):
            continue
        p = load_video(Path(s["pred"]), size=(height, width), max_frames=num_frames)
        g = load_video(Path(s["gt"]), size=(height, width), max_frames=num_frames)
        if p.shape[0] != num_frames or g.shape[0] != num_frames:
            raise ValueError(
                f"{s['video']}/{s['trajectory']}: PSNR requires exactly "
                f"{num_frames} decoded frames; got prediction={p.shape[0]}, gt={g.shape[0]}"
            )
        p = torch.nn.functional.interpolate(p[::temporal_stride], size=(eval_height, eval_width), mode="area")
        g = torch.nn.functional.interpolate(g[::temporal_stride], size=(eval_height, eval_width), mode="area")
        preds.append(p)
        gts.append(g)

    if not preds:
        return {"psnr": None, "ssim": None, "lpips": None, "count": 0}

    v1 = torch.stack(preds, dim=0)  # (N,T,C,H,W) in [0,1]
    v2 = torch.stack(gts, dim=0)

    psnr = calculate_psnr(v1, v2, only_final=True)["value"][0]
    ssim = calculate_ssim(v1, v2, only_final=True)["value"][0]
    lpips_v = calculate_lpips(v1, v2, device, only_final=True)["value"][0]
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "lpips": float(lpips_v),
        "count": v1.shape[0],
        "num_frames": int(expected_frames),
        "height": int(eval_height),
        "width": int(eval_width),
        "source_num_frames": int(num_frames),
        "source_height": int(height),
        "source_width": int(width),
        "spatial_stride": int(spatial_stride),
        "temporal_stride": int(temporal_stride),
    }
