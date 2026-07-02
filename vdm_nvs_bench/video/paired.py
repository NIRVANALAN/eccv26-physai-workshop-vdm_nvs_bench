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
    size: int = 256,
) -> dict:
    """PSNR/SSIM/LPIPS over (pred, gt) pairs. Samples need a ``gt`` path; those
    without are skipped. All pairs are resized to a common (size,size) and the
    common min frame count, then scored as one batch."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_import()
    from calculate_lpips import calculate_lpips
    from calculate_psnr import calculate_psnr
    from calculate_ssim import calculate_ssim

    preds, gts = [], []
    for s in samples:
        if not s.get("gt"):
            continue
        p = load_video(Path(s["pred"]), size=(size, size))
        g = load_video(Path(s["gt"]), size=(size, size))
        n = min(p.shape[0], g.shape[0])
        preds.append(p[:n])
        gts.append(g[:n])

    if not preds:
        return {"psnr": None, "ssim": None, "lpips": None, "count": 0}

    T = min(v.shape[0] for v in preds)
    v1 = torch.stack([v[:T] for v in preds], dim=0)  # (N,T,C,H,W) in [0,1]
    v2 = torch.stack([v[:T] for v in gts], dim=0)

    psnr = calculate_psnr(v1, v2, only_final=True)["value"][0]
    ssim = calculate_ssim(v1, v2, only_final=True)["value"][0]
    lpips_v = calculate_lpips(v1, v2, device, only_final=True)["value"][0]
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "lpips": float(lpips_v),
        "count": v1.shape[0],
        "num_frames": int(T),
    }
