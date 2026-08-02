#!/usr/bin/env python3
"""Evaluate 288x512 iPhone NVS validation predictions.

The iPhone validation bundle uses five public, paired source-to-target clips.
This lightweight evaluator reports PSNR and SSIM over all pixels, plus masked
PSNR over the DyCheck headline region ``valid & covisible``.  It deliberately
does not rank the hidden Syn4D Kaggle test set.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


NUM_FRAMES, HEIGHT, WIDTH = 49, 288, 512


def _read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    video = np.stack(frames, axis=0) if frames else np.empty((0, HEIGHT, WIDTH, 3), np.uint8)
    if video.shape != (NUM_FRAMES, HEIGHT, WIDTH, 3):
        raise ValueError(
            f"{path}: expected ({NUM_FRAMES}, {HEIGHT}, {WIDTH}, 3), got {video.shape}"
        )
    return video.astype(np.float32) / 255.0


def _psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = (pred - target) ** 2
    if mask is not None:
        if not mask.any():
            raise ValueError("evaluation mask contains no valid pixels")
        error = error[mask]
    mse = float(np.mean(error))
    return 100.0 if mse < 1e-10 else float(10.0 * np.log10(1.0 / mse))


def _ssim_frame(pred: np.ndarray, target: np.ndarray) -> float:
    """Standard 11x11 Gaussian SSIM, matching the bundled paired evaluator."""
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)
    values = []
    for channel in range(3):
        a, b = pred[..., channel].astype(np.float64), target[..., channel].astype(np.float64)
        mu_a = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
        mu_b = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
        sigma_a = cv2.filter2D(a * a, -1, window)[5:-5, 5:-5] - mu_a * mu_a
        sigma_b = cv2.filter2D(b * b, -1, window)[5:-5, 5:-5] - mu_b * mu_b
        sigma_ab = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu_a * mu_b
        numerator = (2.0 * mu_a * mu_b + 0.01 ** 2) * (2.0 * sigma_ab + 0.03 ** 2)
        denominator = (mu_a * mu_a + mu_b * mu_b + 0.01 ** 2) * (sigma_a + sigma_b + 0.03 ** 2)
        values.append(float(np.mean(numerator / denominator)))
    return float(np.mean(values))


def _pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True,
                        help="Root containing <sequence>/<trajectory>/pred.mp4.")
    parser.add_argument("--out", type=Path,
                        help="Optional JSON report path (default: <pred>/iphone_nvs_metrics.json).")
    args = parser.parse_args()

    rows = _pairs(args.bundle / "test_pairs.csv")
    per_sequence: dict[str, dict[str, float]] = {}
    for row in rows:
        sequence, trajectory = row["video"], row["trajectory"]
        pred = _read_video(args.pred / sequence / trajectory / "pred.mp4")
        target = _read_video(args.bundle / "gt" / sequence / trajectory / "gt.mp4")
        masks = np.load(args.bundle / "masks" / sequence / f"{trajectory}.npz")
        evaluation_mask = masks["evaluation_mask"].astype(bool)
        metrics = {
            "psnr": _psnr(pred, target),
            "ssim": float(np.mean([_ssim_frame(p, t) for p, t in zip(pred, target)])),
            "mpsnr": _psnr(pred, target, evaluation_mask),
            "mask_fraction": float(evaluation_mask.mean()),
        }
        per_sequence[sequence] = metrics
        print(sequence + ": " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))

    aggregate = {
        metric: float(np.mean([entry[metric] for entry in per_sequence.values()]))
        for metric in next(iter(per_sequence.values()))
    }
    report = {"protocol": "iPhone NVS 288x512x49", "per_sequence": per_sequence, "aggregate": aggregate}
    out = args.out or args.pred / "iphone_nvs_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("aggregate: " + " ".join(f"{key}={value:.4f}" for key, value in aggregate.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
