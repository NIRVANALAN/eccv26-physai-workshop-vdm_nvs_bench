#!/usr/bin/env python3
"""Convert fixed-tree NVS prediction MP4s into Kaggle ``submission.csv``.

The Kaggle metric receives RGB CSV rows, not MP4 files. For every official
pair this tool center-crops/resizes the 49-frame prediction to 288x512, applies
8x8 ``cv2.INTER_AREA`` downsampling to 36x64, and retains frames 0,8,...,48.
It writes the canonical ``id,R,G,B`` rows expected by ``metric_video_psnr.py``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


HEIGHT, WIDTH = 288, 512
SPATIAL_STRIDE, TEMPORAL_STRIDE, NUM_FRAMES = 8, 8, 49
GRID_HEIGHT, GRID_WIDTH = HEIGHT // SPATIAL_STRIDE, WIDTH // SPATIAL_STRIDE
FRAME_IDS = range(0, NUM_FRAMES, TEMPORAL_STRIDE)


def _standardize(frame: np.ndarray) -> np.ndarray:
    """Match the official center-crop + Pillow bilinear standardization."""
    image = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
    source_w, source_h = image.size
    scale = max(WIDTH / source_w, HEIGHT / source_h)
    resized = image.resize((round(source_w * scale), round(source_h * scale)), Image.Resampling.BILINEAR)
    left = max(0, int(round((resized.width - WIDTH) / 2.0)))
    top = max(0, int(round((resized.height - HEIGHT) / 2.0)))
    return np.asarray(resized.crop((left, top, left + WIDTH, top + HEIGHT)), dtype=np.uint8)


def _sequence_from_video(video: str) -> str:
    try:
        variant, scene, seq_root = video.split("__", 2)
    except ValueError as exc:
        raise ValueError(f"Invalid canonical video id {video!r}; expected variant__scene__seq_root") from exc
    return f"{variant}/{scene}/{seq_root}_0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-root", type=Path, required=True,
                    help="root containing <video>/<trajectory>/pred.mp4")
    ap.add_argument("--pairs", type=Path, required=True, help="official nvs_inputs/test_pairs.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--filename", default="pred.mp4", help="prediction filename within each pair directory")
    ap.add_argument("--limit", type=int, help="convert only the first N pairs (local smoke test)")
    args = ap.parse_args()

    with args.pairs.open(newline="") as fh:
        pairs = list(csv.DictReader(fh))
    if not pairs or not {"video", "trajectory"}.issubset(pairs[0]):
        raise SystemExit("pairs CSV must have video,trajectory columns")
    if args.limit is not None:
        pairs = pairs[:args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.out.open("w", newline="", buffering=1024 * 1024) as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["id", "R", "G", "B"])
        for index, pair in enumerate(pairs, 1):
            video, trajectory = pair["video"], pair["trajectory"]
            prediction = args.pred_root / video / trajectory / args.filename
            if not prediction.is_file():
                raise FileNotFoundError(f"Missing required prediction: {prediction}")
            reader = imageio.get_reader(str(prediction))
            frames = [np.asarray(frame)[..., :3] for frame in reader]
            reader.close()
            if len(frames) != NUM_FRAMES:
                raise ValueError(f"{prediction}: expected exactly {NUM_FRAMES} frames, found {len(frames)}")
            sequence = _sequence_from_video(video)
            id_prefix = sequence.replace("/", "-")
            for frame_id in FRAME_IDS:
                grid = cv2.resize(_standardize(frames[frame_id]), (GRID_WIDTH, GRID_HEIGHT), interpolation=cv2.INTER_AREA)
                for query_id, rgb in enumerate(grid.reshape(-1, 3)):
                    writer.writerow([f"{id_prefix}-q{query_id:04d}-f{frame_id:03d}", int(rgb[0]), int(rgb[1]), int(rgb[2])])
                    rows += 1
            print(f"[{index}/{len(pairs)}] {video}/{trajectory}")
    print(f"[kaggle-nvs] wrote {rows:,} rows -> {args.out}")


if __name__ == "__main__":
    main()
