#!/usr/bin/env python3
"""Build the public Track-2 source MP4 tree from released Syn4D source PNGs.

This participant-facing utility reads source-view images only. It never reads
target-view images, camera trajectories, ground truth, annotations, depth, or
other private organizer data.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


HEIGHT, WIDTH, NUM_FRAMES, FPS = 288, 512, 49, 12


def _resolve_root(path: Path) -> Path:
    if (path / "challenge_eval").is_dir():
        return path / "challenge_eval"
    return path


def _frame_paths(root: Path, video: str) -> list[Path]:
    variant, scene, seq_root = video.split("__", 2)
    png_root = root / variant / scene / "png"
    candidates = [
        (png_root / seq_root, f"{seq_root}_{{frame:04d}}.png"),
        (png_root / f"{seq_root}_0", f"{seq_root}_0_{{frame:04d}}.png"),
    ]
    for directory, pattern in candidates:
        frames = [directory / pattern.format(frame=frame) for frame in range(NUM_FRAMES)]
        if all(path.is_file() for path in frames):
            return frames

    # A compatibility fallback for equivalent public layouts with differing
    # zero-padding. It still accepts source-view files only.
    for directory, _ in candidates:
        if not directory.is_dir():
            continue
        matcher = re.compile(rf"^{re.escape(directory.name)}_(\d+)\.png$")
        indexed = {
            int(match.group(1)): path
            for path in directory.glob("*.png")
            if (match := matcher.match(path.name))
        }
        if all(frame in indexed for frame in range(NUM_FRAMES)):
            return [indexed[frame] for frame in range(NUM_FRAMES)]
    raise FileNotFoundError(f"Cannot find source frames 0..{NUM_FRAMES - 1} for {video} under {png_root}")


def _standardize(image: Image.Image) -> np.ndarray:
    """Center-crop/resize to the official 512x288 source-video canvas."""
    source_w, source_h = image.size
    scale = max(WIDTH / source_w, HEIGHT / source_h)
    resized = image.resize((round(source_w * scale), round(source_h * scale)), Image.Resampling.BILINEAR)
    left = max(0, int(round((resized.width - WIDTH) / 2.0)))
    top = max(0, int(round((resized.height - HEIGHT) / 2.0)))
    return np.asarray(resized.crop((left, top, left + WIDTH, top + HEIGHT)).convert("RGB"), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="Syn4D_Benchmark or Syn4D_Benchmark/challenge_eval")
    ap.add_argument("--pairs", type=Path, required=True, help="nvs_inputs/test_pairs.csv")
    ap.add_argument("--out", type=Path, required=True, help="output nvs_inputs/sources directory")
    ap.add_argument("--limit", type=int, help="build only the first N pairs for a smoke test")
    args = ap.parse_args()

    with args.pairs.open(newline="") as fh:
        pairs = list(csv.DictReader(fh))
    if not pairs or not {"video", "trajectory"}.issubset(pairs[0]):
        raise SystemExit("pairs CSV must contain video,trajectory columns")
    if args.limit is not None:
        pairs = pairs[:args.limit]

    root = _resolve_root(args.dataset_root)
    for index, pair in enumerate(pairs, 1):
        frames = [_standardize(Image.open(path)) for path in _frame_paths(root, pair["video"])]
        destination = args.out / pair["video"] / pair["trajectory"] / "source.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(destination), frames, fps=FPS, quality=8)
        print(f"[{index}/{len(pairs)}] {pair['video']}/{pair['trajectory']}")
    print(f"[syn4d-nvs] wrote {len(pairs)} source videos -> {args.out}")


if __name__ == "__main__":
    main()
