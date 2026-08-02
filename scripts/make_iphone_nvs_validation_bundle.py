#!/usr/bin/env python3
"""Materialize the public 288x512 iPhone NVS validation bundle.

The bundle is intentionally separate from the hidden-target Syn4D Kaggle test
set.  It packages the five Shape-of-Motion iPhone sequences used for local
model development, with source LiDAR depth, refined camera trajectories and
target-view RGB/masks so a participant can run an end-to-end NVS validation.

The source iPhone benchmark root is organizer-local.  This script converts it
to the portable participant layout documented in ``docs/iphone_nvs_validation.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


SEQUENCES = ("apple", "block", "paper-windmill", "spin", "teddy")
TRAJECTORY = "src0_tgt"
NUM_FRAMES = 49
HEIGHT, WIDTH = 288, 512
FPS = 12


def _crop_resize(frames: np.ndarray, *, interpolation: int) -> np.ndarray:
    """Center-crop a ``(T,H,W[,C])`` stack to 16:9, then resize to 512x288."""
    if frames.ndim not in (3, 4):
        raise ValueError(f"expected (T,H,W[,C]), got {frames.shape}")
    h, w = frames.shape[1:3]
    target_aspect = WIDTH / HEIGHT
    source_aspect = w / h
    if source_aspect > target_aspect:
        crop_w = int(round(h * target_aspect))
        left = (w - crop_w) // 2
        cropped = frames[:, :, left:left + crop_w]
    else:
        crop_h = int(round(w / target_aspect))
        top = (h - crop_h) // 2
        cropped = frames[:, top:top + crop_h]
    return np.stack(
        [cv2.resize(frame, (WIDTH, HEIGHT), interpolation=interpolation) for frame in cropped],
        axis=0,
    )


def _intrinsics_after_crop_resize(K: np.ndarray, *, src_h: int, src_w: int) -> np.ndarray:
    """Apply the same 16:9 crop + resize transform as ``_crop_resize`` to K."""
    K = np.asarray(K, dtype=np.float32).copy()
    target_aspect = WIDTH / HEIGHT
    source_aspect = src_w / src_h
    if source_aspect > target_aspect:
        crop_w = int(round(src_h * target_aspect))
        left = (src_w - crop_w) // 2
        crop_h = src_h
        K[0, 2] -= left
    else:
        crop_h = int(round(src_w / target_aspect))
        top = (src_h - crop_h) // 2
        crop_w = src_w
        K[1, 2] -= top
    sx, sy = WIDTH / crop_w, HEIGHT / crop_h
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def _read_rgb_stack(folder: Path) -> np.ndarray:
    frames = []
    for index in range(NUM_FRAMES):
        path = folder / f"{index:03d}.png"
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def _read_mask_stack(folder: Path) -> np.ndarray:
    masks = []
    for index in range(NUM_FRAMES):
        path = folder / f"{index:03d}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        masks.append(mask)
    return np.stack(masks, axis=0)


def _write_video(path: Path, frames: np.ndarray) -> None:
    """Write a portable H.264 MP4 with RGB frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(path), frames, fps=FPS, codec="libx264", quality=8, macro_block_size=None,
    )


def _assert_video(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open generated video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if (count, height, width) != (NUM_FRAMES, HEIGHT, WIDTH):
        raise RuntimeError(
            f"{path}: expected {NUM_FRAMES} frames at {WIDTH}x{HEIGHT}; "
            f"got {count} at {width}x{height}"
        )


def _export_sequence(benchmark_root: Path, source_run: Path, out: Path, sequence: str) -> int:
    cond = benchmark_root / "conditions" / sequence
    gt = benchmark_root / "gt" / sequence
    with (cond / "metadata.json").open() as handle:
        metadata = json.load(handle)
    if int(metadata["num_frames"]) != NUM_FRAMES:
        raise ValueError(f"{sequence}: expected {NUM_FRAMES} frames")
    trajectory = f"{TRAJECTORY}{int(metadata['tgt_camera_id'])}"

    # The existing canonical source is already the center-cropped 512x288 clip.
    source_in = source_run / sequence / "source_rgb.mp4"
    source_out = out / "sources" / sequence / trajectory / "source.mp4"
    if not source_in.is_file():
        raise FileNotFoundError(source_in)
    source_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_in, source_out)
    _assert_video(source_out)

    # The ``gt_rgb.mp4`` alongside a model run is a synthetic source warp, not
    # target GT.  Rebuild the real target video from Shape-of-Motion target PNGs.
    target = _crop_resize(
        _read_rgb_stack(gt / "target_frames_2x"), interpolation=cv2.INTER_AREA,
    )
    target_out = out / "gt" / sequence / trajectory / "gt.mp4"
    _write_video(target_out, target)
    _assert_video(target_out)

    source_depth = np.stack([
        np.load(cond / "src_depth_1x" / f"{index:03d}.npy").astype(np.float32)
        for index in range(NUM_FRAMES)
    ])
    source_depth = _crop_resize(source_depth, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    depth_out = out / "source_depth" / sequence / f"{trajectory}.npz"
    depth_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        depth_out,
        depth_m=source_depth,
        height=np.int32(HEIGHT), width=np.int32(WIDTH), num_frames=np.int32(NUM_FRAMES),
    )

    covisible = _crop_resize(
        _read_mask_stack(gt / "covisible_2x"), interpolation=cv2.INTER_NEAREST,
    ) > 127
    valid = _crop_resize(
        _read_mask_stack(gt / "valid_mask_2x"), interpolation=cv2.INTER_NEAREST,
    ) > 127
    fg_dir = gt / "fg_mask_2x"
    if all((fg_dir / f"{index:03d}.png").is_file() for index in range(NUM_FRAMES)):
        foreground = _crop_resize(
            _read_mask_stack(fg_dir), interpolation=cv2.INTER_NEAREST,
        ) > 127
    else:
        # The five released iPhone validation sequences currently do not carry
        # foreground masks.  Keep the field with an explicit all-false value so
        # downstream consumers have a stable schema.
        foreground = np.zeros_like(valid, dtype=bool)
    mask_out = out / "masks" / sequence / f"{trajectory}.npz"
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mask_out,
        valid=valid, covisible=covisible, foreground=foreground,
        evaluation_mask=valid & covisible,
    )

    src_cam = np.load(cond / "src_cameras.npz")
    tgt_cam = np.load(cond / "tgt_cameras.npz")
    src_h, src_w = (int(x) for x in src_cam["src_hw"])
    tgt_h, tgt_w = (int(x) for x in tgt_cam["tgt_hw"])
    src_K = np.stack([
        _intrinsics_after_crop_resize(K, src_h=src_h, src_w=src_w)
        for K in src_cam["Ks_1x"]
    ]).astype(np.float32)
    tgt_K = np.stack([
        _intrinsics_after_crop_resize(K, src_h=tgt_h, src_w=tgt_w)
        for K in tgt_cam["Ks_1x"]
    ]).astype(np.float32)
    src_c2w_world = src_cam["c2ws"].astype(np.float32)
    tgt_c2w_world = tgt_cam["c2ws"].astype(np.float32)
    world_to_query = np.linalg.inv(src_c2w_world[0]).astype(np.float32)
    src_c2w_query = (world_to_query[None] @ src_c2w_world).astype(np.float32)
    tgt_c2w_query = (world_to_query[None] @ tgt_c2w_world).astype(np.float32)
    camera_out = out / "cameras" / sequence / f"{trajectory}.npz"
    camera_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        camera_out,
        source_K=src_K, target_K=tgt_K,
        source_c2w_world=src_c2w_world, target_c2w_world=tgt_c2w_world,
        source_w2c_world=np.linalg.inv(src_c2w_world).astype(np.float32),
        target_w2c_world=np.linalg.inv(tgt_c2w_world).astype(np.float32),
        source_c2w_query=src_c2w_query, target_c2w_query=tgt_c2w_query,
        source_w2c_query=np.linalg.inv(src_c2w_query).astype(np.float32),
        target_w2c_query=np.linalg.inv(tgt_c2w_query).astype(np.float32),
        image_hw=np.array([HEIGHT, WIDTH], dtype=np.int32),
        time_ids=src_cam["time_ids"].astype(np.int32),
        target_camera_id=np.int32(metadata["tgt_camera_id"]),
    )
    return int(metadata["tgt_camera_id"])


def _write_readme(out: Path) -> None:
    (out / "README.md").write_text("""# iPhone NVS validation bundle

This is a public **local validation** set, separate from the hidden-target
Syn4D Kaggle test set.  Every sequence is a 49-frame, 12 fps, 288x512
(height x width) source-to-target NVS pair.

For each row in `test_pairs.csv`, place a prediction at
`predictions/<sequence>/<trajectory>/pred.mp4`.  It must decode to exactly 49
frames at 512x288.  The corresponding source is at
`sources/<sequence>/<trajectory>/source.mp4`, the true target video is at
`gt/<sequence>/<trajectory>/gt.mp4`, and the requested camera trajectory is
`cameras/<sequence>/<trajectory>.npz`.

`source_depth/<sequence>/<trajectory>.npz` stores metric-LiDAR `depth_m` with
shape `(49, 288, 512)`; `depth_m > 0` is its valid mask.  `cameras/...npz` contains crop-and-resize adjusted
intrinsics (`source_K`, `target_K`) and source/target poses in both the original
COLMAP world coordinate system and the source-frame-0 query coordinate system.
Use the `*_query` poses when following the original iPhone NVS pipeline.

`masks/...npz` has boolean `(49, 288, 512)` arrays.  The DyCheck headline mask
is `evaluation_mask = valid & covisible`; `foreground` is provided for optional
foreground/background breakdowns.

Run the lightweight iPhone evaluator (full-image PSNR/SSIM plus masked PSNR):

```bash
python scripts/eval_iphone_nvs_validation.py --bundle . --pred predictions
```

For the repository's unmasked local PSNR/SSIM/LPIPS evaluator:

```bash
vdm-nvs-bench eval --track syn4d --only paired --strict_submission \\
  --pred predictions --pairs test_pairs.csv --source sources --gt gt \\
  --cameras cameras --out results/iphone_validation
```

See `docs/iphone_nvs_validation.md` in the repository for the full protocol.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True,
                        help="Existing run containing canonical <sequence>/source_rgb.mp4 clips.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite nonempty directory: {args.out} (pass --overwrite)")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence in SEQUENCES:
        target_camera_id = _export_sequence(args.benchmark_root, args.source_run, args.out, sequence)
        rows.append({
            "video": sequence,
            "trajectory": f"{TRAJECTORY}{target_camera_id}",
            "source_camera_id": 0,
            "target_camera_id": target_camera_id,
            "num_frames": NUM_FRAMES,
            "height": HEIGHT,
            "width": WIDTH,
        })
        print(f"[iphone-nvs] exported {sequence}/src0_tgt{target_camera_id}")
    with (args.out / "test_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_readme(args.out)
    print(f"[iphone-nvs] ready: {args.out}")


if __name__ == "__main__":
    main()
