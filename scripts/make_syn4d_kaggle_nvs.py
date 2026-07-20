#!/usr/bin/env python3
"""Build the NVS challenge package from the Syn4D Kaggle/validation layout.

The 3D-tracking challenge and NVS challenge deliberately use the same Syn4D
sequences.  For each ``<variant>/<scene>/seq_*`` this materializes the NVS
contract for source view 0 and target view 1:

  <out>/sources/<sequence>/src0_tgt1/source.mp4
  <out>/cameras/<sequence>/src0_tgt1.npz
  <out>/test_pairs.csv

``--include-gt`` additionally writes the target videos under ``gt/``.  Do not
publish that directory for the hidden test set; it is for the local validation
split only.

Accepted roots are either the ``challenge_eval`` directory itself or its
parent.  The layout must be ``<variant>/<scene>/png/<seq>`` plus the matching
camera CSVs under ``ground_truth/meta_exr_csv``.  This is the layout used by
``/scratch/shared/beegfs/kelvin/Syn4D/subsets/kaggle_eval``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

# Support direct use from a checkout before `pip install -e .`, matching the
# copy-paste organizer command in the README.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _save_mp4(frames, path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), [np.asarray(f)[..., :3] for f in frames], fps=fps, quality=8)


def _resolve_root(path: Path) -> Path:
    """Accept either ``.../challenge_eval`` or a parent that contains it."""
    direct = [p for p in path.iterdir() if p.is_dir() and any((p / s / "png").is_dir() for s in p.iterdir() if s.is_dir())]
    if direct:
        return path
    nested = path / "challenge_eval"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(
        f"Cannot find <variant>/<scene>/png under {path}; pass the challenge_eval directory."
    )


def _sequence_id(variant: str, scene: str, seq_root: str) -> str:
    # Avoid path separators in the submission contract while retaining all
    # identity fields from the tracking challenge's sequence id.
    return f"{variant}__{scene}__{seq_root}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-view", type=int, default=0)
    ap.add_argument("--target-view", type=int, default=1)
    ap.add_argument("--num-frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--include-gt", action="store_true", help="write target GT (validation only)")
    ap.add_argument("--limit", type=int, help="materialize at most this many pairs (smoke test)")
    args = ap.parse_args()

    from vdm_nvs_bench.data.syn4d_loader import Camera, Syn4DEvalLoader, get_relative_pose

    root = _resolve_root(args.dataset_root)
    pairs = []
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for scene_dir in sorted(p for p in variant_dir.iterdir() if (p / "png").is_dir()):
            png_root = scene_dir / "png"
            for src_dir in sorted(p for p in png_root.iterdir() if p.is_dir() and p.name.endswith(f"_{args.source_view}")):
                seq_root = src_dir.name.rsplit("_", 1)[0]
                target_dir = png_root / f"{seq_root}_{args.target_view}"
                camera_csv = scene_dir / "ground_truth" / "meta_exr_csv" / f"{seq_root}_{args.target_view}_camera.csv"
                if target_dir.is_dir() and camera_csv.is_file():
                    pairs.append((variant_dir.name, scene_dir.name, seq_root))

    if args.limit is not None:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit("No source/target view pairs found. Does this root include the private validation views?")

    args.out.mkdir(parents=True, exist_ok=True)
    trajectory = f"src{args.source_view}_tgt{args.target_view}"
    rows = []
    for index, (variant, scene, seq_root) in enumerate(pairs, 1):
        # The loader treats its dataset root as the variant directory.  Captions
        # are optional here, so it deterministically takes frames 0..num_frames-1.
        loader = Syn4DEvalLoader(
            dataset_root=root / variant, scene_name=scene, seq_root=seq_root,
            num_frames=args.num_frames, height=args.height, width=args.width,
            caption_chunk_size=args.num_frames,
        )
        frame_ids = list(range(args.num_frames))
        sequence = _sequence_id(variant, scene, seq_root)
        source = args.out / "sources" / sequence / trajectory / "source.mp4"
        _save_mp4(loader.load_video_frames(args.source_view, frame_ids), source, args.fps)

        source0 = Camera(loader.load_pose(args.source_view, frame_ids[0]))
        c2w = np.stack([
            get_relative_pose([source0, Camera(loader.load_pose(args.target_view, frame_id))])[1]
            for frame_id in frame_ids
        ]).astype(np.float32)
        camera = args.out / "cameras" / sequence / f"{trajectory}.npz"
        camera.parent.mkdir(parents=True, exist_ok=True)
        np.savez(camera, cam_c2w=c2w)

        if args.include_gt:
            _save_mp4(
                loader.load_video_frames(args.target_view, frame_ids),
                args.out / "gt" / sequence / trajectory / "gt.mp4", args.fps,
            )
        rows.append({
            "video": sequence,
            "trajectory": trajectory,
            "variant": variant,
            "scene": scene,
            "seq_root": seq_root,
            "source_view": args.source_view,
            "target_view": args.target_view,
            "num_frames": args.num_frames,
        })
        print(f"[{index}/{len(pairs)}] {sequence}/{trajectory}")

    with open(args.out / "test_pairs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[syn4d-kaggle-nvs] wrote {len(rows)} pairs -> {args.out}")
    print("  publish: sources/, cameras/, test_pairs.csv")
    if args.include_gt:
        print("  NOTE: gt/ was written for validation; keep it private for the hidden test.")


if __name__ == "__main__":
    main()
