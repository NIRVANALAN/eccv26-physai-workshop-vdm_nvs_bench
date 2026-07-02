#!/usr/bin/env python3
"""Materialize the Syn4D held-out challenge package (source + GT target + camera GT).

Self-contained (uses vdm_nvs_bench.data.syn4d_loader — no recammaster/diffsynth deps).
Emits the vdm-nvs-bench submission-contract GT for a reserved Syn4D scene:

  <out>/sources/<seq>/<traj>/source.mp4   # source/input video
  <out>/gt/<seq>/<traj>/gt.mp4            # paired target-view GT
  <out>/cameras/<seq>/<traj>.npz          # key cam_c2w (T,4,4) = requested trajectory

  seq  = "<scene>_<seq_root>";  traj = "src<source_view>_tgt<target_view>"

Held-out eval scenes (recammaster convention): train_group, flying_group (source
view 0, target views 1..7). Example:
  python scripts/make_syn4d_heldout.py --dataset_root /scratch/shared/beegfs/zeren/Syn4D \
      --scene flying_group --seq_root seq_000000 --source_view 0 --target_views 1,2,3 \
      --num_frames 49 --caption_chunk_size 81 --out heldout/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _save_mp4(frames, path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), [np.asarray(f)[..., :3] for f in frames], fps=fps, quality=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seq_root", required=True)
    ap.add_argument("--source_view", type=int, default=0)
    ap.add_argument("--target_views", default="1,2,3", help="comma list")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--frame_offset_in_chunk", type=int, default=0)
    ap.add_argument("--caption_chunk_size", type=int, default=81)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    from vdm_nvs_bench.data.syn4d_loader import Camera, Syn4DEvalLoader, get_relative_pose

    target_views = [int(v) for v in args.target_views.split(",") if v.strip()]
    loader = Syn4DEvalLoader(
        dataset_root=args.dataset_root, scene_name=args.scene, seq_root=args.seq_root,
        num_frames=args.num_frames, height=args.height, width=args.width,
        frame_offset_in_chunk=args.frame_offset_in_chunk, caption_chunk_size=args.caption_chunk_size,
    )
    chunk_tag, frame_ids = loader.choose_chunk_and_frames(args.source_view, target_views)
    seq = f"{args.scene}_{args.seq_root}"
    print(f"[syn4d] {seq}: chunk={chunk_tag} frames={len(frame_ids)} targets={target_views}")

    source_frames = loader.load_video_frames(args.source_view, frame_ids)
    cond_cam = Camera(loader.load_pose(args.source_view, frame_ids[0]))

    for tv in target_views:
        traj = f"src{args.source_view}_tgt{tv}"
        _save_mp4(source_frames, args.out / "sources" / seq / traj / "source.mp4", args.fps)
        _save_mp4(loader.load_video_frames(tv, frame_ids), args.out / "gt" / seq / traj / "gt.mp4", args.fps)

        c2ws = [get_relative_pose([cond_cam, Camera(loader.load_pose(tv, fid))])[1] for fid in frame_ids]
        cam_npz = args.out / "cameras" / seq / f"{traj}.npz"
        cam_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cam_npz, cam_c2w=np.stack(c2ws).astype(np.float32))
        print(f"  wrote {traj}: source.mp4 + gt.mp4 + cameras/{traj}.npz")

    print(f"[syn4d] done -> {args.out}")


if __name__ == "__main__":
    main()
