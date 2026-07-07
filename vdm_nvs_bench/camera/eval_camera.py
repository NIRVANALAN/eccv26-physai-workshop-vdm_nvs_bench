"""VGGT-Omega camera-trajectory evaluation (self-contained).

pred.mp4 -> standardized mp4 -> frames -> VGGT-Omega -> predicted c2w
        -> Sim(3)-aligned ATE/RPE vs the requested target trajectory (GT).

The camera pose estimator is the OFFICIAL vggt_omega package
(https://github.com/facebookresearch/vggt-omega). It is NOT vendored here — install it
(`pip install -e vggt-omega`) or point $VGGT_OMEGA_REPO at a checkout. See the README
"Setup" section.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .pose_metrics import (
    compute_pose_metrics,
    load_gt_c2w,
    resolve_pred_mp4,
    standardize_pred_mp4,
)

DEFAULT_PRED_FILENAMES = "pred.mp4,pred_rgb.mp4,{traj}.mp4,gen.mp4,output.mp4,render.mp4,color_{traj}.mp4"

_VGGT_SETUP_HINT = (
    "vggt_omega (camera pose estimator) not found. Set up the OFFICIAL repo:\n"
    "  git clone https://github.com/facebookresearch/vggt-omega\n"
    "  pip install -e vggt-omega\n"
    "or point $VGGT_OMEGA_REPO at the checkout. See the README 'Setup' section."
)


def _ensure_vggt_omega_import() -> None:
    """Make the OFFICIAL vggt_omega package importable: a pip install first, else a
    checkout pointed to by $VGGT_OMEGA_REPO. Raises with setup instructions otherwise."""
    try:
        import vggt_omega  # noqa: F401
        return
    except ImportError:
        pass
    repo = os.environ.get("VGGT_OMEGA_REPO", "").strip()
    if repo and (Path(repo) / "vggt_omega").is_dir():
        if repo not in sys.path:
            sys.path.insert(0, repo)
        return
    raise ImportError(_VGGT_SETUP_HINT)


def _extract_frames(mp4_path: Path, frames_dir: Path, num_frames: int) -> List[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("frame_*.png"))
    if len(existing) >= num_frames and all(
        p.stat().st_mtime >= mp4_path.stat().st_mtime for p in existing[:num_frames]
    ):
        return existing[:num_frames]
    for p in frames_dir.glob("frame_*.png"):
        p.unlink()

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open standardized mp4: {mp4_path}")
    paths: List[Path] = []
    idx = 0
    while idx < num_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        out = frames_dir / f"frame_{idx:05d}.png"
        if not cv2.imwrite(str(out), frame_bgr):
            raise RuntimeError(f"failed to write extracted frame: {out}")
        paths.append(out)
        idx += 1
    cap.release()
    if not paths:
        raise RuntimeError(f"no frames extracted from {mp4_path}")
    while len(paths) < num_frames:  # pad with last frame to keep timeline length stable
        src = paths[-1]
        dst = frames_dir / f"frame_{len(paths):05d}.png"
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None or not cv2.imwrite(str(dst), img):
            raise RuntimeError(f"failed to pad frame {dst} from {src}")
        paths.append(dst)
    return paths


def load_model(checkpoint_path: Path, device, enable_alignment: bool, enable_depth: bool):
    import torch

    _ensure_vggt_omega_import()
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(enable_depth=enable_depth, enable_alignment=enable_alignment).to(device).eval()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=enable_depth)
    if missing or unexpected:
        print(f"[vggt-omega] checkpoint load: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def run_vggt_omega_inference(
    pred_mp4: Path, out_dir: Path, model, device, num_frames: int,
    target_w: int, target_h: int, image_resolution: int, image_mode: str, force: bool,
) -> Tuple[Path, bool]:
    import torch

    _ensure_vggt_omega_import()
    from vggt_omega.utils.geometry import closed_form_inverse_se3
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    out_dir.mkdir(parents=True, exist_ok=True)
    std_mp4 = out_dir / f"_input_{target_h}x{target_w}x{num_frames}.mp4"
    standardize_pred_mp4(pred_mp4, std_mp4, target_w, target_h, num_frames)

    pose_npz = out_dir / "pose" / "vggt_omega.npz"
    if not force and pose_npz.is_file() and pose_npz.stat().st_mtime >= std_mp4.stat().st_mtime:
        return pose_npz, True

    frame_paths = _extract_frames(std_mp4, out_dir / "frames", num_frames)
    images = load_and_preprocess_images(
        [str(p) for p in frame_paths], mode=image_mode, image_resolution=image_resolution
    ).to(device)
    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
    extr = extrinsics[0] if extrinsics.ndim == 4 else extrinsics
    extr = extr.detach().float().cpu()
    c2w = closed_form_inverse_se3(extr).detach().cpu().numpy().astype(np.float32)

    pose_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pose_npz, data=c2w, cam_c2w=c2w)
    return pose_npz, False


def _stats(rows: list, key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"mean": None, "std": None, "median": None, "n": 0}
    v = np.asarray(vals, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std()),
            "median": float(np.median(v)), "n": int(len(vals))}


def run_camera_eval(
    pairs: List[Tuple[str, str]],
    pred_root: Path,
    cameras_root: Path,
    cache_root: Path,
    checkpoint: Path,
    output_json: Optional[Path] = None,
    pred_filenames: str = DEFAULT_PRED_FILENAMES,
    num_frames: int = 49,
    target_w: int = 512,
    target_h: int = 288,
    image_resolution: int = 512,
    image_mode: str = "balanced",
    enable_alignment: bool = False,
    enable_depth: bool = True,
    gpu: int = 0,
    force_pose: bool = False,
) -> dict:
    """Run VGGT-Omega camera eval over ``pairs`` and return the summary dict."""
    import torch

    templates = [s.strip() for s in pred_filenames.split(",") if s.strip()]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VGGT-Omega camera eval requires CUDA")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")

    model = load_model(Path(checkpoint), device, enable_alignment, enable_depth)

    per_pair: List[dict] = []
    for i, (seq, traj) in enumerate(pairs):
        pred_mp4 = resolve_pred_mp4(pred_root, seq, traj, templates)
        cam_npz = cameras_root / seq / f"{traj}.npz"
        if pred_mp4 is None:
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (pred mp4 missing)")
            continue
        if not cam_npz.is_file():
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (gt camera npz missing)")
            continue
        try:
            pose_npz, cached = run_vggt_omega_inference(
                pred_mp4, cache_root / f"{seq}__{traj}", model, device, num_frames,
                target_w, target_h, image_resolution, image_mode, force_pose,
            )
            pred_c2w = np.load(pose_npz)["data"][:num_frames]
            gt_c2w = load_gt_c2w(cam_npz, num_frames)
            metrics = compute_pose_metrics(pred_c2w, gt_c2w)
            per_pair.append({"seq": seq, "traj": traj, **metrics})
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: ate={metrics['ate']:.4f} "
                  f"trans={metrics['trans_err']:.4f} rot={metrics['rot_err']:.3f}"
                  f"{' (cached)' if cached else ''}")
        except Exception as exc:  # noqa: BLE001 — keep going across pairs
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: FAIL ({exc})")

    sim3 = [r for r in per_pair if r.get("alignment_mode") == "umeyama_align_scale"]
    origin = [r for r in per_pair if r.get("alignment_mode") == "origin_align_only"]
    has_norm = [r for r in per_pair if r.get("ate_norm") is not None]
    summary = {
        "pred_root": str(pred_root),
        "pose_estimator": "vggt_omega",
        "num_pairs_evaluated": len(per_pair),
        "num_pairs_total": len(pairs),
        "checkpoint": str(checkpoint),
        "ate": _stats(per_pair, "ate"),
        "trans_err": _stats(per_pair, "trans_err"),
        "rot_err": _stats(per_pair, "rot_err"),
        "sim3_subset": {"ate": _stats(sim3, "ate"), "trans_err": _stats(sim3, "trans_err"),
                        "rot_err": _stats(sim3, "rot_err")},
        "origin_subset": {"ate": _stats(origin, "ate"), "trans_err": _stats(origin, "trans_err"),
                          "rot_err": _stats(origin, "rot_err")},
        "all_norm": {"ate": _stats(has_norm, "ate_norm"),
                     "trans_err": _stats(has_norm, "trans_err_norm"),
                     "rot_err": _stats(per_pair, "rot_err")},
        "per_pair": per_pair,
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2))
        print(f"[vggt-omega-cam] wrote {output_json}")
    return summary


def _read_pairs_csv(pairs_csv: Path) -> List[Tuple[str, str]]:
    import csv

    with open(pairs_csv, newline="") as fh:
        return [(r["video"], r["trajectory"]) for r in csv.DictReader(fh)]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="VGGT-Omega camera-trajectory eval")
    ap.add_argument("--pred_root", type=Path, required=True)
    ap.add_argument("--pairs_csv", type=Path, required=True)
    ap.add_argument("--cameras_root", type=Path, required=True)
    ap.add_argument("--cache_root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--pred_filenames", default=DEFAULT_PRED_FILENAMES)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--target_w", type=int, default=512)
    ap.add_argument("--target_h", type=int, default=288)
    ap.add_argument("--image_resolution", type=int, default=512)
    ap.add_argument("--image_mode", choices=["balanced", "max_size"], default="balanced")
    ap.add_argument("--enable_alignment", action="store_true")
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force_pose", action="store_true")
    args = ap.parse_args()

    run_camera_eval(
        pairs=_read_pairs_csv(args.pairs_csv),
        pred_root=args.pred_root, cameras_root=args.cameras_root, cache_root=args.cache_root,
        checkpoint=args.checkpoint, output_json=args.output_json,
        pred_filenames=args.pred_filenames, num_frames=args.num_frames,
        target_w=args.target_w, target_h=args.target_h,
        image_resolution=args.image_resolution, image_mode=args.image_mode,
        enable_alignment=args.enable_alignment, enable_depth=not args.no_depth,
        gpu=args.gpu, force_pose=args.force_pose,
    )


if __name__ == "__main__":
    main()
