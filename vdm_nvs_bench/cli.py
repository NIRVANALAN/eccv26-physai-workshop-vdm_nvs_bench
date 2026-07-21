"""vdm-nvs-bench — evaluate VDM novel-view camera-control submissions.

Two tracks:
  davis  (no target GT): camera(VGGT vs requested traj) + video(FVD-vs-source, CLIP-F/T) + vbench
  syn4d  (paired GT):     camera(VGGT vs GT) + video(paired FVD, CLIP-V) + paired(PSNR/SSIM/LPIPS) + vbench

Usage:
  vdm-nvs-bench eval --track davis --pred <root> --cameras <root> [--source <root>] \
      [--prompts prompts.json] [--pairs pairs.csv] --out results/
  vdm-nvs-bench eval --track syn4d --pred <root> --gt <root> --cameras <root> \
      --source <root> --out results/
Optional: --only camera,video,paired,vbench  --clip_model ViT-B-32  --checkpoint <ckpt>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.contracts import build_samples, discover_pairs, load_prompts, read_pairs_csv

DEFAULT_COMPONENTS = {
    "davis": ("camera", "video", "vbench"),
    "syn4d": ("camera", "video", "paired", "vbench"),
}


def _pairs_to_csv_list(samples):
    return [(s["seq"], s["traj"]) for s in samples]


def _validate_strict_submission(samples: list[dict], expected_pairs: int, expected_frames: int) -> None:
    """Validate the public Kaggle video contract before expensive metrics run."""
    if len(samples) != expected_pairs:
        raise SystemExit(
            f"Strict submission requires every official pair: resolved {len(samples)}/{expected_pairs} pred.mp4 files."
        )
    import cv2

    invalid = []
    for sample in samples:
        cap = cv2.VideoCapture(str(sample["pred"]))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        if count != expected_frames:
            invalid.append(f"{sample['seq']}/{sample['traj']} ({count} frames)")
    if invalid:
        detail = ", ".join(invalid[:8])
        suffix = " ..." if len(invalid) > 8 else ""
        raise SystemExit(
            f"Strict submission requires exactly {expected_frames} frames in every pred.mp4: {detail}{suffix}"
        )


def cmd_eval(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.strict_submission and not args.pairs:
        raise SystemExit("--strict_submission requires --pairs pointing to the official test_pairs.csv.")
    if args.strict_submission and args.num_frames != 49:
        raise SystemExit("--strict_submission uses the official 49-frame NVS contract; pass --num_frames 49.")
    if args.strict_submission and (args.target_w != 512 or args.target_h != 288):
        raise SystemExit("--strict_submission uses the official 512x288 NVS canvas; pass --target_w 512 --target_h 288.")
    pairs = read_pairs_csv(Path(args.pairs)) if args.pairs else discover_pairs(Path(args.pred))
    prompts = load_prompts(Path(args.prompts) if args.prompts else None)
    samples = build_samples(
        pairs, Path(args.pred),
        source_root=Path(args.source) if args.source else None,
        gt_root=Path(args.gt) if args.gt else None,
        prompts=prompts,
        strict_pred_name=args.strict_submission,
    )
    if not samples:
        raise SystemExit("No samples resolved — check --pred layout and --pairs.")
    if args.strict_submission:
        _validate_strict_submission(samples, expected_pairs=len(pairs), expected_frames=args.num_frames)
    print(f"[vdm-nvs-bench] track={args.track}  pairs={len(samples)}")

    components = args.only.split(",") if args.only else list(DEFAULT_COMPONENTS[args.track])
    results: dict = {"track": args.track, "num_pairs": len(samples), "components": components}

    if "camera" in components:
        if not args.cameras:
            raise SystemExit("--cameras is required for the camera component.")
        from .camera.eval_camera import run_camera_eval
        from .weights import resolve_vggt_checkpoint

        ckpt = Path(args.checkpoint) if args.checkpoint else resolve_vggt_checkpoint()
        results["camera"] = run_camera_eval(
            pairs=_pairs_to_csv_list(samples),
            pred_root=Path(args.pred), cameras_root=Path(args.cameras),
            cache_root=out / "_cam_cache", checkpoint=ckpt,
            output_json=out / "camera_metrics.json",
            num_frames=args.num_frames, target_w=args.target_w, target_h=args.target_h,
            gpu=args.gpu,
            pred_filenames="pred.mp4" if args.strict_submission else None,
        )

    if "video" in components:
        from .video.eval_video import run_video_eval

        results["video"] = run_video_eval(
            samples, track=args.track, clip_model=args.clip_model, fvd_size=args.fvd_size,
        )
        (out / "video_metrics.json").write_text(json.dumps(results["video"], indent=2))

    if "paired" in components:
        if args.track != "syn4d":
            print("[vdm-nvs-bench] skipping paired metrics (only meaningful for syn4d)")
        else:
            from .video.paired import run_paired_eval

            results["paired"] = run_paired_eval(
                samples,
                height=args.target_h,
                width=args.target_w,
                num_frames=args.num_frames,
            )
            (out / "paired_metrics.json").write_text(json.dumps(results["paired"], indent=2))

    if "vbench" in components:
        from .vbench_eval.run_vbench import run_vbench_eval

        try:
            results["vbench"] = run_vbench_eval(samples, out_dir=out / "vbench")
        except Exception as exc:  # noqa: BLE001 — vbench is an optional extra
            print(f"[vdm-nvs-bench] vbench skipped: {exc}")
            results["vbench"] = {"error": str(exc)}

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    # `leaderboard.csv` is the challenge-facing artifact: one complete row per
    # submission that the organizer can concatenate and rank by PSNR. Keep the
    # historical TSV name as a compatibility alias for existing scripts.
    _write_leaderboard_row(out / "leaderboard.csv", args.track, results, delimiter=",")
    _write_leaderboard_row(out / "leaderboard.tsv", args.track, results, delimiter="\t")
    print(f"[vdm-nvs-bench] wrote {out/'summary.json'}")


def _write_leaderboard_row(path: Path, track: str, results: dict, delimiter: str = ",") -> None:
    def g(d, *keys):
        for k in keys:
            d = (d or {}).get(k) if isinstance(d, dict) else None
        return "" if d is None else (f"{d:.4f}" if isinstance(d, (int, float)) else str(d))

    cam, vid, paired = results.get("camera"), results.get("video"), results.get("paired")
    vbench = (results.get("vbench") or {}).get("vbench") or {}
    cols = {
        "track": track,
        "num_pairs": results.get("num_pairs", ""),
        "rank_metric": "psnr",
        "rank_direction": "descending",
        "ate": g(cam, "ate", "mean"),
        "rot_err": g(cam, "rot_err", "mean"),
        "trans_err": g(cam, "trans_err", "mean"),
        "fvd": g(vid, "fvd", "value"),
        "clip_f": g(vid, "clip_f", "value"),
        "clip_t": g(vid, "clip_t", "value"),
        "clip_v": g(vid, "clip_v", "value"),
        "psnr": g(paired, "psnr"),
        "ssim": g(paired, "ssim"),
        "lpips": g(paired, "lpips"),
        "vbench_aesthetic_quality": g(vbench, "aesthetic_quality"),
        "vbench_imaging_quality": g(vbench, "imaging_quality"),
        "vbench_subject_consistency": g(vbench, "subject_consistency"),
        "vbench_background_consistency": g(vbench, "background_consistency"),
        "vbench_temporal_style": g(vbench, "temporal_style"),
    }
    path.write_text(delimiter.join(cols) + "\n" + delimiter.join(str(v) for v in cols.values()) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(prog="vdm-nvs-bench")
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("eval", help="Evaluate a submission")
    e.add_argument("--track", choices=["davis", "syn4d"], required=True)
    e.add_argument("--pred", required=True, help="prediction root: <seq>/<traj>/pred.mp4")
    e.add_argument("--cameras", help="requested-trajectory GT root: <seq>/<traj>.npz (cam_c2w)")
    e.add_argument("--source", help="source video root (CLIP-V / davis flat-FVD)")
    e.add_argument("--gt", help="Syn4D paired target-view GT root: <seq>/<traj>/gt.mp4")
    e.add_argument("--prompts", help="prompts.json {seq: caption}")
    e.add_argument("--pairs", help="pairs csv (video,trajectory); default auto-discover")
    e.add_argument("--strict_submission", action="store_true",
                   help="enforce official Kaggle contract: every --pairs row has predictions/<video>/<trajectory>/pred.mp4 with exactly 49 frames")
    e.add_argument("--out", required=True)
    e.add_argument("--only", help="comma list: camera,video,paired,vbench")
    e.add_argument("--checkpoint", help="VGGT-Omega checkpoint (default: auto-download)")
    e.add_argument("--clip_model", default="hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    e.add_argument("--fvd_size", type=int, default=256)
    e.add_argument("--num_frames", type=int, default=49)
    e.add_argument("--target_w", type=int, default=512)
    e.add_argument("--target_h", type=int, default=288)
    e.add_argument("--gpu", type=int, default=0)
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
