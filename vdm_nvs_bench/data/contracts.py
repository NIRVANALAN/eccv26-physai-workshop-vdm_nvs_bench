"""Submission / data folder contract for both tracks.

Convention (per (seq, traj) pair):
  pred_root/<seq>/<traj>/pred.mp4            # participant prediction   (required)
  cameras_root/<seq>/<traj>.npz  key cam_c2w # requested target trajectory = camera GT
  source_root/<seq>/<traj>/source.mp4        # source/input video (CLIP-V, davis flat-FVD)
  gt_root/<seq>/<traj>/gt.mp4                # Syn4D only: paired target-view GT
  prompts.json : {"<seq>": "caption", ...}   # optional (CLIP-T / VBench temporal_style)

The pair list comes from a CSV with columns `video,trajectory`, or is auto-discovered
by walking pred_root/<seq>/<traj>/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

PRED_NAMES = ("pred.mp4", "pred_rgb.mp4", "{traj}.mp4", "gen.mp4", "output.mp4", "render.mp4")


def read_pairs_csv(pairs_csv: Path) -> List[Tuple[str, str]]:
    with open(pairs_csv, newline="") as fh:
        return [(r["video"], r["trajectory"]) for r in csv.DictReader(fh)]


def discover_pairs(pred_root: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    root = pred_root / "predictions" if (pred_root / "predictions").is_dir() else pred_root
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for traj_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
            pairs.append((seq_dir.name, traj_dir.name))
    return pairs


def _resolve(root: Optional[Path], seq: str, traj: str, names) -> Optional[Path]:
    if root is None:
        return None
    pair_dir = root / seq / traj
    if not pair_dir.is_dir() and (root / "predictions" / seq / traj).is_dir():
        pair_dir = root / "predictions" / seq / traj
    for nm in names:
        cand = pair_dir / nm.format(seq=seq, traj=traj)
        if cand.is_file():
            return cand
    return None


def build_samples(
    pairs: List[Tuple[str, str]],
    pred_root: Path,
    source_root: Optional[Path] = None,
    gt_root: Optional[Path] = None,
    prompts: Optional[dict] = None,
    strict_pred_name: bool = False,
) -> List[dict]:
    """Resolve every pair into a sample dict with pred/gt/source/prompt paths."""
    prompts = prompts or {}
    pred_names = ("pred.mp4",) if strict_pred_name else PRED_NAMES
    samples: List[dict] = []
    for seq, traj in pairs:
        pred = _resolve(pred_root, seq, traj, pred_names)
        if pred is None:
            print(f"[contracts] WARN: no pred mp4 for {seq}/{traj}; skipping")
            continue
        samples.append({
            "seq": seq,
            "traj": traj,
            "pred": pred,
            "source": _resolve(source_root, seq, traj, ("source.mp4", "source_{traj}.mp4", "src.mp4")),
            "gt": _resolve(gt_root, seq, traj, ("gt.mp4", "gt_{traj}.mp4", "target.mp4")),
            "prompt": prompts.get(seq),
        })
    return samples


def load_prompts(prompts_json: Optional[Path]) -> dict:
    if prompts_json is None:
        return {}
    return json.loads(Path(prompts_json).read_text())
