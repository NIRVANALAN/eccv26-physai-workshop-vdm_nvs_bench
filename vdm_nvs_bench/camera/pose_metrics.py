"""Camera-trajectory metrics: Sim(3)-aligned ATE + RPE-trans + RPE-rot via evo.

Extracted verbatim (logic-preserving) from the VideoX-Fun ViPE/VGGT-Omega camera
eval so the workshop repo is self-contained. The predicted camera trajectory
(recovered from the generated video by VGGT-Omega) is compared against the
requested target trajectory (the control signal = camera GT).

Public functions:
  - resolve_pred_mp4(pred_root, seq, traj, templates) -> Path | None
  - standardize_pred_mp4(src, dst, W, H, T)           -> Path   (ffmpeg)
  - load_gt_c2w(cameras_npz, num_frames)              -> (T,4,4)
  - compute_pose_metrics(pred_c2w, gt_c2w)            -> dict
"""
from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import numpy as np


# --------------------------------------------------------------------------
# Prediction resolution + standardization
# --------------------------------------------------------------------------
def resolve_pred_mp4(
    pred_root: Path, seq: str, traj: str, pred_filename_templates: List[str]
) -> Optional[Path]:
    """First existing candidate under <pred_root>/<seq>/<traj>/ (or predictions/)."""
    pair_dir = pred_root / seq / traj
    if not pair_dir.is_dir():
        pair_dir = pred_root / "predictions" / seq / traj
        if not pair_dir.is_dir():
            return None
    for tpl in pred_filename_templates:
        cand = pair_dir / tpl.format(seq=seq, traj=traj)
        if cand.is_file():
            return cand
    return None


def standardize_pred_mp4(
    src: Path, dst: Path, target_w: int, target_h: int, num_frames: int
) -> Path:
    """ffmpeg-resize (stretch) + truncate <src> -> <dst> at W×H×<=T.

    Force-stretch to a canonical (W,H) and take the first ``num_frames`` frames so
    pose recovery is comparable across submissions that emit different resolutions
    / frame counts. Sim(3)-aligned ATE absorbs the residual squash. Idempotent.
    """
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"scale={target_w}:{target_h}:flags=bicubic",
        "-frames:v", str(num_frames),
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg standardize failed for {src} -> {dst}\nstderr: {proc.stderr[:500]}"
        )
    return dst


# --------------------------------------------------------------------------
# GT camera loading + trajectory conversions
# --------------------------------------------------------------------------
def load_gt_c2w(cameras_npz: Path, num_frames: int) -> np.ndarray:
    """(T,4,4) c2w from <cameras_root>/<seq>/<traj>.npz (key ``cam_c2w`` or ``c2w``)."""
    d = np.load(cameras_npz)
    if "cam_c2w" in d.files:
        c2w = np.asarray(d["cam_c2w"], dtype=np.float64)
    elif "c2w" in d.files:
        c2w = np.asarray(d["c2w"], dtype=np.float64)
    else:
        raise ValueError(f"no cam_c2w/c2w key in {cameras_npz} (keys: {d.files})")
    return c2w[:num_frames]


def relative_c2w(c2w_mats: np.ndarray) -> np.ndarray:
    inv0 = np.linalg.inv(c2w_mats[0])
    return np.stack([inv0 @ c for c in c2w_mats], axis=0)


def c2w_to_tum(c2w_mats: np.ndarray) -> np.ndarray:
    """(T,4,4) c2w -> TUM-style (T,7) [x,y,z, qw,qx,qy,qz]."""
    from scipy.spatial.transform import Rotation

    out = []
    for c2w in c2w_mats:
        xyz = c2w[:3, 3]
        xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
        out.append(np.concatenate([xyz, wxyz], axis=0))
    return np.stack(out, axis=0)


# --------------------------------------------------------------------------
# The metric core
# --------------------------------------------------------------------------
def compute_pose_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict:
    """Sim(3)-aligned ATE + RPE-trans + RPE-rot, with origin-only fallback and a
    per-pair scale-normalized ATE/trans for rank-deficient (pan/zoom) GT."""
    from evo.core import sync
    from evo.core.geometry import GeometryException
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D
    from evo.main_ape import ape as main_ape_fn
    from evo.main_rpe import rpe as main_rpe_fn

    pred_rel = relative_c2w(pred_c2w)
    gt_rel = relative_c2w(gt_c2w)
    pred_tum = c2w_to_tum(pred_rel)
    gt_tum = c2w_to_tum(gt_rel)
    n = min(pred_tum.shape[0], gt_tum.shape[0])
    timestamps = np.arange(n, dtype=np.float64)

    pred_traj = PoseTrajectory3D(
        positions_xyz=pred_tum[:n, :3],
        orientations_quat_wxyz=pred_tum[:n, 3:],
        timestamps=timestamps,
    )
    gt_traj = PoseTrajectory3D(
        positions_xyz=gt_tum[:n, :3],
        orientations_quat_wxyz=gt_tum[:n, 3:],
        timestamps=timestamps,
    )
    gt_traj, pred_traj = sync.associate_trajectories(deepcopy(gt_traj), deepcopy(pred_traj))

    align_mode = "umeyama_align_scale"
    try:
        ate = main_ape_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True, correct_scale=True, align_origin=False,
        )
        rpe_rot = main_rpe_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
            align=True, correct_scale=True, align_origin=False,
        )
        rpe_trans = main_rpe_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.translation_part,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
            align=True, correct_scale=True, align_origin=False,
        )
    except GeometryException:
        align_mode = "origin_align_only"
        ate = main_ape_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=False, correct_scale=False, align_origin=True,
        )
        rpe_rot = main_rpe_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
            align=False, correct_scale=False, align_origin=True,
        )
        rpe_trans = main_rpe_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.translation_part,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
            align=False, correct_scale=False, align_origin=True,
        )

    # Per-pair scale-normalized ATE/trans: scale pred by ||gt_centered||/||pred_centered||
    # then origin-align. Defined whenever GT has non-zero translation spread.
    gt_pos = gt_tum[:n, :3]
    pred_pos = pred_tum[:n, :3]
    gt_centered_norm = float(np.linalg.norm(gt_pos - gt_pos.mean(axis=0, keepdims=True)))
    pred_centered_norm = float(np.linalg.norm(pred_pos - pred_pos.mean(axis=0, keepdims=True)))
    eps = 1e-6
    ate_norm = trans_err_norm = None
    if gt_centered_norm > eps and pred_centered_norm > eps:
        scale_factor = gt_centered_norm / pred_centered_norm
        pred_traj_scaled = PoseTrajectory3D(
            positions_xyz=pred_pos * scale_factor,
            orientations_quat_wxyz=pred_tum[:n, 3:],
            timestamps=timestamps,
        )
        gt_traj_2 = deepcopy(gt_traj)
        gt_traj_2, pred_traj_scaled = sync.associate_trajectories(gt_traj_2, pred_traj_scaled)
        try:
            ate_n = main_ape_fn(
                gt_traj_2, pred_traj_scaled, est_name="traj",
                pose_relation=PoseRelation.translation_part,
                align=False, correct_scale=False, align_origin=True,
            )
            rpe_t_n = main_rpe_fn(
                gt_traj_2, pred_traj_scaled, est_name="traj",
                pose_relation=PoseRelation.translation_part,
                delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
                align=False, correct_scale=False, align_origin=True,
            )
            ate_norm = float(ate_n.stats["rmse"])
            trans_err_norm = float(rpe_t_n.stats["rmse"])
        except Exception:
            pass

    return {
        "ate": float(ate.stats["rmse"]),
        "trans_err": float(rpe_trans.stats["rmse"]),
        "rot_err": float(rpe_rot.stats["rmse"]),
        "ate_norm": ate_norm,
        "trans_err_norm": trans_err_norm,
        "gt_centered_norm": gt_centered_norm,
        "pred_centered_norm": pred_centered_norm,
        "num_aligned_poses": int(gt_traj.num_poses),
        "alignment_mode": align_mode,
    }
