"""Dependency-light self-checks (no GPU, no model weights).

Run: python tests/test_smoke.py
Covers the two pieces of non-trivial logic that don't need CUDA/weights:
  1. submission-folder resolution (data.contracts)
  2. the evo pose-metric core (camera.pose_metrics) on synthetic trajectories
"""
import tempfile
from pathlib import Path

import numpy as np

from vdm_nvs_bench.data.contracts import build_samples, discover_pairs
from vdm_nvs_bench.camera.pose_metrics import compute_pose_metrics


def test_contracts_resolution():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for seq, traj in [("bear", "orbit_left"), ("camel", "zoom_in")]:
            d = root / "preds" / seq / traj
            d.mkdir(parents=True)
            (d / "pred_rgb.mp4").write_bytes(b"x")  # only the fallback name exists
            g = root / "gt" / seq / traj
            g.mkdir(parents=True)
            (g / "gt.mp4").write_bytes(b"x")
        pairs = discover_pairs(root / "preds")
        assert set(pairs) == {("bear", "orbit_left"), ("camel", "zoom_in")}, pairs
        samples = build_samples(pairs, root / "preds", gt_root=root / "gt",
                                prompts={"bear": "a bear"})
        assert len(samples) == 2
        s = {x["seq"]: x for x in samples}
        assert s["bear"]["pred"].name == "pred_rgb.mp4"
        assert s["bear"]["gt"].name == "gt.mp4"
        assert s["bear"]["prompt"] == "a bear"
        assert s["camel"]["prompt"] is None
    print("OK  test_contracts_resolution")


def _orbit_c2w(n=20, radius=1.0):
    """Synthetic forward-facing orbit: c2w with frame 0 ~ identity."""
    mats = []
    for i in range(n):
        a = 0.15 * i
        c2w = np.eye(4)
        c2w[:3, 3] = [radius * np.sin(a), 0.0, radius * (1 - np.cos(a))]
        mats.append(c2w)
    return np.stack(mats)


def test_pose_metrics_identity_and_perturbed():
    gt = _orbit_c2w()
    # identical -> ATE ~ 0
    m0 = compute_pose_metrics(gt.copy(), gt.copy())
    assert m0["ate"] < 1e-4, m0["ate"]
    assert m0["rot_err"] < 1e-2, m0["rot_err"]
    # perturbed translations -> strictly positive, finite ATE
    pred = gt.copy()
    pred[:, :3, 3] += 0.1 * np.random.RandomState(0).randn(gt.shape[0], 3)
    m1 = compute_pose_metrics(pred, gt.copy())
    assert m1["ate"] > m0["ate"] and np.isfinite(m1["ate"]), m1
    assert m1["alignment_mode"] in ("umeyama_align_scale", "origin_align_only")
    print(f"OK  test_pose_metrics (ate identity={m0['ate']:.2e}, perturbed={m1['ate']:.4f})")


if __name__ == "__main__":
    test_contracts_resolution()
    test_pose_metrics_identity_and_perturbed()
    print("\nAll smoke checks passed.")
