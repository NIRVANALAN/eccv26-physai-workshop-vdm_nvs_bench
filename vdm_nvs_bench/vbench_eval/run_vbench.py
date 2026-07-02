"""VBench (classic) no-reference video-quality wrapper.

Builds a flat folder of prediction mp4s and drives the `vbench` CLI in
custom-input mode, then parses the 5 classic dimensions out of the result JSON.
VBench downloads its own model weights on first use. Requires `pip install
vdm-nvs-bench[vbench]` (the `vbench` extra) and a GPU.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

DIMENSIONS = (
    "aesthetic_quality",
    "imaging_quality",
    "subject_consistency",
    "background_consistency",
    "temporal_style",
)


def _metric_value(payload: object) -> Optional[float]:
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
        return float(payload[0])
    if isinstance(payload, dict):
        for key in ("value", "score", "mean"):
            val = payload.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _build_flat_dir(samples: List[dict], work_dir: Path) -> Path:
    flat = work_dir / "_vbench_inputs"
    flat.mkdir(parents=True, exist_ok=True)
    for old in flat.glob("video*.mp4"):
        old.unlink()
    for i, s in enumerate(samples):
        link = flat / f"video{i}.mp4"
        try:
            link.symlink_to(Path(s["pred"]).resolve())
        except OSError:
            shutil.copy2(Path(s["pred"]), link)
    return flat


def run_vbench_eval(
    samples: List[dict],
    out_dir: Path,
    dimensions=DIMENSIONS,
    imaging_quality_preprocessing_mode: str = "longer",
    ngpus: int = 1,
) -> dict:
    """Run classic VBench on the predictions and return {dim: value}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = _build_flat_dir(samples, out_dir)

    if shutil.which("vbench") is None:
        raise RuntimeError(
            "`vbench` CLI not found. Install the extra: pip install 'vdm-nvs-bench[vbench]'"
        )
    cmd = [
        "vbench", "evaluate",
        "--dimension", *list(dimensions),
        "--videos_path", str(flat),
        "--mode", "custom_input",
        "--output_path", str(out_dir),
        "--imaging_quality_preprocessing_mode", imaging_quality_preprocessing_mode,
        "--ngpus", str(ngpus),
    ]
    print("[vbench]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return parse_vbench_results(out_dir, dimensions)


def parse_vbench_results(out_dir: Path, dimensions=DIMENSIONS) -> dict:
    """Read the newest value for each dimension from *_eval_results.json."""
    jsons = sorted(Path(out_dir).glob("*_eval_results.json"))
    result = {"vbench": {}, "eval_jsons": [str(p) for p in jsons]}
    for dim in dimensions:
        val = None
        for jp in reversed(jsons):
            data = json.loads(jp.read_text())
            val = _metric_value(data.get(dim))
            if val is not None:
                break
        result["vbench"][dim] = val
    return result


def main() -> None:
    import argparse
    import csv

    ap = argparse.ArgumentParser(description="Classic VBench over a pred tree")
    ap.add_argument("--pred_root", type=Path, required=True)
    ap.add_argument("--pairs_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--pred_name", default="pred.mp4")
    args = ap.parse_args()

    with open(args.pairs_csv, newline="") as fh:
        pairs = [(r["video"], r["trajectory"]) for r in csv.DictReader(fh)]
    samples = []
    for seq, traj in pairs:
        p = args.pred_root / seq / traj / args.pred_name
        if p.is_file():
            samples.append({"pred": p})
    res = run_vbench_eval(samples, args.out_dir)
    (args.out_dir / "vbench_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["vbench"], indent=2))


if __name__ == "__main__":
    main()
