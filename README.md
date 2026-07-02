# vdm-nvs-bench

Self-contained evaluation for the **ECCV'26 workshop challenge: video-diffusion
novel-view synthesis (NVS) under camera control**. A participant clones this repo,
`pip install`s it, downloads weights once, and scores their generated videos —
no dependence on any private/cluster code.

There are two tracks:

| Track | Target-view GT? | Camera metric | Video quality | Paired photometric |
|-------|:---:|--------|---------------|--------|
| **davis** | ❌ no | VGGT-Omega recovered pose vs the **requested trajectory** | FVD (pred vs source) + CLIP-F + CLIP-T + VBench | — |
| **syn4d** | ✅ yes (held-out) | VGGT-Omega pose vs the **exact GT** trajectory | paired FVD (pred vs gt) + CLIP-V + CLIP-F/T + VBench | PSNR / SSIM / LPIPS |

Why two tracks: DAVIS is real footage with **no** rendered novel target view, so
video quality is a *distribution* comparison and camera accuracy is estimated
from the generated pixels. Syn4D is synthetic, so every "novel" camera has a
pixel-aligned GT frame — enabling **paired** scoring.

---

## TL;DR for a coding agent

```bash
# 1. install (core = camera + video + paired). Needs a CUDA GPU and ffmpeg.
pip install -e .
python scripts/download_weights.py          # VGGT-Omega ~4.6GB + I3D ~49MB

# 2a. DAVIS track
vdm-nvs-bench eval --track davis \
  --pred  PREDS/     `# PREDS/<seq>/<traj>/pred.mp4` \
  --cameras CAMERAS/ `# CAMERAS/<seq>/<traj>.npz  (key cam_c2w)` \
  --source SOURCES/  `# SOURCES/<seq>/<traj>/source.mp4  (optional but needed for FVD)` \
  --prompts prompts.json  `# optional {seq: caption}` \
  --out results/davis/

# 2b. Syn4D held-out track (needs paired GT)
vdm-nvs-bench eval --track syn4d \
  --pred PREDS/ --gt GT/ --cameras CAMERAS/ --source SOURCES/ \
  --out results/syn4d/

# 3. read results/<track>/summary.json  and results/<track>/leaderboard.tsv
```
Add `--only camera,video` to run a subset. Add `--clip_model ViT-B-32` for a fast,
lighter (non-canonical) CLIP. VBench is an optional extra (`pip install -e '.[vbench]'`).

---

## Requirements
- **Linux + CUDA GPU** (VGGT-Omega and FVD/CLIP need it); ≥ ~12 GB free for the camera model.
- `ffmpeg` binary on `PATH` (predictions are standardized with it before pose recovery).
- Weights auto-downloaded on first use (~9 GB total):
  - VGGT-Omega `vggt_omega_1b_512.pt` (~4.6 GB, HF `facebook/VGGT-Omega`) — camera.
  - CLIP ViT-H-14 laion2B (~3.7 GB, HF, via `open_clip`) — canonical CLIP metrics.
  - styleganv I3D (~49 MB) — FVD.
  - VBench models download themselves on first use (only if you run vbench).

---

## Input format (what YOU provide)

Everything is keyed by a `(seq, traj)` pair. `seq` = a scene/clip id, `traj` =
a camera-trajectory id. The pair list is either a CSV or auto-discovered by walking
`--pred`.

```
<pred_root>/<seq>/<traj>/pred.mp4         # REQUIRED  your generated video (or pred_rgb.mp4)
<cameras_root>/<seq>/<traj>.npz           # REQUIRED  requested trajectory = camera GT
<source_root>/<seq>/<traj>/source.mp4     # source/input video
<gt_root>/<seq>/<traj>/gt.mp4            # SYN4D ONLY  paired target-view GT
prompts.json                              # optional  {"<seq>": "caption", ...}
pairs.csv                                 # optional  header: video,trajectory
```

### File specs
| File | Format | Notes |
|------|--------|-------|
| `pred.mp4` | mp4, **≥ 10 frames** | any resolution/fps; auto-resized. `pred_rgb.mp4`, `<traj>.mp4`, `gen.mp4`, `output.mp4`, `render.mp4` are also accepted names. |
| `<traj>.npz` | `np.savez(cam_c2w=A)` | `A` shape **(T, 4, 4)** float32, **camera-to-world**, OpenCV convention (+Z forward, −Y up), frame 0 ≈ identity. Key may be `cam_c2w` or `c2w`. This is the trajectory the model was *asked* to follow. |
| `source.mp4` | mp4 | the input/source view. Needed for CLIP-V (syn4d) and the DAVIS FVD reference. |
| `gt.mp4` | mp4 | Syn4D only: the rendered target-view video for the requested trajectory. |
| `prompts.json` | JSON | `{seq: caption}`. Enables CLIP-T and is used by VBench temporal_style. |
| `pairs.csv` | CSV | columns `video,trajectory`. If omitted, pairs are discovered from `<pred_root>/<seq>/<traj>/`. |

### Required vs optional, per track
- **davis**: `--pred` + `--cameras` required. `--source` strongly recommended (else FVD is skipped). `--prompts` optional (else CLIP-T skipped).
- **syn4d**: `--pred` + `--cameras` + `--gt` + `--source` required for the full suite.

The camera metric is **alignment-invariant** (Sim(3) Umeyama with scale, with an
origin-only fallback for pan/zoom-only GT), so your `cam_c2w` world frame, global
rotation, and scale need not match any canonical frame — only the trajectory
*shape* matters.

---

## Output format (what the tool writes under `--out`)

```
results/<track>/
  summary.json          # everything, merged
  leaderboard.tsv       # one header row + one values row (see columns below)
  camera_metrics.json   # full camera schema (also embedded in summary.json)
  video_metrics.json
  paired_metrics.json   # syn4d only
  vbench/               # vbench raw outputs + parsed (if run)
  _cam_cache/           # per-pair VGGT pose npz + standardized frames (reusable cache)
```

### `leaderboard.tsv` columns
`track  num_pairs  ate  rot_err  trans_err  fvd  clip_f  clip_t  clip_v  psnr  ssim  lpips`
(blank cell = not computed for this track/inputs).

### `summary.json` schema
```jsonc
{
  "track": "syn4d",
  "num_pairs": 7,
  "components": ["camera", "video", "paired"],
  "camera": { /* see camera_metrics.json below */ },
  "video":  {
    "track": "syn4d", "num_samples": 7, "clip_model": "...",
    "clip_t": {"value": float, "std": float, "count": int} | null,
    "clip_f": {"value": ..., "std": ..., "count": ...} | null,
    "clip_v": {"value": ..., "std": ..., "count": ...} | null,
    "fvd":    {"value": float, "reference": "gt"|"source",
               "num_videos": int, "num_frames": int} | null
  },
  "paired": {"psnr": float, "ssim": float, "lpips": float,
             "count": int, "num_frames": int},        // syn4d only
  "vbench": {"vbench": {"aesthetic_quality": float, ...}, "eval_jsons": [...]}
}
```

### `camera_metrics.json` schema
```jsonc
{
  "pred_root": "...", "pose_estimator": "vggt_omega",
  "num_pairs_evaluated": int, "num_pairs_total": int, "checkpoint": "...",
  "ate":       {"mean": float, "std": float, "median": float, "n": int},
  "trans_err": {"mean": ..., "std": ..., "median": ..., "n": ...},
  "rot_err":   {"mean": ..., "std": ..., "median": ..., "n": ...},   // degrees
  "sim3_subset":  {"ate": {...}, "trans_err": {...}, "rot_err": {...}}, // pairs aligned with Sim3
  "origin_subset":{"ate": {...}, "trans_err": {...}, "rot_err": {...}}, // pairs that fell back to origin-align
  "all_norm":     {"ate": {...}, "trans_err": {...}, "rot_err": {...}}, // scale-normalized ATE/trans
  "per_pair": [
    {"seq": "...", "traj": "...", "ate": float, "trans_err": float,
     "rot_err": float, "ate_norm": float|null, "trans_err_norm": float|null,
     "gt_centered_norm": float, "pred_centered_norm": float,
     "num_aligned_poses": int, "alignment_mode": "umeyama_align_scale"|"origin_align_only"}
  ]
}
```

---

## Metric definitions & direction

| Metric | ↑/↓ | What it measures |
|--------|:--:|------------------|
| **ATE** | ↓ | Absolute trajectory error (RMSE of translation) after Sim(3) alignment, in GT units. Camera-following accuracy. |
| **rot_err** | ↓ | Relative pose error, rotation angle (degrees), 1-frame deltas, all pairs. |
| **trans_err** | ↓ | Relative pose error, translation part (RMSE). |
| **FVD** | ↓ | Fréchet Video Distance (styleganv I3D features). Distribution distance: pred vs **source** (davis) or pred vs **gt** (syn4d). |
| **CLIP-T** | ↑ | Mean cosine( frame image-embedding, prompt text-embedding ). Needs prompts. |
| **CLIP-F** | ↑ | Mean cosine between adjacent pred frames. Temporal consistency. |
| **CLIP-V** | ↑ | Mean cosine( source frame, pred frame ) at matched timesteps. Cross-view consistency. |
| **PSNR / SSIM** | ↑ | Paired per-pixel/structural similarity vs the GT target video (syn4d). |
| **LPIPS** | ↓ | Paired perceptual distance (AlexNet) vs GT target (syn4d). |
| **VBench dims** | ↑ | aesthetic_quality, imaging_quality, subject_consistency, background_consistency, temporal_style (no-reference). |

Camera pose is recovered from the generated video by **VGGT-Omega** and compared
against the requested trajectory — so it scores *whether the video actually moved
the camera as instructed*, independent of image content.

---

## GT-vs-GT sanity check (verified)

Feeding the **GT target video as the "prediction"** must produce near-perfect
scores. This was run end-to-end on the held-out Syn4D scene `flying_group/seq_000000`,
source view 0 → target views 1..7 (7 pairs), CLIP `ViT-B-32`, VGGT-Omega `1b_512`:

**Camera + video + paired:**

| track | pairs | ate ↓ | rot_err° ↓ | trans_err ↓ | fvd ↓ | clip_f ↑ | clip_v ↑ | psnr ↑ | ssim ↑ | lpips ↓ |
|-------|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| syn4d | 7 | **0.0041** | **0.0413** | **0.0036** | **−0.0009** | **0.9857** | **0.8583** | **100.0** | **1.000** | **0.000** |

**VBench (classic, no-reference, ↑; run in a `vbench` env on the same 7 clips):**

| aesthetic_quality | imaging_quality | subject_consistency | background_consistency | temporal_style |
|--:|--:|--:|--:|--:|
| 0.6018 | 0.7261 | 0.9325 | 0.9538 | 0.0790 |

Interpretation (this is the expected shape of a correct install):
- **Paired metrics are perfect**: PSNR caps at 100, SSIM = 1, LPIPS = 0 — pred is byte-identical GT.
- **FVD ≈ 0**: identical distributions (tiny negative = numerical noise in the Fréchet estimate).
- **Camera ATE/RPE are small but not exactly 0**: VGGT re-estimates poses from *pixels*, so even the true target video yields a slightly noisy trajectory — this is the estimator's noise floor, not a bug.
- **CLIP-V = 0.86 (< 1)**: source and target are genuinely different viewpoints, so cross-view cosine is high but not 1. CLIP-F = 0.99 (adjacent frames are very similar).
- **CLIP-T blank**: no prompts were passed.
- **VBench**: subject/background consistency are high (~0.93–0.95, coherent real renders); `temporal_style` is low (~0.08) because it is a prompt-referenced dimension and no prompts were supplied in `custom_input` mode — pass `--prompts` to make it meaningful.

Reproduce on a single GPU:
```bash
bash scripts/run_gt_sanity_1gpu.sh
# knobs: GPU=0 CLIP_MODEL=ViT-B-32 SCENE=flying_group SEQ_ROOT=seq_000000 \
#        TARGET_VIEWS=1,2,3,4,5,6,7 DATASET_ROOT=/path/to/Syn4D VGGT_CKPT=/path/to/ckpt \
#        bash scripts/run_gt_sanity_1gpu.sh
# to score real predictions instead of the GT sanity: set PRED_ROOT=/your/preds
```
The raw JSON of this run is committed at `examples/syn4d_gt_sanity_summary.json`.

---

## Building the Syn4D held-out GT package

`scripts/make_syn4d_heldout.py` converts a reserved Syn4D scene into the submission
contract (`sources/`, `gt/`, `cameras/`). It is self-contained (PIL + numpy;
`vdm_nvs_bench/data/syn4d_loader.py`) and does **not** import the training pipeline.

```bash
python scripts/make_syn4d_heldout.py \
  --dataset_root /path/to/Syn4D --scene flying_group --seq_root seq_000000 \
  --source_view 0 --target_views 1,2,3 --num_frames 49 --caption_chunk_size 81 \
  --out heldout/
```
Held-out scenes (recammaster convention): `flying_group`, `train_group` (excluded
from training). `cam_c2w` is the target view's pose relative to the source frame 0.

---

## CLI reference (`vdm-nvs-bench eval`)
| Flag | Default | Meaning |
|------|---------|---------|
| `--track` | (req) | `davis` or `syn4d`. |
| `--pred` | (req) | prediction root `<seq>/<traj>/pred.mp4`. |
| `--cameras` | — | requested-trajectory GT root `<seq>/<traj>.npz`. Required for the camera component. |
| `--source` | — | source-video root. |
| `--gt` | — | Syn4D paired target-GT root. |
| `--prompts` | — | `prompts.json`. |
| `--pairs` | — | pairs CSV; else auto-discover. |
| `--out` | (req) | output dir. |
| `--only` | all for the track | comma list of `camera,video,paired,vbench`. |
| `--checkpoint` | auto-download | VGGT-Omega `.pt`. |
| `--clip_model` | `hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | canonical; use `ViT-B-32` for a light run. |
| `--fvd_size` | 256 | frames resized to this square for FVD/paired. |
| `--num_frames` | 49 | frames used for pose recovery / trajectory length. |
| `--target_w/--target_h` | 512/288 | standardization size before VGGT. |
| `--gpu` | 0 | GPU index (also honor `CUDA_VISIBLE_DEVICES`). |

---

## Worked example — `assets/example_pair/`

A concrete, committed instance of the contract (one Syn4D held-out pair,
`flying_group/seq_000000`, source view 0 → target view 1):
```
assets/example_pair/
  source.mp4             # input/source view       (49 frames, 512x288)
  target.mp4             # GT target-view video for the requested trajectory
  target_trajectory.npz  # np.load(...)["cam_c2w"] -> (49,4,4) float32 c2w
```
Inspect the npz:
```python
import numpy as np
d = np.load("assets/example_pair/target_trajectory.npz")
print(d["cam_c2w"].shape, d["cam_c2w"].dtype)   # (49, 4, 4) float32
```
Turn it into a runnable one-pair submission tree and score it (perfect "prediction" = GT):
```bash
seq=demo; traj=t0
mkdir -p ex/preds/$seq/$traj ex/gt/$seq/$traj ex/sources/$seq/$traj ex/cameras/$seq
cp assets/example_pair/target.mp4            ex/preds/$seq/$traj/pred.mp4
cp assets/example_pair/target.mp4            ex/gt/$seq/$traj/gt.mp4
cp assets/example_pair/source.mp4            ex/sources/$seq/$traj/source.mp4
cp assets/example_pair/target_trajectory.npz ex/cameras/$seq/$traj.npz
vdm-nvs-bench eval --track syn4d \
  --pred ex/preds --gt ex/gt --source ex/sources --cameras ex/cameras \
  --out ex/results --only camera,paired --clip_model ViT-B-32
```
(one pair is enough for `camera` + `paired`; `fvd` needs ≥ 2 videos so it is `null` with a single pair.)

## Full reproduction walkthrough (agent copy-paste)

```bash
# 0) env: a CUDA box with ffmpeg. From the repo root:
pip install -e .
python scripts/download_weights.py            # VGGT-Omega + I3D (CLIP auto-downloads on first CLIP call)

# 1) fastest end-to-end proof it all works (no dataset needed): the smoke self-checks
python tests/test_smoke.py                     # contracts + evo pose core, no GPU/weights

# 2) GT-vs-GT sanity on real Syn4D held-out data (materialize GT, set pred=gt, score)
#    Requires the Syn4D dataset; override DATASET_ROOT/VGGT_CKPT as needed.
bash scripts/run_gt_sanity_1gpu.sh             # prints a leaderboard row; see table above

# 3) score a REAL submission
vdm-nvs-bench eval --track syn4d \
  --pred  /path/preds   --gt /path/gt --source /path/sources --cameras /path/cameras \
  --out results/syn4d/
cat results/syn4d/leaderboard.tsv
python -c "import json;print(json.dumps(json.load(open('results/syn4d/summary.json'))['video'],indent=2))"
```

## Running VBench (classic, 5 no-reference dims)

VBench is a separate suite; it downloads its own models on first use and its CLI
takes **one `--dimension` per call** (the wrapper loops over the 5 dims). Install
the extra, or reuse an env that already has the `vbench` CLI:
```bash
pip install -e '.[vbench]'                     # needs CUDA present at build time (deepspeed dep)
# or reuse an existing env:
conda activate /path/to/vbench-env             # must have `vbench` on PATH
CUDA_VISIBLE_DEVICES=0 vdm-nvs-bench eval --track davis \
  --pred PREDS/ --out results/ --only vbench
```
Output: `results/<track>/vbench/*_eval_results.json` (raw) parsed into
`summary.json["vbench"]`. Dims (all ↑): aesthetic_quality, imaging_quality,
subject_consistency, background_consistency, temporal_style.

## Troubleshooting
- **CUDA OOM on the camera step**: VGGT-Omega needs ~8–10 GB. Use a freer GPU (`--gpu N` / `CUDA_VISIBLE_DEVICES`), and `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **`vbench` not found**: install the extra `pip install -e '.[vbench]'` (needs CUDA present at build time for its deepspeed dep). vbench is skipped gracefully otherwise.
- **FVD errors "< 10 frames"**: every video must have ≥ 10 frames.
- **ffmpeg missing**: install it (`conda install ffmpeg` or system package); predictions are re-encoded before pose recovery.
- **Weights**: run `python scripts/download_weights.py`, or set `VGGT_OMEGA_CKPT=/path/to/vggt_omega_1b_512.pt`. The I3D weight is gitignored and fetched on demand (never committed, to avoid git-LFS stubs).

---

## Repo layout
```
vdm_nvs_bench/
  cli.py                       # `vdm-nvs-bench eval ...` orchestrator
  data/contracts.py            # submission-folder parsing
  data/syn4d_loader.py         # dependency-light Syn4D reader (held-out prep)
  weights.py                   # VGGT-Omega + I3D download / resolve
  camera/eval_camera.py        # VGGT-Omega inference driver
  camera/pose_metrics.py       # evo ATE/RPE core
  camera/vggt_omega/           # vendored VGGT-Omega package
  video/eval_video.py          # FVD + CLIP-T/F/V (track-aware)
  video/paired.py              # PSNR/SSIM/LPIPS (syn4d)
  video/common_metrics/        # vendored styleganv-I3D FVD + psnr/ssim
  vbench_eval/run_vbench.py    # classic VBench CLI wrapper
scripts/{download_weights,make_syn4d_heldout,run_gt_sanity_1gpu}.py|sh
configs/{davis,syn4d_heldout}.yaml
tests/test_smoke.py            # no-GPU self-checks (contracts + evo core)
examples/                      # submission-format doc + gt-sanity summary
```

Run `python tests/test_smoke.py` (no GPU/weights) to validate contracts + the pose-metric core.
