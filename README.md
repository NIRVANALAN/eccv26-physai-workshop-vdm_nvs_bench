# PhysAI Dynamic Video Novel-View Synthesis Challenge — participant starter kit

Starter evaluation code and public-data conventions for the PhysAI Dynamic
Video Novel-View Synthesis (NVS) challenge. It shares Syn4D sequence identities
with the companion 3D point-tracking challenge, but asks participants to
synthesize a video from a requested novel camera view.

**➡️ Submit predictions on Kaggle:**
[PhysAI Dynamic Video Novel View Synthesis](https://www.kaggle.com/competitions/phys-ai-dynamic-video-novel-view-synthesis)

## Task

Given a source video and a requested target-camera trajectory, generate the
corresponding target-view video. The public inputs use the same canonical Syn4D
`variant / scene / sequence` identities as the [Syn4D 3D Point-Tracking
Challenge starter kit](https://github.com/jzr99/syn4d-kaggle-challenge-participants).
For the current NVS protocol, source camera 0 is rendered toward target camera 1.

## Data

Download the NVS competition data from the [Kaggle challenge
page](https://www.kaggle.com/competitions/phys-ai-dynamic-video-novel-view-synthesis).
The public package provides source videos, requested target-camera trajectories,
query pixels, and a canonical pair CSV. It deliberately does **not** include
target-view RGB for the hidden test split.

The package follows this layout:

```
nvs_inputs/
  sources/<video>/<trajectory>/source.mp4
  cameras/<video>/<trajectory>.npz          # requested cam_c2w trajectory
  test_pairs.csv                            # canonical pairs and metadata
```

## Metric

The Kaggle leaderboard ranks submissions by **mean dense-video PSNR** against
hidden target-view videos (**higher is better**). Kaggle's metric callback
receives pre-extracted RGB values in a CSV: each row represents one pixel of a
**288×512** (height × width), first-49-frame target clip. It performs no video
file I/O or resizing itself. SSIM, LPIPS, camera ATE/RPE, CLIP/FVD, and VBench
are retained as diagnostic metrics in the local evaluator; they do not change
the Kaggle rank.

The Kaggle submission is therefore `submission.csv` with `id,R,G,B`. The
private `solution.csv` additionally contains `sequence`, `valid`, and `Usage`.
`Usage` assigns Public/Private rows and is removed before the metric callback.
The canonical row ids enumerate pixels in row-major order (`q = y * 512 + x`)
and frames `f000` through `f048`; no tracking 512-query subsampling is used.
The official split contains 128 clips, so a fully dense submission contains
`128 × 49 × 288 × 512 = 924,844,032` RGB rows. Generate this CSV upstream from
the fixed MP4 prediction tree using the exact canonical enumeration.

## Quickstart: validate a generated submission locally

```bash
# environment (Python >=3.10, CUDA GPU, ffmpeg)
git clone https://github.com/facebookresearch/vggt-omega && pip install -e vggt-omega
pip install -e .

# score a local validation split for which the organizer has target GT
vdm-nvs-bench eval --track syn4d \
  --pred submission/predictions --pairs nvs_inputs/test_pairs.csv --strict_submission \
  --source nvs_validation/sources --cameras nvs_validation/cameras \
  --gt nvs_validation/gt --out results/my_submission
cat results/my_submission/leaderboard.csv
```

For local development, a method writes one fixed-name MP4 per pair:

```
local_predictions/
  predictions/<video>/<trajectory>/pred.mp4
```

Every row in the provided `test_pairs.csv` must have one `pred.mp4`, and every
video must contain exactly **49 frames**. `--strict_submission` enforces this
fixed filename/path/frame-count contract locally before scoring. For Kaggle,
the MP4 predictions must be converted upstream to the dense `submission.csv`
described above; the metric callback scores that CSV only.

---

## Evaluation toolkit and developer reference

`vdm-nvs-bench` is the self-contained local evaluator for the ECCV'26 workshop
challenge. It installs in a Python environment and scores generated videos
without any private/cluster code.

There are two evaluation modes. The workshop **NVS challenge** uses the Syn4D
Kaggle split described in [Syn4D 3D Point-Tracking Challenge — participant
starter kit](https://github.com/jzr99/syn4d-kaggle-challenge-participants), so
the tracking and NVS challenges share exactly the same sequence identities and
render variants.

| Track | Target-view GT? | Camera metric | Video quality | Paired photometric |
|-------|:---:|--------|---------------|--------|
| **davis** | ❌ no | VGGT-Omega recovered pose vs the **requested trajectory** | FVD (pred vs source) + CLIP-F + CLIP-T + VBench | — |
| **syn4d** (official NVS) | hidden test; available on validation | VGGT-Omega pose vs the **requested target trajectory** | paired FVD (pred vs target) + CLIP-V + CLIP-F/T + VBench | PSNR / SSIM / LPIPS |

Why two tracks: DAVIS is real footage with **no** rendered novel target view, so
video quality is a *distribution* comparison and camera accuracy is estimated
from the generated pixels. Syn4D is synthetic, so every "novel" camera has a
pixel-aligned GT frame — enabling **paired** scoring.

---

## TL;DR for a coding agent

```bash
# 0. python env (>=3.10), on a box with a CUDA GPU + ffmpeg on PATH
conda create -n vdm-nvs-bench python=3.10 -y && conda activate vdm-nvs-bench
# 1. install the OFFICIAL camera model (vggt-omega) + this bench  (see "Setup" below)
git clone https://github.com/facebookresearch/vggt-omega && pip install -e vggt-omega
pip install -e .                            # this repo (vdm-nvs-bench); core = camera + video + paired
python scripts/download_weights.py          # VGGT-Omega ckpt ~4.6GB + I3D ~49MB

# 2a. DAVIS track
vdm-nvs-bench eval --track davis \
  --pred  PREDS/     `# PREDS/<seq>/<traj>/pred.mp4` \
  --cameras CAMERAS/ `# CAMERAS/<seq>/<traj>.npz  (key cam_c2w)` \
  --source SOURCES/  `# SOURCES/<seq>/<traj>/source.mp4  (optional but needed for FVD)` \
  --prompts prompts.json  `# optional {seq: caption}` \
  --out results/davis/

# 2b. Official Syn4D NVS submission / local validation
vdm-nvs-bench eval --track syn4d \
  --pred submission/predictions --pairs nvs_inputs/test_pairs.csv --strict_submission \
  --gt private_validation/gt --cameras nvs_inputs/cameras --source nvs_inputs/sources \
  --out results/syn4d/

# 3. read results/<track>/summary.json and results/<track>/leaderboard.csv
```
Add `--only camera,video` to run a subset. Add `--clip_model ViT-B-32` for a fast,
lighter (non-canonical) CLIP. VBench is an optional extra (`pip install -e '.[vbench]'`).

---

## Syn4D NVS data contract (local scorer and organizer)

The NVS challenge intentionally reuses the **Syn4D Kaggle sequence IDs** from
Zeren's [participant starter kit](https://github.com/jzr99/syn4d-kaggle-challenge-participants).
It is a separate NVS task: the input is a source video and a requested target-camera
trajectory; the submission is a synthesized target-view video. The target RGB and
all reference metrics remain private on the test split.

The Syn4D tracking starter kit is the shared upstream data/index reference:

```bash
git clone https://github.com/jzr99/syn4d-kaggle-challenge-participants
cd syn4d-kaggle-challenge-participants

# Follow the upstream kit to fetch Syn4D_Benchmark and unpack its public files.
# The organizer additionally releases nvs_inputs/ (below) for the NVS task.
hf download Syn4D/Syn4D_Benchmark --repo-type dataset --local-dir Syn4D_Benchmark
```

The NVS release used by the local scorer contains these public files:

```
nvs_inputs/
  sources/<video>/<trajectory>/source.mp4
  cameras/<video>/<trajectory>.npz          # requested cam_c2w; (T,4,4)
  test_pairs.csv                            # canonical list of all required pairs
```

`video` is the canonical Syn4D identity
`<variant>__<scene>__<seq_root>` (for example
`mixed__gothic__seq_000006`); `trajectory` is currently `src0_tgt1`. The
source is view 0 and the requested output is view 1. This gives both workshop
challenges a common, traceable split without exposing NVS test targets.

The local-video contract is a fixed MP4 tree:

```
local_predictions/
  predictions/<video>/<trajectory>/pred.mp4  # one generated video per required pair
```

`test_pairs.csv` is supplied by the organizer and fixes the evaluation set; it
is **not** a participant submission. Do not rename `pred.mp4`, substitute a
fallback filename, or omit a pair. Each MP4 must have exactly 49 frames. For
Kaggle, convert this tree upstream to the canonical dense CSV.

For local validation, the organizer keeps `gt/` beside the public inputs and
runs:

```bash
vdm-nvs-bench eval --track syn4d \
  --pred submission/predictions --pairs nvs_validation/test_pairs.csv --strict_submission \
  --source nvs_validation/sources --cameras nvs_validation/cameras \
  --gt nvs_validation/gt --num_frames 49 --out results/my_submission
cat results/my_submission/leaderboard.csv
```

`leaderboard.csv` contains one aggregate row for the submission. **PSNR
descending is the Kaggle ranking key** (`rank_metric=psnr`,
`rank_direction=descending`), while every computed metric remains in that same
CSV: ATE, rotation and
translation error, FVD, CLIP-F/T/V, PSNR, SSIM, LPIPS, and all five classic
VBench dimensions. Empty fields mean a metric was intentionally not computed
or lacked its required input; they are not silently treated as a good score.
FVD is set-level (not a per-video value), so it is reported for the complete
submission row.

### Organizer: materialize public inputs and private validation GT

The same local Syn4D layout used for tracking validation (for example
`/scratch/shared/beegfs/kelvin/Syn4D/subsets/kaggle_eval`) contains source and
target views plus camera CSVs. Generate the NVS package directly from it:

```bash
# Smoke test one pair first.
python scripts/make_syn4d_kaggle_nvs.py \
  --dataset-root /scratch/shared/beegfs/kelvin/Syn4D/subsets/kaggle_eval \
  --out nvs_validation --include-gt --limit 1

# Full validation package. Publish only sources/, cameras/, and test_pairs.csv;
# retain nvs_validation/gt privately.
python scripts/make_syn4d_kaggle_nvs.py \
  --dataset-root /scratch/shared/beegfs/kelvin/Syn4D/subsets/kaggle_eval \
  --out nvs_validation --include-gt
```

The materializer uses source view 0 → target view 1 and the first 49 frames by
default. Use the same `--num-frames` value in the evaluation command. It emits
the official `test_pairs.csv`; no hand-authored pair list is needed.

### Organizer: Kaggle ground truth

Keep the generated `solution.csv`, target videos, and any ground-truth
generation utilities private. Upload the resulting CSV only through Kaggle's
private competition-data interface; do not add it to this repository.

---

## Setup (one-time)

The evaluation pipeline is **not** magic — set it up explicitly so it is reproducible.
Nothing is silently vendored: the camera model comes from its **official** repo.

```bash
# 1) a clean python env (>=3.10) on a CUDA box with `ffmpeg` on PATH
conda create -n vdm-nvs-bench python=3.10 -y
conda activate vdm-nvs-bench

# 2) the OFFICIAL camera pose estimator — vggt-omega — cloned + installed
git clone https://github.com/facebookresearch/vggt-omega
pip install -e vggt-omega                      # provides the top-level `vggt_omega` package

# 3) this benchmark
pip install -e .                               # installs the `vdm-nvs-bench` CLI + `vdm_nvs_bench`

# 4) model weights (once)
python scripts/download_weights.py             # VGGT-Omega ckpt (~4.6GB) + styleganv I3D (~49MB)
                                               # CLIP ViT-H-14 (~3.7GB) auto-downloads on first CLIP call
```

Notes:
- If you cannot `pip install -e vggt-omega`, point the bench at the checkout instead:
  `export VGGT_OMEGA_REPO=/abs/path/to/vggt-omega`. The camera step raises a clear
  setup message if `vggt_omega` is neither installed nor reachable via that variable.
- **Re-running RecamMaster inference** additionally needs the `recammaster-official`
  repo (for its `diffsynth` `WanVideoReCamMasterPipeline`), the ReCamMaster `step20000`
  ckpt + `Wan2.1-T2V-1.3B` base, and the Syn4D dataset — see the data-package `REPRODUCE.md §6`.
  Scoring the shipped predictions does **not** need any of that.

---

## Requirements
- **Linux + CUDA GPU** (VGGT-Omega and FVD/CLIP need it); ≥ ~12 GB free for the camera model.
- `ffmpeg` binary on `PATH` (predictions are standardized with it before pose recovery).
- **`vggt-omega` from its official repo**, installed (`pip install -e vggt-omega`) or
  reachable via `$VGGT_OMEGA_REPO` — it is a code dependency, not bundled. See **Setup**.
- Weights auto-downloaded on first use (~9 GB total):
  - VGGT-Omega `vggt_omega_1b_512.pt` (~4.6 GB, HF `facebook/VGGT-Omega`) — camera.
  - CLIP ViT-H-14 laion2B (~3.7 GB, HF, via `open_clip`) — canonical CLIP metrics.
  - styleganv I3D (~49 MB) — FVD.
  - VBench models download themselves on first use (only if you run vbench).

---

## Input format (what YOU provide)

Everything is keyed by a `(seq, traj)` pair. `seq` = a scene/clip id, `traj` =
a camera-trajectory id. For the official Syn4D NVS challenge, use the supplied
`test_pairs.csv` as `--pairs` together with `--strict_submission`; auto-discovery
and fallback video names are private-development conveniences only.

```
<pred_root>/<seq>/<traj>/pred.mp4         # REQUIRED  your generated video (or pred_rgb.mp4)
<cameras_root>/<seq>/<traj>.npz           # REQUIRED  requested trajectory = camera GT
<source_root>/<seq>/<traj>/source.mp4     # source/input video
<gt_root>/<seq>/<traj>/gt.mp4            # SYN4D ONLY  paired target-view GT
prompts.json                              # optional  {"<seq>": "caption", ...}
pairs.csv                                 # official NVS: required header video,trajectory (+ optional metadata)
```

### File specs
| File | Format | Notes |
|------|--------|-------|
| `pred.mp4` | mp4, **≥ 10 frames** | any resolution/fps; auto-resized. `pred_rgb.mp4`, `<traj>.mp4`, `gen.mp4`, `output.mp4`, `render.mp4` are also accepted names. |
| `<traj>.npz` | `np.savez(cam_c2w=A)` | `A` shape **(T, 4, 4)** float32, **camera-to-world**, OpenCV convention (+Z forward, −Y up), frame 0 ≈ identity. Key may be `cam_c2w` or `c2w`. This is the trajectory the model was *asked* to follow. |
| `source.mp4` | mp4 | the input/source view. Needed for CLIP-V (syn4d) and the DAVIS FVD reference. |
| `gt.mp4` | mp4 | Syn4D only: the rendered target-view video for the requested trajectory. |
| `prompts.json` | JSON | `{seq: caption}`. Enables CLIP-T and is used by VBench temporal_style. |
| `pairs.csv` | CSV | Required challenge manifest: columns `video,trajectory`; extra metadata columns are allowed. If omitted, pairs are discovered from `<pred_root>/<seq>/<traj>/` (private use only). |

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
  leaderboard.csv       # challenge row: PSNR-ranked, all computed metrics (see below)
  leaderboard.tsv       # compatibility copy of the same row
  camera_metrics.json   # full camera schema (also embedded in summary.json)
  video_metrics.json
  paired_metrics.json   # syn4d only
  vbench/               # vbench raw outputs + parsed (if run)
  _cam_cache/           # per-pair VGGT pose npz + standardized frames (reusable cache)
```

### `leaderboard.csv` columns
`track,num_pairs,rank_metric,rank_direction,ate,rot_err,trans_err,fvd,clip_f,clip_t,clip_v,psnr,ssim,lpips,vbench_aesthetic_quality,vbench_imaging_quality,vbench_subject_consistency,vbench_background_consistency,vbench_temporal_style`
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

| dimension | score |
|-----------|:--:|
| aesthetic_quality | 0.602 |
| imaging_quality | 0.726 |
| subject_consistency | 0.932 |
| background_consistency | 0.954 |
| temporal_style | 0.079 |

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

## Reference baseline — RecamMaster on the held-out Syn4D subset

A real (non-GT) reference so you know what a working camera-control model scores on
this bench. **RecamMaster** (original ReCamMaster step20000, raw 480×832×81) on the
held-out Syn4D scene `flying_group/seq_000001`, source view 0 → target views 1..7
(7 pairs, the recammaster-official eval convention), scored with the **canonical
ViT-H-14** CLIP at native 81 frames:

| n | ate ↓ | trans_err ↓ | rot_err° ↓ | clip_v ↑ | clip_f ↑ | psnr ↑ | ssim ↑ | lpips ↓ |
|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 7 | **0.0478** | **0.0224** | **0.434** | **0.862** | **0.969** | **12.61** | **0.227** | **0.573** |

> **FVD is omitted here on purpose:** 7 clips is far too small for a stable Fréchet
> estimate. FVD remains a first-class bench metric on larger sets (e.g. the DAVIS track);
> it is just not meaningful as a 7-clip reference.

Contrast with the GT-vs-GT row above (ate 0.004, psnr 100): a real generative model
follows the requested camera to ~0.4° but its pixels are a genuine novel-view
*synthesis* (psnr ≈ 12.6, not 100). This is the expected shape of a real submission.

### Reproduce this row step-by-step (no Syn4D dataset, no inference needed)

The data package — the held-out inputs (rgb + depth + camera + mask) **and** the
RecamMaster raw-resolution (832×480×81) predictions — is published on Hugging Face:
[`yslan/ECCV26_PhysAI_Challenge_NVS_Syn4D_subset`](https://huggingface.co/datasets/yslan/ECCV26_PhysAI_Challenge_NVS_Syn4D_subset).

```bash
# a) one-time setup: env + official vggt-omega + this bench + weights (see "Setup" above)
git clone https://github.com/facebookresearch/vggt-omega && pip install -e vggt-omega
pip install -e . && python scripts/download_weights.py      # VGGT-Omega + I3D (CLIP auto-downloads)

# b) pull the data package and extract it
hf download yslan/ECCV26_PhysAI_Challenge_NVS_Syn4D_subset --repo-type=dataset --local-dir nvs_syn4d_subset
tar --use-compress-program=unzstd -xf nvs_syn4d_subset/nvs_syn4d_eval_set_recammaster.tar.zst
cd nvs_syn4d_eval_set

# c) score the shipped predictions and print the row above
export VDM_NVS_BENCH=/abs/path/to/this/vdm-nvs-bench/checkout
python tools/score_recammaster.py --out score_out          # camera(VGGT-Omega)+CLIP+PSNR/SSIM/LPIPS, ViT-H-14, 81 frames
python tools/final_row.py                                  # -> the reference row
```
`tools/score_recammaster.py` is a thin adapter: it maps the shipped RecamMaster outputs
into this bench's `(seq,traj)` contract (`cam_c2w = rel_target_c2w`) and calls
`vdm-nvs-bench eval --track syn4d`. To **re-infer** the predictions from the original
RecamMaster instead of using the shipped ones, follow `REPRODUCE.md §6` inside the archive
(needs the recammaster-official repo for `diffsynth`, the `step20000` ckpt, and the Syn4D dataset).

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
| `--pairs` | — | Official `test_pairs.csv` (`video,trajectory` + metadata) with `--strict_submission`; else auto-discover for private work. |
| `--strict_submission` | off | Official Kaggle validation: requires every pair's exact `predictions/<video>/<trajectory>/pred.mp4`, with exactly 49 frames. |
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
cat results/syn4d/leaderboard.csv
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
  camera/                      # VGGT-Omega adapter; official model is an external dependency
  video/eval_video.py          # FVD + CLIP-T/F/V (track-aware)
  video/paired.py              # PSNR/SSIM/LPIPS (syn4d)
  video/common_metrics/        # vendored styleganv-I3D FVD + psnr/ssim
  vbench_eval/run_vbench.py    # classic VBench CLI wrapper
scripts/{download_weights,make_syn4d_heldout,make_syn4d_kaggle_nvs,run_gt_sanity_1gpu}.py|sh
configs/{davis,syn4d_heldout}.yaml
tests/test_smoke.py            # no-GPU self-checks (contracts + evo core)
examples/                      # submission-format doc + gt-sanity summary
```

Run `python tests/test_smoke.py` (no GPU/weights) to validate contracts + the pose-metric core.
