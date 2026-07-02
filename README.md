# vdm-nvs-bench

Self-contained evaluation for the **ECCV'26 workshop challenge: video-diffusion
novel-view synthesis under camera control**. Clone, install, download weights, run.

Two tracks:

| Track | Target-view GT | Camera | Video quality | Paired |
|-------|:---:|--------|---------------|--------|
| **davis** | no | VGGT-Omega recovered pose vs requested trajectory | FVD (pred vs source) + CLIP-F/CLIP-T + VBench | — |
| **syn4d** | yes (held-out) | VGGT-Omega pose vs exact GT | paired FVD (pred vs gt) + CLIP-V + CLIP-F/T + VBench | PSNR / SSIM / LPIPS |

## Requirements
- Linux + **CUDA GPU** (VGGT-Omega and FVD/CLIP need it), `ffmpeg` on PATH.
- ~9 GB of weights auto-download on first run (VGGT-Omega ~4.6 GB, CLIP ViT-H-14 ~3.7 GB, I3D ~49 MB).

## Install
```bash
pip install -e .            # core (camera + video)
pip install -e '.[vbench]'  # add VBench (needs CUDA at build time)
python scripts/download_weights.py
```

## Submission format
```
<pred_root>/<seq>/<traj>/pred.mp4        # your prediction        (required, >=10 frames)
<cameras_root>/<seq>/<traj>.npz          # requested trajectory   (key: cam_c2w, (T,4,4) c2w)
<source_root>/<seq>/<traj>/source.mp4    # source/input video     (CLIP-V, davis flat-FVD)
<gt_root>/<seq>/<traj>/gt.mp4            # Syn4D only: paired target-view GT
prompts.json  = {"<seq>": "caption", ...} # optional (CLIP-T / VBench temporal_style)
```
Pairs are read from a `--pairs pairs.csv` (columns `video,trajectory`) or auto-discovered
by walking `<pred_root>/<seq>/<traj>/`.

## Run
```bash
# DAVIS
vdm-nvs-bench eval --track davis \
  --pred preds/ --cameras cameras/ --source sources/ --prompts prompts.json \
  --out results/davis/

# Syn4D held-out
vdm-nvs-bench eval --track syn4d \
  --pred preds/ --gt gt/ --cameras cameras/ --source sources/ \
  --out results/syn4d/
```
Outputs `summary.json` + `leaderboard.tsv` (plus per-component JSONs) under `--out`.
Run a subset with `--only camera,video`. Use `--clip_model ViT-B-32` for a lighter smoke run
(non-canonical numbers).

## What each metric is
- **Camera** (`camera_metrics.json`): ATE / RPE-trans / RPE-rot via evo, Sim(3)-Umeyama
  alignment (origin-only fallback for pan/zoom GT). Pose recovered from the generated
  video by VGGT-Omega, compared to the requested target trajectory.
- **Video** (`video_metrics.json`): styleganv-I3D FVD (distribution) + CLIP-T/F/V (open_clip).
  DAVIS uses source as the FVD reference; Syn4D uses the paired GT.
- **Paired** (`paired_metrics.json`, Syn4D only): PSNR / SSIM / LPIPS vs the GT target video.
- **VBench** (`vbench/`): 5 no-reference dims (aesthetic_quality, imaging_quality,
  subject_consistency, background_consistency, temporal_style).

## Layout
```
vdm_nvs_bench/
  cli.py                     # `vdm-nvs-bench eval ...`
  data/contracts.py          # submission-folder parsing
  weights.py                 # VGGT-Omega + I3D download
  camera/{eval_camera,pose_metrics}.py + vggt_omega/   # vendored VGGT-Omega
  video/{eval_video,paired}.py + common_metrics/       # vendored FVD/PSNR/SSIM
  vbench_eval/run_vbench.py  # classic VBench CLI wrapper
```
