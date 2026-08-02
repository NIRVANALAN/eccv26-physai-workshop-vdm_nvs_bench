# iPhone NVS validation benchmark

This is a small, public validation benchmark for developing dynamic novel-view
synthesis models.  It is separate from the official Syn4D Kaggle test set:
the iPhone bundle deliberately includes target RGB and evaluation masks so
participants can measure their model locally.

Download and extract the validation bundle:

```bash
hf download yslan/ECCV26_PhysAI_Challenge_NVS_Syn4D_subset \
  iphone_nvs_validation_bundle.tar.gz --repo-type dataset --local-dir iphone_nvs_download
tar -xzf iphone_nvs_download/iphone_nvs_validation_bundle.tar.gz
```

The bundle contains five Shape-of-Motion iPhone sequences: `apple`, `block`,
`paper-windmill`, `spin`, and `teddy`.  Each is a source-camera-0 to target-view
pair (`tgt2` for `apple`, `tgt1` for the other four sequences).

## Protocol

Every source, target, and prediction is exactly **49 frames**, **288×512**
(height × width), RGB, and 12 fps.  The iPhone images are center-cropped to
16:9 before resizing.  The released source depth and camera intrinsics have
the identical crop/resize transformation applied.

`cameras/<sequence>/<trajectory>.npz` contains:

- `source_K`, `target_K`: `(49, 3, 3)` intrinsics for the 288×512 videos.
- `source_c2w_query`, `target_c2w_query`: `(49, 4, 4)` poses in the
  source-frame-0 query coordinate system.  `source_c2w_query[0]` is identity.
- `source_c2w_world`, `target_c2w_world` and matching `w2c` arrays: poses in
  the refined COLMAP world coordinate system.

`source_depth/<sequence>/<trajectory>.npz` stores source LiDAR `depth_m`, with
shape `(49, 288, 512)`; its validity mask is `depth_m > 0`.  `masks/...npz` contains target-view boolean
`valid`, `covisible`, `foreground`, and `evaluation_mask = valid & covisible`.
The current five-sequence release has no foreground annotation, so
`foreground` is explicitly all false rather than omitted.

## Directory contract

After extracting the bundle, its layout is:

```text
iphone_nvs_validation/
  test_pairs.csv
  sources/<sequence>/<trajectory>/source.mp4
  source_depth/<sequence>/<trajectory>.npz
  cameras/<sequence>/<trajectory>.npz
  gt/<sequence>/<trajectory>/gt.mp4
  masks/<sequence>/<trajectory>.npz
```

Run your method on every source/camera pair and create:

```text
my_predictions/
  apple/src0_tgt2/pred.mp4
  block/src0_tgt1/pred.mp4
  paper-windmill/src0_tgt1/pred.mp4
  spin/src0_tgt1/pred.mp4
  teddy/src0_tgt1/pred.mp4
```

The fixed filename, resolution, and frame count are intentional.  They make
the validation contract identical to the video contract used by the Syn4D NVS
challenge, apart from iPhone validation exposing its target RGB.

## Evaluate a prediction

The lightweight evaluator reports full-image PSNR/SSIM and masked PSNR over
the DyCheck headline region `valid & covisible`:

```bash
python scripts/eval_iphone_nvs_validation.py \
  --bundle iphone_nvs_validation \
  --pred my_predictions \
  --out results/iphone_nvs_metrics.json
```

For the repository's standard PSNR/SSIM/LPIPS report, use:

```bash
vdm-nvs-bench eval --track syn4d --only paired --strict_submission \
  --pred my_predictions --pairs iphone_nvs_validation/test_pairs.csv \
  --source iphone_nvs_validation/sources --gt iphone_nvs_validation/gt \
  --cameras iphone_nvs_validation/cameras \
  --out results/iphone_nvs_paired
```

These iPhone scores are local validation diagnostics.  They are not a Kaggle
submission and do not change the official leaderboard, whose test target RGB
remains hidden.
