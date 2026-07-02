#!/bin/bash
# 1-GPU GT-vs-GT sanity check for the Syn4D held-out track.
#
# Materializes a held-out Syn4D pair (source + GT target + camera GT), sets the
# "prediction" equal to the GT target video, and runs the full syn4d eval
# (camera + video + paired). A correct install yields near-perfect scores:
#   FVD ~ 0, PSNR ~ 100 (capped), SSIM ~ 1, LPIPS ~ 0, small camera ATE/RPE.
#
# Run on any node with 1 GPU (>= ~12 GB free) after `pip install -e .` (or with
# PYTHONPATH=. from the repo root). Override anything via env vars.
#
#   bash scripts/run_gt_sanity_1gpu.sh
#
# To score REAL predictions instead of the GT sanity, point PRED_ROOT at your
# own <seq>/<traj>/pred.mp4 tree (same seq/traj names this script emits).

set -euo pipefail

# ---- config (override via env) ----------------------------------------------
DATASET_ROOT="${DATASET_ROOT:-/scratch/shared/beegfs/zeren/Syn4D}"
SCENE="${SCENE:-flying_group}"           # held-out scenes: flying_group, train_group
SEQ_ROOT="${SEQ_ROOT:-seq_000000}"
SOURCE_VIEW="${SOURCE_VIEW:-0}"
TARGET_VIEWS="${TARGET_VIEWS:-1,2}"
NUM_FRAMES="${NUM_FRAMES:-49}"
CAPTION_CHUNK_SIZE="${CAPTION_CHUNK_SIZE:-81}"
HEIGHT="${HEIGHT:-288}"; WIDTH="${WIDTH:-512}"

GPU="${GPU:-0}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B-32}"     # canonical leaderboard: hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K
VGGT_CKPT="${VGGT_CKPT:-}"               # default: auto-download via weights.py (~4.6 GB)
OUT="${OUT:-./gt_sanity_out}"
PRED_ROOT="${PRED_ROOT:-}"               # empty => pred = gt (sanity); else your predictions

# ---- driver: prefer PYTHONPATH from repo, fall back to installed entrypoint --
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="python -m vdm_nvs_bench.cli"

echo "=== [1/3] materialize held-out GT: ${SCENE}/${SEQ_ROOT} src${SOURCE_VIEW} -> tgt{${TARGET_VIEWS}} ==="
python "${REPO_DIR}/scripts/make_syn4d_heldout.py" \
  --dataset_root "${DATASET_ROOT}" --scene "${SCENE}" --seq_root "${SEQ_ROOT}" \
  --source_view "${SOURCE_VIEW}" --target_views "${TARGET_VIEWS}" \
  --num_frames "${NUM_FRAMES}" --caption_chunk_size "${CAPTION_CHUNK_SIZE}" \
  --height "${HEIGHT}" --width "${WIDTH}" --out "${OUT}"

if [ -z "${PRED_ROOT}" ]; then
  echo "=== [2/3] pred = gt (sanity) ==="
  PRED_ROOT="${OUT}/preds"
  for d in "${OUT}"/gt/*/*; do
    seq="$(basename "$(dirname "${d}")")"; traj="$(basename "${d}")"
    mkdir -p "${PRED_ROOT}/${seq}/${traj}"
    cp "${d}/gt.mp4" "${PRED_ROOT}/${seq}/${traj}/pred.mp4"
  done
else
  echo "=== [2/3] scoring provided PRED_ROOT=${PRED_ROOT} ==="
fi

echo "=== [3/3] eval (camera + video + paired) on GPU ${GPU}, CLIP=${CLIP_MODEL} ==="
CKPT_ARG=(); [ -n "${VGGT_CKPT}" ] && CKPT_ARG=(--checkpoint "${VGGT_CKPT}")
${PY} eval --track syn4d \
  --pred "${PRED_ROOT}" --gt "${OUT}/gt" --source "${OUT}/sources" --cameras "${OUT}/cameras" \
  --out "${OUT}/results" --only camera,video,paired \
  --clip_model "${CLIP_MODEL}" --num_frames "${NUM_FRAMES}" --gpu 0 "${CKPT_ARG[@]}"

echo
echo "=== leaderboard (${OUT}/results/leaderboard.tsv) ==="
column -t -s$'\t' "${OUT}/results/leaderboard.tsv"
echo
echo "full JSON: ${OUT}/results/summary.json"
