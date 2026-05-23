#!/usr/bin/env bash
# Fresh-start supervisor for the 4 imitation trainings, picking up the new
# env.py (track_root_translation + r_com). One task per GPU.
set -euo pipefail

BASE='/lus/lfs1aip2/scratch/u6ey/chunleili.u6ey/Dev/rlmuscleV2'
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)-nid011212-n64-rootxlat-rcom}"
cd "$BASE"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

printf 'fresh_start=%s host=%s slurm_cuda=%s run_id=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-}" "$RUN_ID"

launch() {
  local name="$1" cfg="$2" gpu="$3"
  local out="$BASE/output/$name/runs/$RUN_ID"
  mkdir -p "$out/nn"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    printf 'task_start=%s host=%s gpu=%s task=%s cfg=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "$CUDA_VISIBLE_DEVICES" \
      "$name" "$cfg"
    exec .venv/bin/python -m rl_mass.train \
      -d "$cfg" \
      --outdir "$out" \
      --num_envs 64
  ) > "$out/train.log" 2>&1 &
  echo "$!" > "$out/python.pid"
  printf 'launched task=%s pid=%s out=%s\n' "$name" "$(cat "$out/python.pid")" "$out"
}

launch swim-cyclic    data/rl_mass/configs/swim-cyclic.json    0
launch overhead-squat data/rl_mass/configs/overhead-squat.json 1
launch kick-preroll   data/rl_mass/configs/kick-preroll.json   2
launch walk           data/rl_mass/configs/walk.json           3

wait
