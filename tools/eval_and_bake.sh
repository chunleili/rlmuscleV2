#!/usr/bin/env bash
# Eval rl_mass checkpoint(s) -> save body trajectory NPZ -> bake snuV2 USD + BVH.
#
# Usage:
#   tools/eval_and_bake.sh <motion> [<motion> ...]
#
# Env vars:
#   CUDA_VISIBLE_DEVICES   GPU index (default: 0)
#   MAX_STEPS              eval max control steps (default: 300)
#   OUT_DIR                output dir (default: output/anim)
#   ASSET                  rest-pose USD (default: data/rl_mass/snuV2.usd)
#   CKPT_DIR               checkpoint dir (default: data/rl_mass/checkpoints)
#   CFG_DIR                config dir (default: data/rl_mass/configs)
#   STOCHASTIC=1           use stochastic actions (default: deterministic)
#   NUM_EPISODES           rollouts per motion (default: 1; only LAST episode is baked)
#   NO_DIAG=1              skip muscle activation diag + CSV
#
# Example:
#   tools/eval_and_bake.sh balance kick dance
#   CUDA_VISIBLE_DEVICES=1 MAX_STEPS=600 tools/eval_and_bake.sh walk
#
# Outputs (per <motion>):
#   $OUT_DIR/<motion>_traj.npz   body world transforms + qpos
#   $OUT_DIR/<motion>_act.csv    header: frame,<284 muscle names>; rows of float [0,1]
#   $OUT_DIR/<motion>.usd        SkelAnimation (snuV2 skeleton) + per-frame muscle:activation
#   $OUT_DIR/<motion>.bvh        HIERARCHY + MOTION (ZXY euler)

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <motion> [<motion> ...]" >&2
    echo "  e.g. $0 balance kick dance" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
OUT_DIR="${OUT_DIR:-output/anim}"
ASSET="${ASSET:-data/rl_mass/snuV2.usd}"
CKPT_DIR="${CKPT_DIR:-data/rl_mass/checkpoints}"
CFG_DIR="${CFG_DIR:-data/rl_mass/configs}"
NUM_EPISODES="${NUM_EPISODES:-1}"
STOCHASTIC_FLAG=""
if [ "${STOCHASTIC:-0}" = "1" ]; then
    STOCHASTIC_FLAG="--stochastic"
fi

mkdir -p "$OUT_DIR"

for m in "$@"; do
    cfg="$CFG_DIR/${m}.json"
    ckpt="$CKPT_DIR/${m}_max"
    traj="$OUT_DIR/${m}_traj.npz"
    diag_full="$OUT_DIR/${m}_diag_full.npz"
    act="$OUT_DIR/${m}_act.csv"
    out_base="$OUT_DIR/${m}"

    DIAG_FLAG=""
    if [ "${NO_DIAG:-0}" != "1" ]; then
        DIAG_FLAG="--diag $diag_full"
    fi

    if [ ! -f "$cfg" ]; then
        echo "[$m] missing config: $cfg" >&2; exit 2
    fi
    if [ ! -f "${ckpt}.pt" ] || [ ! -f "${ckpt}_muscle.pt" ]; then
        echo "[$m] missing ckpt: ${ckpt}.pt / ${ckpt}_muscle.pt" >&2; exit 2
    fi

    echo "=== [$m] eval ($NUM_EPISODES ep, max_steps=$MAX_STEPS) ==="
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" MUJOCO_GL=egl \
        uv run python -m rl_mass.eval \
            -d "$cfg" -m "$ckpt" \
            --num-episodes "$NUM_EPISODES" --max-steps "$MAX_STEPS" \
            --save-traj "$traj" $DIAG_FLAG $STOCHASTIC_FLAG

    BAKE_DIAG_FLAG=""
    if [ "${NO_DIAG:-0}" != "1" ]; then
        BAKE_DIAG_FLAG="--diag $diag_full"
    fi

    echo "=== [$m] bake snuV2 USD + BVH ==="
    uv run python tools/bake_motion.py \
        --traj "$traj" --asset "$ASSET" -o "$out_base" $BAKE_DIAG_FLAG

    if [ "${NO_DIAG:-0}" != "1" ]; then
        echo "=== [$m] strip diag -> activations CSV ==="
        uv run python -c "
import numpy as np, sys, csv
d = np.load(sys.argv[1], allow_pickle=False)
a = d['a'].astype(np.float32)
names = [str(n) for n in d['muscle_names']]
with open(sys.argv[2], 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frame', *names])
    for i, row in enumerate(a):
        w.writerow([i, *(f'{v:.6f}' for v in row)])
print(f'wrote {sys.argv[2]}: shape={a.shape}')
" "$diag_full" "$act"
        rm -f "$diag_full"
    fi
done

echo
echo "Done. Outputs in $OUT_DIR/:"
for m in "$@"; do
    ls -la "$OUT_DIR/${m}".{usd,bvh} \
           "$OUT_DIR/${m}_traj.npz" \
           "$OUT_DIR/${m}_act.csv" 2>/dev/null || true
done
