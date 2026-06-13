#!/usr/bin/env bash
# Pull all demo outputs from the server (SSH alias `gnome`) to local.
# Run this ON YOUR LOCAL machine:  bash pull_demos.sh [local_dest_dir]
set -euo pipefail

REMOTE=gnome
REMOTE_ROOT=/mnt/storage_disk_1/chunleli/Dev/rlmuscleV2/output
DEST="${1:-./rlmuscle_demos}"

# demo artifacts: USD files + mesh-sequence dirs
ITEMS=(
  bicep_oneway.usd          # MuscleSim (bicep)
  udcf_u_drop.usd           # MassSpringSim U-drop (self-collision)
  sdf_drop.usd              # SDF U vs plane
  rigid_soft.usd            # SDF rigid sphere vs soft
  humanV2_sdf_coll.usd      # SDF muscle vs animated bone
  lion_oneway.usd           # OneWaySim lion
  humanv2_oneway.usd        # OneWaySim humanV2 gmg
  minotaur_oneway.usd       # OneWaySim minotaur
  minotaur_leg_oneway.usd   # OneWaySim minotaur leg
)

mkdir -p "$DEST"
echo "pulling demos from $REMOTE:$REMOTE_ROOT -> $DEST"
for it in "${ITEMS[@]}"; do
  echo "  $it"
  scp -r "$REMOTE:$REMOTE_ROOT/$it" "$DEST/" 2>/dev/null \
    || echo "    SKIP (not found on server): $it"
done
echo "done -> $DEST"
ls -la "$DEST"
