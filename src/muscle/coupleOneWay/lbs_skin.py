"""Linear Blend Skinning (LBS) baseline for one-way bone-driven muscles.

Why
---
Without collision, constraint-only solve makes muscle bodies lag far behind
the bones during fast motion. Solution: LBS provides a kinematic prediction
(rigid-skin each muscle vertex to its nearest bone), and XPBD constraints
(volume / ARAP / fiber / attach / pintotarget) act as elastic corrections on
top of that prediction. ARAP automatically blends across bone boundaries
even with 1-hot weights.

Weight strategy
---------------
1-hot nearest-bone assignment: for each muscle vertex, find the nearest
bone-mesh vertex via KDTree, then look up which bone slice it belongs to.
The assignment is cached to JSON so subsequent runs skip the KDTree build.

Per-step transform
------------------
Row-vector convention: ``x_world(t) = x_local @ M_t`` where
``M_t = offset_bone @ joint_world_t``. We pre-compute ``inv(M_rest_bone)``
once at init, so per substep we only need a (4,4) @ (4,4) per bone:

    T_rel[bone] = inv(M_rest_bone) @ M_t_bone

and the LBS target is ``x_lbs = x_pos0 @ T_rel[bone_of_vert]``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _build_vertex_bone_assignment(
    muscle_pos_rest_world: np.ndarray,
    bone_pos_rest_world: np.ndarray,
    bone_slices,
) -> np.ndarray:
    """Nearest-bone-vertex 1-hot assignment.

    Returns ``(N_muscle_verts,) int32`` of bone slice indices in the same
    order as ``bone_slices``. -1 if no bones present.
    """
    if bone_pos_rest_world.shape[0] == 0:
        return np.full(muscle_pos_rest_world.shape[0], -1, dtype=np.int32)

    # bone-pos vert -> bone slice idx
    bone_of_vert = np.empty(bone_pos_rest_world.shape[0], dtype=np.int32)
    for bone_idx, (_, sl, _, _) in enumerate(bone_slices):
        bone_of_vert[sl] = bone_idx

    # Lazy import scipy to avoid making it a hard dep when LBS is disabled.
    from scipy.spatial import cKDTree
    tree = cKDTree(bone_pos_rest_world)
    _, j = tree.query(muscle_pos_rest_world, k=1)
    return bone_of_vert[j].astype(np.int32)


class LBSDriver:
    """Per-substep LBS predictor that snaps muscle pos toward bone-skinned target.

    Stores the per-vertex bone index plus pre-inverted rest transforms, then
    each substep computes ``T_rel = inv(M_rest) @ M_t`` and uploads it to a
    flat Warp buffer. The actual GPU apply is in ``muscle_warp.lbs_apply_kernel``.

    Parameters
    ----------
    muscle_pos_rest_world
        Rest-pose muscle vertex positions in world space, ``(N, 3)``.
    anim
        ``BoneAnimation`` instance — provides ``_bone_slices`` (rest world xforms),
        ``_offset``, ``_driver_idx``.
    cache_path
        Optional path to cache the per-vertex bone assignment as JSON. If
        the file exists and matches the muscle vertex count, the assignment
        is loaded instead of recomputed.
    """

    def __init__(self, muscle_pos_rest_world: np.ndarray, anim, cache_path=None):
        self.anim = anim
        self.n_verts = int(muscle_pos_rest_world.shape[0])
        self.n_bones = len(anim._bone_slices)

        # Pre-compute inverse rest transforms (4,4) per bone.
        self._M_rest_inv = np.empty((self.n_bones, 4, 4), dtype=np.float64)
        for i, (_, _, _, M_rest) in enumerate(anim._bone_slices):
            self._M_rest_inv[i] = np.linalg.inv(M_rest)

        # Resolve / cache vertex->bone assignment.
        self.vertex_bone = self._load_or_build(
            muscle_pos_rest_world, anim, cache_path
        )

        # Sanity: bones with no muscle vertices assigned still get processed
        # (cheap), but we log so users know.
        bone_counts = np.bincount(
            self.vertex_bone[self.vertex_bone >= 0],
            minlength=self.n_bones,
        )
        empty = int((bone_counts == 0).sum())
        log.info(f"[LBS] {self.n_verts} verts → {self.n_bones} bones, "
                 f"{empty} bones unused, top: "
                 f"{sorted(zip(bone_counts, range(self.n_bones)), reverse=True)[:5]}")

        # Persistent buffer for per-call T_rel_flat (cuts allocations per substep).
        self._T_rel_flat = np.zeros(self.n_bones * 12, dtype=np.float32)

    def _load_or_build(self, muscle_pos_rest_world, anim, cache_path):
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                try:
                    d = json.loads(cache_path.read_text())
                    arr = np.asarray(d["vertex_bone"], dtype=np.int32)
                    if arr.shape[0] == self.n_verts and int(d.get("n_bones", -1)) == self.n_bones:
                        log.info(f"[LBS] loaded cached weights from {cache_path}")
                        return arr
                    log.warning(f"[LBS] cache mismatch at {cache_path} — rebuilding")
                except Exception as e:
                    log.warning(f"[LBS] failed to load {cache_path}: {e!r} — rebuilding")

        log.info(f"[LBS] building vertex→bone assignment for {self.n_verts} verts...")
        arr = _build_vertex_bone_assignment(
            muscle_pos_rest_world, anim.bone_pos_rest, anim._bone_slices,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "n_verts": self.n_verts,
                "n_bones": self.n_bones,
                "bone_names": [b[0] for b in anim._bone_slices],
                "strategy": "nearest_bone_vertex_1hot",
                "vertex_bone": arr.tolist(),
            }))
            log.info(f"[LBS] cached weights to {cache_path}")
        return arr

    # ------------------------------------------------------------------
    # Per-substep transform (numpy)
    # ------------------------------------------------------------------

    def compute_T_rel_flat(self, joint_world: np.ndarray) -> np.ndarray:
        """Compute (n_bones, 12) flat T_rel rows: [r00,r01,r02, r10,r11,r12, r20,r21,r22, t0,t1,t2].

        Result is a (n_bones * 12,) float32 array suitable for upload to a Warp
        ``wp.array(dtype=wp.float32)`` and direct indexing in the LBS kernel.
        """
        anim = self.anim
        out = self._T_rel_flat
        for i, (_, _, _, M_rest) in enumerate(anim._bone_slices):
            j = anim._driver_idx[i]
            if j < 0:
                # Bone has no driver joint — stay at rest, T_rel = identity.
                out[i * 12 : i * 12 + 12] = (
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0,
                )
                continue
            M_t = anim._offset[i] @ joint_world[j]  # (4,4) row-vector
            T_rel = self._M_rest_inv[i] @ M_t        # row-vector: x_t = x_rest @ T_rel
            # Flatten rows 0..2 cols 0..2, then row 3 cols 0..2.
            block = out[i * 12 : i * 12 + 12]
            block[0:3] = T_rel[0, :3]
            block[3:6] = T_rel[1, :3]
            block[6:9] = T_rel[2, :3]
            block[9:12] = T_rel[3, :3]
        return out

    def apply(self, sim, joint_world: np.ndarray, blend: float):
        """Compute T_rel for ``joint_world`` and overwrite sim.pos with the
        LBS target (lerped by ``blend`` from the current sim.pos).

        Caller is responsible for picking the right time/substep for joint_world
        (e.g. fractional-frame sample).
        """
        if blend <= 0.0:
            return
        flat = self.compute_T_rel_flat(joint_world)
        sim.set_lbs_T_rel(flat)  # uploads to GPU
        sim.apply_lbs(self.vertex_bone_field(sim), blend=blend)

    # ------------------------------------------------------------------
    # GPU resource helpers (lazy-allocated on the sim's device)
    # ------------------------------------------------------------------

    def vertex_bone_field(self, sim):
        """Lazy-init Warp int32 array of vertex→bone idx, on sim's device."""
        import warp as wp
        if getattr(self, "_vertex_bone_wp", None) is None:
            self._vertex_bone_wp = wp.from_numpy(self.vertex_bone, dtype=wp.int32)
        return self._vertex_bone_wp
