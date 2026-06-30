"""UsdSkel motion reader -> DART/MuJoCo qpos, mirroring rl_mass.bvh.BVH.

Lets `Character` consume a UsdSkel `SkelAnimation` as the reference motion
instead of a `.bvh` file. Exposes the exact interface `Character` touches on a
motion source (`T0`, `T1`, `cyclic`, `max_time`, `get_motion_dart`), so
`Character.load_usd` swaps it in transparently for `load_bvh`.

The motion USD is expected to carry the SAME per-joint local rotations and
root translation (centimetres) conventions as the source BVH -- e.g. a faithful
Houdini `bvh -> UsdSkel` export. Joint names follow `bvh_map` (Character1_*
Mixamo by default; pass `MASS_NATIVE_BVH_MAP` for MASS-native names).

Usage:
    from rl_mass.usd_motion import UsdSkelMotion
    m = UsdSkelMotion()
    m.parse("data/rl_mass/motion/motion.mass.walk.usd", cyclic=True)
    dart_q = m.get_motion_dart(t=0.0)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from pxr import Usd, UsdSkel

from rl_mass.bvh import REVOLUTE_AXES, _make_transform, dart_qpos_from_local_R
from rl_mass.joint_utils import (
    MASS_BODY_NAMES,
    MASS_BVH_MAP,
    MASS_JOINT_TYPES,
)

# UsdSkel translations inherited from the BVH are in centimetres (BVH unit);
# DART/MuJoCo want metres. Matches BVH's hard-coded 0.01 cm->m factor.
_CM_TO_M = 0.01


class UsdSkelMotion:
    """UsdSkel SkelAnimation motion source, interface-compatible with `BVH`.

    Args:
        bvh_map: body_name -> source joint (leaf) name. None -> MASS_BVH_MAP.
        joint_types / body_names / revolute_axes: skeleton description; default
            to the MASS humanoid tables (same defaults as `BVH`).
    """

    def __init__(
        self,
        bvh_map: dict[str, str] | None = None,
        joint_types: list[str] | None = None,
        body_names: list[str] | None = None,
        revolute_axes: dict[str, np.ndarray] | None = None,
    ):
        self.bvh_map = bvh_map or MASS_BVH_MAP
        self.joint_types = joint_types or MASS_JOINT_TYPES
        self.body_names = body_names or MASS_BODY_NAMES
        self.revolute_axes = revolute_axes or REVOLUTE_AXES
        self.cyclic = True

        # Motion data (filled by parse)
        self.time_step = 0.0
        self.num_total_frames = 0
        self._local_R: Optional[np.ndarray] = None     # (F, J, 3, 3)
        self._root_trans: Optional[np.ndarray] = None  # (F, 3) metres
        self._name2idx: dict[str, int] = {}

        # Root transforms (first and last frame), for Character cyclic blending.
        self.T0 = np.eye(4, dtype=np.float64)
        self.T1 = np.eye(4, dtype=np.float64)

    @property
    def max_time(self) -> float:
        """Total duration of the motion clip."""
        return self.num_total_frames * self.time_step

    def parse(
        self,
        usd_path: str,
        cyclic: bool = True,
        skel_path: str | None = None,
        trans_scale: float = _CM_TO_M,
    ) -> None:
        """Read a UsdSkel motion clip into per-frame local rotations.

        Args:
            usd_path: path to the motion USD (UsdSkel Skeleton + SkelAnimation).
            cyclic: whether the clip loops (matches BVH semantics).
            skel_path: optional explicit Skeleton prim path; otherwise the first
                Skeleton found is used.
            trans_scale: root-translation unit scale (cm->m by default).
        """
        self.cyclic = cyclic
        stage = Usd.Stage.Open(str(usd_path))
        assert stage, f"could not open USD: {usd_path}"

        skel = None
        if skel_path:
            skel = UsdSkel.Skeleton.Get(stage, skel_path)
        else:
            for prim in stage.Traverse():
                if prim.GetTypeName() == "Skeleton":
                    skel = UsdSkel.Skeleton(prim)
                    break
        assert skel, f"no UsdSkel.Skeleton found in {usd_path}"

        skel_query = UsdSkel.Cache().GetSkelQuery(skel)
        leaf_names = [p.split("/")[-1] for p in skel_query.GetJointOrder()]
        self._name2idx = {n: i for i, n in enumerate(leaf_names)}

        # Verify the skeleton carries every mapped source joint.
        missing = sorted(
            self.bvh_map[b] for b in self.body_names
            if b in self.bvh_map and self.bvh_map[b] not in self._name2idx
        )
        if missing:
            raise KeyError(
                f"USD skeleton {usd_path} is missing mapped joints: {missing}"
            )

        fps = float(stage.GetTimeCodesPerSecond()) or 60.0
        self.time_step = 1.0 / fps

        # Frame timecodes: prefer the animation's authored samples; fall back to
        # the stage's integer time range.
        anim = None
        for prim in stage.Traverse():
            if prim.GetTypeName() == "SkelAnimation":
                anim = UsdSkel.Animation(prim)
                break
        timecodes: list[float] = []
        if anim is not None:
            timecodes = sorted(anim.GetRotationsAttr().GetTimeSamples())
        if not timecodes:
            t0 = int(round(stage.GetStartTimeCode()))
            t1 = int(round(stage.GetEndTimeCode()))
            timecodes = [float(i) for i in range(t0, t1 + 1)]

        F = len(timecodes)
        J = len(leaf_names)
        self.num_total_frames = F
        self._local_R = np.zeros((F, J, 3, 3), dtype=np.float64)
        self._root_trans = np.zeros((F, 3), dtype=np.float64)

        root_idx = self._name2idx[self.bvh_map[self.body_names[0]]]
        for fi, tc in enumerate(timecodes):
            locals_t = skel_query.ComputeJointLocalTransforms(Usd.TimeCode(tc))
            mats = np.array([np.array(M, dtype=np.float64) for M in locals_t])
            # USD stores row-major (v_world = v_local @ M); transpose upper-left
            # 3x3 to get R with the column convention v_world = R @ v_local.
            self._local_R[fi] = np.transpose(mats[:, :3, :3], (0, 2, 1))
            self._root_trans[fi] = mats[root_idx, 3, :3] * trans_scale

        self.T0 = _make_transform(self._local_R[0, root_idx], self._root_trans[0])
        self.T1 = _make_transform(
            self._local_R[F - 1, root_idx], self._root_trans[F - 1]
        )

    def get_motion_dart(self, t: float) -> np.ndarray:
        """Get motion at time `t` in DART qpos format (exp-map).

        Mirrors `BVH.get_motion_dart`: floor-based frame selection with cyclic
        wrapping, then per-joint local rotation -> DART exp-map.
        """
        k = int(math.floor(t / self.time_step))
        if self.cyclic:
            k = k % self.num_total_frames
        k = max(0, min(k, self.num_total_frames - 1))

        R_k = self._local_R[k]
        return dart_qpos_from_local_R(
            lambda n: R_k[self._name2idx[n]],
            self._root_trans[k],
            self.bvh_map,
            self.joint_types,
            self.body_names,
            self.revolute_axes,
        )

    def get_motion_mujoco(self, t: float) -> np.ndarray:
        """Raw motion at time `t` in MuJoCo qpos (no Character T0/Tc processing)."""
        from rl_mass.joint_utils import DART_ROOT_OFFSET, dart_qpos_to_mujoco

        dart_q = self.get_motion_dart(t)
        return dart_qpos_to_mujoco(dart_q, root_offset=DART_ROOT_OFFSET)
