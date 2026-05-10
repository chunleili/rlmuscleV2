"""Read source motion from a UsdSkel-bearing USD (e.g. walk_snu_ysy_v2.usd).

The source rig has 23 MASS-style joints under /Human/Joint_tree/Skeleton with
hierarchical paths like `Pelvis/FemurR/TibiaR/...`. Joint names used by
`joint_mapping.json` are the *leaf* of each path (e.g. "FemurR", "Torso").

Public:
    load_source_motion(usd_path) -> SourceMotion

A SourceMotion exposes:
    fps, frame_start, frame_end, n_frames
    joint_names: list[str]                       # length J, leaf names
    parent_idx:  list[int]                       # length J, parent index or -1
    rest_world:  np.ndarray (J, 4, 4)            # column-major (P @ direction = ...)
    world_xforms(time_code) -> np.ndarray (J,4,4)

Convention
----------
Returned 4x4 matrices are the *USD* convention (row-major Houdini-style:
translation in row 3, i.e. M[3,:3] is the position). This matches
`otto_motion.json` and Houdini's `export_otto_motion.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdSkel


@dataclass
class SourceMotion:
    fps: float
    frame_start: int
    frame_end: int
    joint_names: list[str]
    parent_idx: list[int]
    rest_world: np.ndarray  # (J, 4, 4)
    _stage: Usd.Stage
    _skel_query: UsdSkel.SkeletonQuery

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def n_frames(self) -> int:
        return self.frame_end - self.frame_start + 1

    def world_xforms(self, time_code: float) -> np.ndarray:
        """World transforms at `time_code` (USD time, == frame index for fps stage)."""
        cache = UsdGeom.XformCache(time_code)
        xforms = self._skel_query.ComputeJointWorldTransforms(cache)
        return np.array([np.array(M, dtype=np.float64) for M in xforms])


def load_source_motion(
    usd_path: str | Path,
    skel_path: str = "/Human/Joint_tree/Skeleton",
) -> SourceMotion:
    usd_path = Path(usd_path)
    assert usd_path.exists(), f"missing USD: {usd_path}"
    stage = Usd.Stage.Open(str(usd_path))
    skel = UsdSkel.Skeleton.Get(stage, skel_path)
    assert skel, f"no UsdSkel.Skeleton at {skel_path} in {usd_path}"

    cache = UsdSkel.Cache()
    skel_query = cache.GetSkelQuery(skel)

    joint_paths = list(skel_query.GetJointOrder())
    joint_names = [p.split("/")[-1] for p in joint_paths]

    topology = skel_query.GetTopology()
    parent_idx = [topology.GetParent(i) for i in range(len(joint_paths))]

    rest_xforms = skel_query.ComputeJointWorldTransforms(
        UsdGeom.XformCache(Usd.TimeCode.Default()), atRest=True
    )
    rest_world = np.array([np.array(M, dtype=np.float64) for M in rest_xforms])

    fps = float(stage.GetTimeCodesPerSecond())
    t0 = int(stage.GetStartTimeCode())
    t1 = int(stage.GetEndTimeCode())

    return SourceMotion(
        fps=fps,
        frame_start=t0,
        frame_end=t1,
        joint_names=joint_names,
        parent_idx=parent_idx,
        rest_world=rest_world,
        _stage=stage,
        _skel_query=skel_query,
    )


def main() -> int:
    """Self-test: load default source, dump shape summary."""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    src = PROJECT_ROOT.parent / "RLMuscle" / "output" / "walk_snu_ysy_v2.usd"
    if not src.exists():
        # fallback to V2-local
        src = PROJECT_ROOT / "output" / "walk_snu_ysy_v2.usd"
    motion = load_source_motion(src)

    print(f"[source_loader] loaded {src}")
    print(f"  fps        : {motion.fps}")
    print(f"  frame range: [{motion.frame_start}, {motion.frame_end}] "
          f"({motion.n_frames} frames)")
    print(f"  joints     : {motion.n_joints}")
    for i, (n, p) in enumerate(zip(motion.joint_names, motion.parent_idx)):
        parent_name = motion.joint_names[p] if p >= 0 else "ROOT"
        print(f"    [{i:2d}] {n:15s}  parent={parent_name}")

    # Sample world xforms at a few frames
    for f in [motion.frame_start, motion.frame_start + 100, motion.frame_end]:
        W = motion.world_xforms(float(f))
        pelvis_t = W[0, 3, :3]
        print(f"  t={f:>4}  Pelvis world translation = {pelvis_t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
