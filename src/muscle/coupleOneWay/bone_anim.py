"""One-way bone-driven muscle: animated bone_pos source.

Consumes:
- Human.usd Ragdoll bones (rest geometry, SNU naming)
- otto_motion.json: per-frame world transforms of every Otto joint after
  Houdini FBIK retargeting (see data/Human/export_otto_motion.py)

For each Ragdoll bone we look up its joint by name (1:1, no ancestor
fallback needed since every Ragdoll bone has a matching Otto joint after
FBIK). At rest we precompute a fixed rigid offset:

    offset = bone_world_rest @ inv(joint_rest)        # row-vector convention
    bone_world(t) = offset @ joint_world(t)

bone_pos vertex layout matches mesh_io._load_bone_usd traversal so the
existing attach KDTree / bone_pos_field pipeline works unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _gf_to_np(m):
    """Convert pxr.Gf.Matrix4d to (4,4) np.float64."""
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def _apply_xform4(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Row-vector convention: pts_w = pts_l @ M[:3,:3] + M[3,:3]."""
    return pts @ M[:3, :3] + M[3, :3]


class BoneAnimation:
    """Animated bone source compatible with `_load_bone_usd` layout.

    Parameters
    ----------
    human_usd : Path or str
        Static SNU rig + bone meshes. Reads ``/Human/Ragdoll/Bones`` for
        rest geometry (mesh-local verts and rest world xform).
    motion_json : Path or str
        JSON produced by ``data/Human/export_otto_motion.py``. Contains
        ``joint_names``, ``rest_transforms`` and per-frame ``transforms``
        (all row-major 4x4, row-vector convention).
    bone_root : str
        SNU bone root prim. Default ``/Human/Ragdoll/Bones``.
    """

    def __init__(
        self,
        human_usd,
        motion_json,
        bone_root: str = "/Human/Ragdoll/Bones",
    ):
        from pxr import Usd, UsdGeom
        self._Usd = Usd
        self._UsdGeom = UsdGeom

        self.human_usd = Path(human_usd)
        self.motion_json = Path(motion_json)
        self.bone_root = bone_root

        # 1) motion JSON
        self._load_motion(self.motion_json)

        # 2) Human.usd: rest bone geometry (matches mesh_io._load_bone_usd order)
        h_stage = Usd.Stage.Open(str(self.human_usd))
        if h_stage is None:
            raise FileNotFoundError(f"Cannot open {self.human_usd}")
        self._load_bones_rest(h_stage)

        # 3) Per-bone driver index + rest offset
        self._compute_offsets()

        # output buffer (will be overwritten each sample)
        self._world_buf = self._bone_pos_rest_local.copy()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_motion(self, path: Path):
        with open(path, "r") as fh:
            d = json.load(fh)
        self.fps = float(d["fps"])
        self.frame_start = int(d["frame_start"])
        self.frame_end = int(d["frame_end"])
        self.joint_names = list(d["joint_names"])
        self._name_to_idx = {n: i for i, n in enumerate(self.joint_names)}

        n_j = len(self.joint_names)
        self.rest_transforms = np.asarray(d["rest_transforms"], dtype=np.float64).reshape(n_j, 4, 4)

        frames = d["frames"]
        n_f = len(frames)
        # Sanity: frames listed in order
        for i, fr in enumerate(frames):
            if int(fr["frame"]) != self.frame_start + i:
                raise ValueError(f"motion_json frames not contiguous "
                                 f"at index {i}: got {fr['frame']}, "
                                 f"expected {self.frame_start + i}")
        self._frames = np.asarray(
            [fr["transforms"] for fr in frames], dtype=np.float64
        ).reshape(n_f, n_j, 4, 4)
        self.n_frames = n_f
        print(f"[BoneAnimation] motion: {n_j} joints, {n_f} frames "
              f"[{self.frame_start},{self.frame_end}] @ {self.fps} fps")

    def _load_bones_rest(self, stage):
        """Walk bone_root with same traversal as mesh_io._load_bone_usd."""
        Usd, UsdGeom = self._Usd, self._UsdGeom
        rest_t = stage.GetStartTimeCode()
        if rest_t == self._Usd.TimeCode.Default().GetValue():
            rest_t = 1.0
        xcache = UsdGeom.XformCache(rest_t)

        root = stage.GetPrimAtPath(self.bone_root)
        if not root or not root.IsValid():
            raise ValueError(f"{self.bone_root} not in {self.human_usd}")

        # entries: (snu_bone_name, slice, local_verts, rest_world_xform)
        self._bone_slices: list = []
        all_pos_world: list[np.ndarray] = []
        all_idx: list[np.ndarray] = []
        muscle_ids_list: list[str] = []
        offset = 0

        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(prim)
            pts_raw = m.GetPointsAttr().Get()
            if pts_raw is None:
                continue
            pts_local = np.asarray(pts_raw, dtype=np.float32)
            idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

            parent = prim.GetParent()
            bone_name = parent.GetName() if parent and parent.IsValid() else prim.GetName()

            world_mat = _gf_to_np(xcache.GetLocalToWorldTransform(prim))
            pts_world = _apply_xform4(world_mat, pts_local).astype(np.float32)

            n = len(pts_local)
            self._bone_slices.append((bone_name, slice(offset, offset + n), pts_local, world_mat))
            all_pos_world.append(pts_world)
            all_idx.append(idx + offset)
            muscle_ids_list.extend([bone_name] * n)
            offset += n

        self._bone_pos_rest_local = np.vstack(all_pos_world).astype(np.float32)
        self.bone_indices = np.concatenate(all_idx).astype(np.int32) if all_idx else np.zeros(0, np.int32)

        muscle_ids: dict[str, list[int]] = {}
        for v_idx, mid in enumerate(muscle_ids_list):
            muscle_ids.setdefault(mid, []).append(v_idx)
        self.bone_muscle_ids = {k: np.array(v, dtype=np.int32) for k, v in muscle_ids.items()}

    def _compute_offsets(self):
        """Resolve each Ragdoll bone to a motion-JSON joint and compute the
        fixed rigid offset that rigidly bolts the bone to that joint frame.

        offset = bone_world_rest @ inv(joint_rest)   (row-vector convention)
        => bone_world(t) = offset @ joint_world(t)
        """
        self._driver_idx: list[int] = []
        self._offset: list[np.ndarray] = []
        unresolved = []
        for (bone_name, _, _, bone_world_rest) in self._bone_slices:
            j = self._name_to_idx.get(bone_name, -1)
            if j < 0:
                self._driver_idx.append(-1)
                self._offset.append(np.eye(4))
                unresolved.append(bone_name)
                continue
            joint_rest = self.rest_transforms[j]
            self._driver_idx.append(j)
            self._offset.append(bone_world_rest @ np.linalg.inv(joint_rest))
        if unresolved:
            print(f"[BoneAnimation] {len(unresolved)} bones not in motion JSON "
                  f"(stay at rest): {unresolved[:8]}"
                  f"{'...' if len(unresolved) > 8 else ''}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def bone_pos_rest(self) -> np.ndarray:
        return self._bone_pos_rest_local

    @property
    def time_seconds_range(self) -> tuple[float, float]:
        return (0.0, (self.n_frames - 1) / self.fps)

    def sample_at_frame(self, frame_idx: int) -> np.ndarray:
        """Return (N,3) world bone_pos at the given frame (clamped, integer)."""
        i = int(np.clip(frame_idx, 0, self.n_frames - 1))
        joint_world = self._frames[i]  # (n_j, 4, 4)
        return self._apply_joint_world(joint_world)

    def sample_at_fractional_frame(self, f: float, loop: bool = True) -> np.ndarray:
        """Return bone_pos at fractional frame `f` via linear-interpolated joint xforms.

        Linear blend of two consecutive 4x4 matrices is approximate for rotations,
        but for small inter-frame angles (which is the regime when num_substeps>>1
        finely subdivides one frame interval) the drift is negligible.
        """
        return self._apply_joint_world(self.joint_world_at_fractional_frame(f, loop))

    def joint_world_at_fractional_frame(self, f: float, loop: bool = True) -> np.ndarray:
        """Return the (n_joints, 4, 4) joint-world matrix at fractional frame `f`.

        Same lerp as ``sample_at_fractional_frame`` but exposes the underlying
        joint transform array (LBS uses this to derive per-bone rigid xforms).
        """
        if loop:
            f = f % self.n_frames
        f = float(np.clip(f, 0.0, self.n_frames - 1))
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, self.n_frames - 1)
        t = f - i0
        if i0 == i1 or t == 0.0:
            return self._frames[i0]
        return (1.0 - t) * self._frames[i0] + t * self._frames[i1]

    def _apply_joint_world(self, joint_world: np.ndarray) -> np.ndarray:
        """Run the bone-rest -> joint-driven world xform pipeline once."""
        out = self._world_buf
        for (bone_name, sl, local_verts, bone_world_rest), j, off in zip(
            self._bone_slices, self._driver_idx, self._offset
        ):
            if j < 0:
                out[sl] = _apply_xform4(bone_world_rest.astype(np.float32), local_verts)
                continue
            bw = (off @ joint_world[j]).astype(np.float32)
            out[sl] = _apply_xform4(bw, local_verts)
        return out

    def sample_at_seconds(self, t_sec: float, loop: bool = True) -> np.ndarray:
        """Return bone_pos at sim-time `t_sec`. Loops by default."""
        f = t_sec * self.fps
        if loop:
            f = f % self.n_frames
        return self.sample_at_frame(int(round(f)))
