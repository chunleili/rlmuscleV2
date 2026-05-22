"""BVH parser ported from MASS (external/MASS/core/BVH.cpp).

Parses BVH motion capture files and maps to MuJoCo qpos format.
The DART original uses exp-map for BallJoint and FreeJoint; we output
MuJoCo quaternions and translations directly.

Usage:
    from rl_mass.bvh import BVH
    bvh = BVH(bvh_map, joint_types, revolute_axes)
    bvh.parse("external/MASS/data/motion/walk.bvh", cyclic=True)
    mj_qpos = bvh.get_motion(t=0.0)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from rl_mass.joint_utils import (
    DART_ROOT_OFFSET,
    MASS_BVH_MAP,
    MASS_BODY_NAMES,
    MASS_JOINT_TYPES,
    expmap_to_quat_wxyz,
    rotmat_to_quat_wxyz,
)

# ---------------------------------------------------------------------------
# Rotation helpers (input in degrees, matches BVH.cpp exactly)
# ---------------------------------------------------------------------------

# Use the same constant as MASS C++ code (3.141592/180.0)
_DEG2RAD = 3.141592 / 180.0


def _R_x(deg: float) -> np.ndarray:
    """Rotation matrix about X axis by `deg` degrees."""
    c = math.cos(deg * _DEG2RAD)
    s = math.sin(deg * _DEG2RAD)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _R_y(deg: float) -> np.ndarray:
    """Rotation matrix about Y axis by `deg` degrees."""
    c = math.cos(deg * _DEG2RAD)
    s = math.sin(deg * _DEG2RAD)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _R_z(deg: float) -> np.ndarray:
    """Rotation matrix about Z axis by `deg` degrees."""
    c = math.cos(deg * _DEG2RAD)
    s = math.sin(deg * _DEG2RAD)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Channel enum
# ---------------------------------------------------------------------------

_CHANNEL_NAME = {
    "Xposition": 0, "XPOSITION": 0,
    "Yposition": 1, "YPOSITION": 1,
    "Zposition": 2, "ZPOSITION": 2,
    "Xrotation": 3, "XROTATION": 3,
    "Yrotation": 4, "YROTATION": 4,
    "Zrotation": 5, "ZROTATION": 5,
}

_Xpos, _Ypos, _Zpos = 0, 1, 2
_Xrot, _Yrot, _Zrot = 3, 4, 5


# ---------------------------------------------------------------------------
# BVHNode
# ---------------------------------------------------------------------------

class BVHNode:
    """A single node in the BVH hierarchy tree."""

    def __init__(self, name: str, parent: Optional[BVHNode] = None):
        self.name = name
        self.parent = parent
        self.children: list[BVHNode] = []
        self.channel_offset = 0
        self.num_channels = 0
        self.channels: list[int] = []
        self.R = np.eye(3, dtype=np.float64)

    def set_channel(self, c_offset: int, c_names: list[str]) -> None:
        """Set channel info from BVH CHANNELS line."""
        self.channel_offset = c_offset
        self.num_channels = len(c_names)
        self.channels = [_CHANNEL_NAME[cn] for cn in c_names]

    def set_from_frame(self, m_t: np.ndarray) -> None:
        """Accumulate rotation from frame data (BVHNode::Set)."""
        self.R = np.eye(3, dtype=np.float64)
        for i in range(self.num_channels):
            val = m_t[self.channel_offset + i]
            ch = self.channels[i]
            if ch == _Xrot:
                self.R = self.R @ _R_x(val)
            elif ch == _Yrot:
                self.R = self.R @ _R_y(val)
            elif ch == _Zrot:
                self.R = self.R @ _R_z(val)
            # Xpos/Ypos/Zpos: skip (don't affect rotation)

    def set_rotation(self, R: np.ndarray) -> None:
        """Directly set rotation matrix."""
        self.R = R.copy()

    def get(self) -> np.ndarray:
        """Return current rotation matrix."""
        return self.R

    def get_node(self, name: str) -> Optional[BVHNode]:
        """Recursively find node by name."""
        if self.name == name:
            return self
        for c in self.children:
            found = c.get_node(name)
            if found is not None:
                return found
        return None


# ---------------------------------------------------------------------------
# Rotation math helpers
# ---------------------------------------------------------------------------

def _logmap(R: np.ndarray) -> np.ndarray:
    """Compute log map of rotation matrix -> axis-angle vector (3D).

    Equivalent to DART BallJoint::convertToPositions(R).
    """
    return Rotation.from_matrix(R).as_rotvec()


def _expmap_to_rotmat(v: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector to rotation matrix.

    Equivalent to DART math::expMapRot(v).
    """
    return Rotation.from_rotvec(v).as_matrix()


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from R (3x3) and t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _transform_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ---------------------------------------------------------------------------
# Revolute axis info
# ---------------------------------------------------------------------------

# Map body name -> revolute axis (from human.xml)
# Only for RevoluteJoint bodies
REVOLUTE_AXES: dict[str, np.ndarray] = {
    "TibiaR": np.array([1.0, 0.0, 0.0]),
    "TibiaL": np.array([1.0, 0.0, 0.0]),
    "FootThumbR": np.array([1.0, 0.0, 0.0]),
    "FootPinkyR": np.array([1.0, 0.0, 0.0]),
    "FootThumbL": np.array([1.0, 0.0, 0.0]),
    "FootPinkyL": np.array([1.0, 0.0, 0.0]),
    "ForeArmR": np.array([0.0, 1.0, 0.0]),
    "ForeArmL": np.array([0.0, 1.0, 0.0]),
}

# DART DOFs per joint type (for indexing into DART qpos)
_DART_DOF = {"Free": 6, "Ball": 3, "Revolute": 1}
# MuJoCo qpos size per joint type
_MJ_QPOS = {"Free": 7, "Ball": 4, "Revolute": 1}


# ---------------------------------------------------------------------------
# BVH class
# ---------------------------------------------------------------------------

class BVH:
    """BVH motion parser that outputs MuJoCo qpos vectors.

    Port of MASS BVH class. The internal representation uses rotation matrices
    (matching DART), but output is in MuJoCo format (quaternion + translation).

    Also stores data in DART format for compatibility with Character class.

    Args:
        bvh_map: Mapping from skeleton body name -> BVH node name.
        joint_types: Joint type list in skeleton order.
        body_names: Body name list in skeleton order.
        revolute_axes: Mapping from body name -> revolute axis vector.
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

        # BVH hierarchy
        self.node_map: dict[str, BVHNode] = {}
        self.root: Optional[BVHNode] = None

        # Motion data
        self.motions: list[np.ndarray] = []
        self.time_step = 0.0
        self.num_total_channels = 0
        self.num_total_frames = 0

        # Root transforms (first and last frame)
        self.T0 = np.eye(4, dtype=np.float64)  # 4x4 homogeneous
        self.T1 = np.eye(4, dtype=np.float64)

    @property
    def max_time(self) -> float:
        """Total duration of the motion clip."""
        return self.num_total_frames * self.time_step

    def parse(self, filepath: str, cyclic: bool = True) -> None:
        """Parse a BVH file. Port of BVH::Parse."""
        self.cyclic = cyclic
        self.node_map.clear()
        self.motions.clear()

        with open(filepath, "r") as f:
            tokens = f.read().split()

        idx = 0
        while idx < len(tokens):
            if tokens[idx] == "HIERARCHY":
                idx += 1
                # ROOT or JOINT keyword
                idx += 1  # skip "ROOT"
                name = tokens[idx]
                idx += 1
                c_offset = [0]  # mutable counter
                self.root, idx = self._read_hierarchy(None, name, c_offset, tokens, idx)
                self.num_total_channels = c_offset[0]
            elif tokens[idx] == "MOTION":
                idx += 1
                # "Frames:"
                idx += 1
                self.num_total_frames = int(tokens[idx])
                idx += 1
                # "Frame" "Time:"
                idx += 2
                self.time_step = float(tokens[idx])
                idx += 1
                # Read frame data
                self.motions = []
                for i in range(self.num_total_frames):
                    frame = np.zeros(self.num_total_channels, dtype=np.float64)
                    for j in range(self.num_total_channels):
                        frame[j] = float(tokens[idx])
                        idx += 1
                    self.motions.append(frame)
            else:
                idx += 1

        # Compute T0 (first frame root transform)
        root_body = self.body_names[0]  # "Pelvis"
        root_bvh_name = self.bvh_map[root_body]

        m0 = self.motions[0]
        self.node_map[root_bvh_name].set_from_frame(m0)
        R0 = self.node_map[root_bvh_name].get()
        t0 = 0.01 * m0[:3]  # BVH cm -> meters
        self.T0 = _make_transform(R0, t0)

        # Compute T1 (last frame root transform)
        m_last = self.motions[self.num_total_frames - 1]
        self.node_map[root_bvh_name].set_from_frame(m_last)
        R1 = self.node_map[root_bvh_name].get()
        t1 = 0.01 * m_last[:3]
        self.T1 = _make_transform(R1, t1)

    def _read_hierarchy(
        self,
        parent: Optional[BVHNode],
        name: str,
        c_offset: list[int],
        tokens: list[str],
        idx: int,
    ) -> tuple[BVHNode, int]:
        """Recursively read BVH hierarchy. Port of BVH::ReadHierarchy."""
        node = BVHNode(name, parent)
        self.node_map[name] = node

        # skip "{"
        idx += 1

        while idx < len(tokens):
            tok = tokens[idx]
            if tok == "}":
                idx += 1
                break
            elif tok == "OFFSET":
                idx += 1
                # Read and ignore 3 offset values
                idx += 3
            elif tok == "CHANNELS":
                idx += 1
                n = int(tokens[idx])
                idx += 1
                c_names = []
                for _ in range(n):
                    c_names.append(tokens[idx])
                    idx += 1
                node.set_channel(c_offset[0], c_names)
                c_offset[0] += n
            elif tok == "JOINT":
                idx += 1
                child_name = tokens[idx]
                idx += 1
                child, idx = self._read_hierarchy(node, child_name, c_offset, tokens, idx)
                node.children.append(child)
            elif tok == "End":
                idx += 1  # skip "Site"
                child, idx = self._read_hierarchy(node, "EndEffector", c_offset, tokens, idx)
                node.children.append(child)
            else:
                idx += 1

        return node, idx

    def get_motion_dart(self, t: float) -> np.ndarray:
        """Get motion at time t in DART qpos format (exp-map).

        Port of BVH::GetMotion. Returns a DART-format qpos vector:
          FreeJoint: [logMap(R)(3), translation(3)]
          BallJoint: [logMap(R)(3)]
          RevoluteJoint: [angle(1)]

        This is needed by Character for transform computations.
        """
        # Frame selection (floor-based, with cyclic wrapping)
        k = int(math.floor(t / self.time_step))
        if self.cyclic:
            k = k % self.num_total_frames
        k = max(0, min(k, self.num_total_frames - 1))
        return self.get_motion_dart_frame(k)

    def get_motion_dart_frame(self, k: int) -> np.ndarray:
        """Get motion for an exact frame index in DART qpos format."""
        m_t = self.motions[k]

        # Set all BVH nodes from frame data
        for node in self.node_map.values():
            node.set_from_frame(m_t)

        # Build DART qpos
        dart_dof = sum(_DART_DOF[jt] for jt in self.joint_types)
        p = np.zeros(dart_dof, dtype=np.float64)

        di = 0  # DART index
        for i, body_name in enumerate(self.body_names):
            jt = self.joint_types[i]
            if body_name not in self.bvh_map:
                # No BVH mapping -> leave as zero
                di += _DART_DOF[jt]
                continue

            bvh_name = self.bvh_map[body_name]
            R = self.node_map[bvh_name].get()

            if jt == "Free":
                # FreeJoint: [logMap(R), 0.01*translation]
                log_r = _logmap(R)
                trans = 0.01 * m_t[:3]  # BVH cm -> meters
                p[di:di + 3] = log_r
                p[di + 3:di + 6] = trans
                di += 6
            elif jt == "Ball":
                # BallJoint: logMap(R)
                p[di:di + 3] = _logmap(R)
                di += 3
            elif jt == "Revolute":
                # RevoluteJoint: extract angle along axis
                aa = _logmap(R)
                axis = self.revolute_axes[body_name]
                if np.linalg.norm(axis - np.array([1, 0, 0])) < 1e-4:
                    val = aa[0]
                elif np.linalg.norm(axis - np.array([0, 1, 0])) < 1e-4:
                    val = aa[1]
                else:
                    val = aa[2]
                # Wrap to [-pi, pi]
                if val > math.pi:
                    val -= 2 * math.pi
                elif val < -math.pi:
                    val += 2 * math.pi
                p[di] = val
                di += 1

        return p

    def get_motion_mujoco(self, t: float) -> np.ndarray:
        """Get raw BVH motion at time t in MuJoCo qpos format.

        NOTE: This returns the raw BVH frame converted to MuJoCo format WITHOUT
        the Character class's T0/Tc root transform processing, so the root
        position is in BVH world coordinates. For the proper target position
        (with cyclic wrapping and root offset), use Character.get_target_positions_mujoco().

        The root_offset is applied here because the DART FreeJoint translation
        is in joint-local space, and MuJoCo freejoint expects world coordinates.
        """
        from rl_mass.joint_utils import dart_qpos_to_mujoco
        dart_q = self.get_motion_dart(t)
        return dart_qpos_to_mujoco(dart_q, root_offset=DART_ROOT_OFFSET)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def _test_bvh() -> None:
    """Load walk.bvh and print basic info."""
    import mujoco

    print("=" * 60)
    print("BVH Parser Test")
    print("=" * 60)

    bvh = BVH()
    bvh.parse("external/MASS/data/motion/walk.bvh", cyclic=True)

    print(f"  Frames: {bvh.num_total_frames}")
    print(f"  Time step: {bvh.time_step}")
    print(f"  Total channels: {bvh.num_total_channels}")
    print(f"  Max time: {bvh.max_time:.4f}")
    print(f"  T0 translation: {bvh.T0[:3, 3]}")
    print(f"  T1 translation: {bvh.T1[:3, 3]}")

    # Get DART qpos at t=0
    dart_q = bvh.get_motion_dart(0.0)
    print(f"\n  DART qpos at t=0 (first 12): {dart_q[:12]}")
    print(f"  DART qpos shape: {dart_q.shape}")

    # Get MuJoCo qpos at t=0
    mj_q = bvh.get_motion_mujoco(0.0)
    print(f"\n  MuJoCo qpos at t=0 (first 12): {mj_q[:12]}")
    print(f"  MuJoCo qpos shape: {mj_q.shape}")

    # Load MuJoCo model and set qpos
    model = mujoco.MjModel.from_xml_path("data/rl_mass/humanoid.xml")
    data = mujoco.MjData(model)
    data.qpos[:] = mj_q
    mujoco.mj_forward(model, data)

    # Print body positions to verify sanity
    print("\n  Body positions after mj_forward:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        pos = data.xpos[i]
        if name and name != "world":
            print(f"    {name:20s}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")

    # Verify root height is reasonable
    # NOTE: Raw BVH gives absolute mocap position. The BVH character's root is
    # at ~1.045m (BVH world) plus the DART joint offset (0.9809m), giving ~2.03m.
    # This is the raw BVH frame before Character class processing (T0/Tc normalization).
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
    pelvis_y = data.xpos[pelvis_id][1]
    print(f"\n  Pelvis Y position: {pelvis_y:.4f} (raw BVH + root offset)")
    assert 1.5 < pelvis_y < 2.5, f"Pelvis height {pelvis_y} seems wrong for raw BVH"
    print("  PASS: Pelvis height is reasonable for raw BVH")

    # Verify foot positions are near ground
    for foot_name in ["TalusR", "TalusL"]:
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
        foot_y = data.xpos[foot_id][1]
        print(f"  {foot_name} Y position: {foot_y:.4f}")

    # Test multiple time steps
    print("\n  Testing motion at various times:")
    for t in [0.0, 0.1, 0.5, 1.0, bvh.max_time - 0.01]:
        mj_q_t = bvh.get_motion_mujoco(t)
        data.qpos[:] = mj_q_t
        mujoco.mj_forward(model, data)
        py = data.xpos[pelvis_id][1]
        print(f"    t={t:.2f}: pelvis_y={py:.4f}")

    print("\n" + "=" * 60)
    print("BVH Parser Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _test_bvh()
