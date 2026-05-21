"""Joint mapping utilities for DART <-> MuJoCo conversions.

DART FreeJoint qpos:  [exp_map(3), translation(3)]  (rotation first)
MuJoCo freejoint qpos: [translation(3), quaternion(4)]  (translation first, quat w,x,y,z)

DART BallJoint:  exp_map(3)   (axis-angle, angle = ||v||)
MuJoCo ball:     quaternion(4)  (w, x, y, z)

RevoluteJoint:  same in both (1D angle)

Usage:
    from rl_mass.joint_utils import (
        dart_qpos_to_mujoco, mujoco_qpos_to_dart,
        expmap_to_quat_wxyz, quat_wxyz_to_expmap,
    )
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Joint types in skeleton traversal order (matches human.xml node order)
MASS_JOINT_TYPES: list[str] = [
    "Free",       # Pelvis (root)
    "Ball",       # FemurR
    "Revolute",   # TibiaR
    "Ball",       # TalusR
    "Revolute",   # FootThumbR
    "Revolute",   # FootPinkyR
    "Ball",       # FemurL
    "Revolute",   # TibiaL
    "Ball",       # TalusL
    "Revolute",   # FootThumbL
    "Revolute",   # FootPinkyL
    "Ball",       # Spine
    "Ball",       # Torso
    "Ball",       # Neck
    "Ball",       # Head
    "Ball",       # ShoulderR
    "Ball",       # ArmR
    "Revolute",   # ForeArmR
    "Ball",       # HandR
    "Ball",       # ShoulderL
    "Ball",       # ArmL
    "Revolute",   # ForeArmL
    "Ball",       # HandL
]

MASS_BODY_NAMES: list[str] = [
    "Pelvis",
    "FemurR", "TibiaR", "TalusR", "FootThumbR", "FootPinkyR",
    "FemurL", "TibiaL", "TalusL", "FootThumbL", "FootPinkyL",
    "Spine", "Torso", "Neck", "Head",
    "ShoulderR", "ArmR", "ForeArmR", "HandR",
    "ShoulderL", "ArmL", "ForeArmL", "HandL",
]

# Bodies marked as endeffector="True" in human.xml
MASS_END_EFFECTORS: list[str] = [
    "TalusR", "TalusL", "Head", "HandR", "HandL",
]

# Mapping from body name to BVH joint name (only those with bvh= attribute)
MASS_BVH_MAP: dict[str, str] = {
    "Pelvis": "Character1_Hips",
    "FemurR": "Character1_RightUpLeg",
    "TibiaR": "Character1_RightLeg",
    "TalusR": "Character1_RightFoot",
    "FemurL": "Character1_LeftUpLeg",
    "TibiaL": "Character1_LeftLeg",
    "TalusL": "Character1_LeftFoot",
    "Spine": "Character1_Spine",
    "Torso": "Character1_Spine1",
    "Neck": "Character1_Neck",
    "ShoulderR": "Character1_RightShoulder",
    "ArmR": "Character1_RightArm",
    "ForeArmR": "Character1_RightForeArm",
    "HandR": "Character1_RightHand",
    "ShoulderL": "Character1_LeftShoulder",
    "ArmL": "Character1_LeftArm",
    "ForeArmL": "Character1_LeftForeArm",
    "HandL": "Character1_LeftHand",
}

# Identity map for BVH files that already use MASS-native body names (the same
# names muscle284.xml and humanoid.xml use). Covers all 23 bodies, including
# Torso / FootPinky{L,R} / FootThumb{L,R} / Head which have no Mixamo
# counterpart in MASS_BVH_MAP. Pick this map when your retarget exports the
# MASS skeleton directly (e.g. Houdini /obj/motion_retarget/OUT_MASS_*).
MASS_NATIVE_BVH_MAP: dict[str, str] = {name: name for name in MASS_BODY_NAMES}

# DART DOFs per joint type
_DART_DOF = {"Free": 6, "Ball": 3, "Revolute": 1}
# MuJoCo qpos size per joint type
_MJ_QPOS = {"Free": 7, "Ball": 4, "Revolute": 1}

# Total DART qpos size
DART_QPOS_SIZE = sum(_DART_DOF[jt] for jt in MASS_JOINT_TYPES)  # 56
# Total MuJoCo qpos size
MJ_QPOS_SIZE = sum(_MJ_QPOS[jt] for jt in MASS_JOINT_TYPES)  # 71

# DART root joint's parent_to_joint translation.
# In DART, the FreeJoint position is joint-local; the world position is
# parent_to_joint * local_transform. For the MASS humanoid root (Pelvis),
# parent_to_joint = (I, (0, 0.9809, -0.0308)) from human.xml.
# MuJoCo freejoint directly stores world position, so we add this offset.
DART_ROOT_OFFSET = np.array([0.0, 0.9809, -0.0308], dtype=np.float64)


# ---------------------------------------------------------------------------
# Precomputed index tables for batched rotation operations
# ---------------------------------------------------------------------------
# These tables map between joint types and their DOF/qpos offsets, enabling
# batched scipy.Rotation calls instead of per-joint loops.

def _build_index_tables(
    joint_types: list[str],
) -> dict:
    """Build index tables for batched rotation conversions.

    Returns dict with:
        ball_dart_starts: DART qpos start indices for each BallJoint (N_ball,)
        ball_mj_starts: MuJoCo qpos start indices for each BallJoint (N_ball,)
        free_dart_idx: DART qpos start index for FreeJoint (or -1)
        free_mj_idx: MuJoCo qpos start index for FreeJoint (or -1)
        rev_dart_indices: DART qpos indices for RevoluteJoints
        rev_mj_indices: MuJoCo qpos indices for RevoluteJoints
    """
    ball_dart = []
    ball_mj = []
    rev_dart = []
    rev_mj = []
    free_dart = -1
    free_mj = -1

    di = 0  # DART offset
    mi = 0  # MuJoCo offset
    for jt in joint_types:
        if jt == "Free":
            free_dart = di
            free_mj = mi
            di += 6
            mi += 7
        elif jt == "Ball":
            ball_dart.append(di)
            ball_mj.append(mi)
            di += 3
            mi += 4
        elif jt == "Revolute":
            rev_dart.append(di)
            rev_mj.append(mi)
            di += 1
            mi += 1

    return {
        "ball_dart_starts": np.array(ball_dart, dtype=np.intp),
        "ball_mj_starts": np.array(ball_mj, dtype=np.intp),
        "free_dart_idx": free_dart,
        "free_mj_idx": free_mj,
        "rev_dart_indices": np.array(rev_dart, dtype=np.intp),
        "rev_mj_indices": np.array(rev_mj, dtype=np.intp),
        "n_ball": len(ball_dart),
    }

# Precomputed tables for the default MASS skeleton
_IDX = _build_index_tables(MASS_JOINT_TYPES)


# ---------------------------------------------------------------------------
# Rotation conversions (single-element, kept for external callers)
# ---------------------------------------------------------------------------

def expmap_to_quat_wxyz(expmap: np.ndarray) -> np.ndarray:
    """Convert DART exponential map (axis-angle 3D) to quaternion (w,x,y,z).

    The exponential map encodes rotation as a 3D vector where:
      - direction = rotation axis
      - magnitude = rotation angle (radians)
    """
    expmap = np.asarray(expmap, dtype=np.float64)
    angle = np.linalg.norm(expmap)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    r = Rotation.from_rotvec(expmap)
    q_xyzw = r.as_quat()  # scipy: (x, y, z, w)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quat_wxyz_to_expmap(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to DART exponential map (axis-angle 3D)."""
    q = np.asarray(q, dtype=np.float64)
    # scipy expects (x, y, z, w)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    # Normalize to avoid numerical issues
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    r = Rotation.from_quat(q_xyzw)
    return r.as_rotvec()


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z)."""
    R = np.asarray(R, dtype=np.float64)
    r = Rotation.from_matrix(R)
    q_xyzw = r.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


# ---------------------------------------------------------------------------
# Full qpos conversions
# ---------------------------------------------------------------------------

def dart_qpos_to_mujoco(
    dart_q: np.ndarray,
    joint_types: list[str] | None = None,
    root_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Convert DART full qpos vector to MuJoCo qpos.

    DART layout (per joint):
      FreeJoint:    [exp_map(3), translation(3)]   (joint-local translation)
      BallJoint:    [exp_map(3)]
      Revolute:     [angle(1)]

    MuJoCo layout (per joint):
      freejoint:    [translation(3), quaternion_wxyz(4)]  (world-frame translation)
      ball:         [quaternion_wxyz(4)]
      hinge:        [angle(1)]

    Args:
        dart_q: DART qpos vector.
        joint_types: Joint types in order.
        root_offset: DART root joint's parent_to_joint translation (3D).
            In DART, the FreeJoint stores joint-local positions, and the
            world position = parent_to_joint * local_transform. For the
            MASS humanoid, this is (0, 0.9809, -0.0308). If None, the
            DART translation is used directly (no offset).
    """
    if joint_types is None:
        joint_types = MASS_JOINT_TYPES

    dart_q = np.asarray(dart_q, dtype=np.float64)

    # Use precomputed index tables for default skeleton (fast path)
    idx = _IDX if joint_types is MASS_JOINT_TYPES else _build_index_tables(joint_types)
    n_ball = idx["n_ball"]
    ball_ds = idx["ball_dart_starts"]
    ball_ms = idx["ball_mj_starts"]
    free_di = idx["free_dart_idx"]
    free_mi = idx["free_mj_idx"]
    rev_ds = idx["rev_dart_indices"]
    rev_ms = idx["rev_mj_indices"]

    mj_size = sum(_MJ_QPOS[jt] for jt in joint_types)
    mj_q = np.zeros(mj_size, dtype=np.float64)

    # --- FreeJoint ---
    if free_di >= 0:
        expmap_free = dart_q[free_di : free_di + 3]
        trans = dart_q[free_di + 3 : free_di + 6]
        if root_offset is not None:
            mj_q[free_mi : free_mi + 3] = trans + root_offset
        else:
            mj_q[free_mi : free_mi + 3] = trans
        # Single rotvec -> quat (scalar, fast enough)
        q_xyzw = Rotation.from_rotvec(expmap_free).as_quat()
        mj_q[free_mi + 3] = q_xyzw[3]       # w
        mj_q[free_mi + 4 : free_mi + 7] = q_xyzw[:3]  # x,y,z

    # --- BallJoints (batched) ---
    if n_ball > 0:
        # Gather all ball exp-maps: (N_ball, 3)
        ball_expmaps = np.empty((n_ball, 3), dtype=np.float64)
        for k in range(n_ball):
            s = ball_ds[k]
            ball_expmaps[k] = dart_q[s : s + 3]

        # Batch convert rotvec -> quaternion (x,y,z,w)
        q_xyzw_batch = Rotation.from_rotvec(ball_expmaps).as_quat()  # (N_ball, 4)

        # Scatter to MuJoCo: [w, x, y, z]
        for k in range(n_ball):
            mi = ball_ms[k]
            mj_q[mi] = q_xyzw_batch[k, 3]         # w
            mj_q[mi + 1 : mi + 4] = q_xyzw_batch[k, :3]  # x,y,z

    # --- RevoluteJoints (direct copy) ---
    if len(rev_ds) > 0:
        mj_q[rev_ms] = dart_q[rev_ds]

    return mj_q


def mujoco_qpos_to_dart(
    mj_q: np.ndarray,
    joint_types: list[str] | None = None,
    root_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Convert MuJoCo full qpos vector to DART qpos.

    Reverse of dart_qpos_to_mujoco.

    Args:
        mj_q: MuJoCo qpos vector.
        joint_types: Joint types in order.
        root_offset: DART root joint's parent_to_joint translation (3D).
            If provided, subtracts it from the MuJoCo world position to get
            DART joint-local translation.
    """
    if joint_types is None:
        joint_types = MASS_JOINT_TYPES

    mj_q = np.asarray(mj_q, dtype=np.float64)

    idx = _IDX if joint_types is MASS_JOINT_TYPES else _build_index_tables(joint_types)
    n_ball = idx["n_ball"]
    ball_ds = idx["ball_dart_starts"]
    ball_ms = idx["ball_mj_starts"]
    free_di = idx["free_dart_idx"]
    free_mi = idx["free_mj_idx"]
    rev_ds = idx["rev_dart_indices"]
    rev_ms = idx["rev_mj_indices"]

    dart_size = sum(_DART_DOF[jt] for jt in joint_types)
    dart_q = np.zeros(dart_size, dtype=np.float64)

    # --- FreeJoint ---
    if free_di >= 0:
        trans = mj_q[free_mi : free_mi + 3].copy()
        quat_wxyz = mj_q[free_mi + 3 : free_mi + 7]
        # Convert wxyz -> scipy xyzw, then to rotvec
        q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        q_xyzw /= np.linalg.norm(q_xyzw)
        expmap = Rotation.from_quat(q_xyzw).as_rotvec()
        if root_offset is not None:
            trans = trans - root_offset
        dart_q[free_di : free_di + 3] = expmap
        dart_q[free_di + 3 : free_di + 6] = trans

    # --- BallJoints (batched) ---
    if n_ball > 0:
        # Gather all ball quaternions in scipy (x,y,z,w) order: (N_ball, 4)
        ball_quats = np.empty((n_ball, 4), dtype=np.float64)
        for k in range(n_ball):
            mi = ball_ms[k]
            # MuJoCo stores (w,x,y,z) -> scipy needs (x,y,z,w)
            ball_quats[k, 0] = mj_q[mi + 1]  # x
            ball_quats[k, 1] = mj_q[mi + 2]  # y
            ball_quats[k, 2] = mj_q[mi + 3]  # z
            ball_quats[k, 3] = mj_q[mi]      # w

        # Normalize (batch)
        norms = np.linalg.norm(ball_quats, axis=1, keepdims=True)
        ball_quats /= norms

        # Batch convert quat -> rotvec
        expmaps = Rotation.from_quat(ball_quats).as_rotvec()  # (N_ball, 3)

        # Scatter to DART
        for k in range(n_ball):
            s = ball_ds[k]
            dart_q[s : s + 3] = expmaps[k]

    # --- RevoluteJoints (direct copy) ---
    if len(rev_ds) > 0:
        dart_q[rev_ds] = mj_q[rev_ms]

    return dart_q


# ---------------------------------------------------------------------------
# DART-space arithmetic (for SPD controller fidelity)
# ---------------------------------------------------------------------------

def dart_position_differences(
    q2_dart: np.ndarray,
    q1_dart: np.ndarray,
    joint_types: list[str] | None = None,
) -> np.ndarray:
    """Compute DART getPositionDifferences(_q2, _q1) in exp-map space.

    For BallJoint: logMap(R1^T * R2)  where R1=expMap(q1), R2=expMap(q2)
    For FreeJoint rotation: same as BallJoint on the rotation part
    For FreeJoint translation: R1^T * (t2 - t1)
    For RevoluteJoint: q2 - q1

    This is equivalent to "q2 - q1" in the tangent space.

    Returns:
        Difference vector in DART qvel space (same dim as dart_q: 56).
    """
    if joint_types is None:
        joint_types = MASS_JOINT_TYPES

    q2 = np.asarray(q2_dart, dtype=np.float64)
    q1 = np.asarray(q1_dart, dtype=np.float64)
    result = np.zeros_like(q2)

    idx_tbl = _IDX if joint_types is MASS_JOINT_TYPES else _build_index_tables(joint_types)
    n_ball = idx_tbl["n_ball"]
    ball_ds = idx_tbl["ball_dart_starts"]
    free_di = idx_tbl["free_dart_idx"]
    rev_ds = idx_tbl["rev_dart_indices"]

    # Total number of rotational joints to batch (FreeJoint rot + all BallJoints)
    has_free = free_di >= 0
    n_rot = n_ball + (1 if has_free else 0)

    if n_rot > 0:
        # Gather all rotational exp-maps for q1 and q2
        e1_batch = np.empty((n_rot, 3), dtype=np.float64)
        e2_batch = np.empty((n_rot, 3), dtype=np.float64)

        k = 0
        if has_free:
            e1_batch[0] = q1[free_di : free_di + 3]
            e2_batch[0] = q2[free_di : free_di + 3]
            k = 1

        for i in range(n_ball):
            s = ball_ds[i]
            e1_batch[k + i] = q1[s : s + 3]
            e2_batch[k + i] = q2[s : s + 3]

        # Batch convert to rotation matrices: (n_rot, 3, 3)
        R1_batch = Rotation.from_rotvec(e1_batch).as_matrix()
        R2_batch = Rotation.from_rotvec(e2_batch).as_matrix()

        # R_diff = R1^T @ R2 for each joint: (n_rot, 3, 3)
        R_diff = np.transpose(R1_batch, (0, 2, 1)) @ R2_batch

        # Batch convert back to rotvec: (n_rot, 3)
        diff_batch = Rotation.from_matrix(R_diff).as_rotvec()

        # Scatter results
        k = 0
        if has_free:
            result[free_di : free_di + 3] = diff_batch[0]
            # FreeJoint translation: R1^T @ (t2 - t1)
            t1 = q1[free_di + 3 : free_di + 6]
            t2 = q2[free_di + 3 : free_di + 6]
            result[free_di + 3 : free_di + 6] = R1_batch[0].T @ (t2 - t1)
            k = 1

        for i in range(n_ball):
            s = ball_ds[i]
            result[s : s + 3] = diff_batch[k + i]

    # --- RevoluteJoints: direct subtraction ---
    if len(rev_ds) > 0:
        result[rev_ds] = q2[rev_ds] - q1[rev_ds]

    return result


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

def _test_roundtrip() -> None:
    """Run round-trip tests for all conversion functions."""
    print("=" * 60)
    print("Joint mapping round-trip tests")
    print("=" * 60)
    rng = np.random.default_rng(42)
    all_passed = True

    # --- Test 1: expmap <-> quaternion ---
    print("\n[Test 1] expmap <-> quat_wxyz round-trip")
    for i in range(10):
        # Random axis-angle with angle in [0, pi]
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, np.pi)
        expmap_orig = axis * angle

        quat = expmap_to_quat_wxyz(expmap_orig)
        expmap_back = quat_wxyz_to_expmap(quat)

        # Exponential map is not unique (v and -v can represent same rotation
        # when angle = pi), so compare via rotation matrices
        R_orig = Rotation.from_rotvec(expmap_orig).as_matrix()
        R_back = Rotation.from_rotvec(expmap_back).as_matrix()
        err = np.max(np.abs(R_orig - R_back))
        ok = err < 1e-10
        if not ok:
            print(f"  FAIL sample {i}: max error = {err:.2e}")
            all_passed = False
    if all_passed:
        print("  PASS (10 random samples, max rotmat error < 1e-10)")

    # --- Test 2: identity cases ---
    print("\n[Test 2] Identity / zero rotation")
    q_id = expmap_to_quat_wxyz(np.zeros(3))
    assert np.allclose(q_id, [1, 0, 0, 0]), f"Zero expmap should give identity quat, got {q_id}"
    em_back = quat_wxyz_to_expmap(q_id)
    assert np.allclose(em_back, [0, 0, 0], atol=1e-10), f"Identity quat should give zero expmap, got {em_back}"
    print("  PASS")

    # --- Test 3: rotmat_to_quat_wxyz ---
    print("\n[Test 3] rotmat_to_quat_wxyz")
    for i in range(5):
        R = Rotation.random(random_state=rng).as_matrix()
        q = rotmat_to_quat_wxyz(R)
        R_back = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        err = np.max(np.abs(R - R_back))
        ok = err < 1e-10
        if not ok:
            print(f"  FAIL sample {i}: max error = {err:.2e}")
            all_passed = False
    print("  PASS (5 random samples)")

    # --- Test 4: full DART <-> MuJoCo qpos round-trip ---
    print("\n[Test 4] dart_qpos_to_mujoco <-> mujoco_qpos_to_dart round-trip")

    # Generate a random but valid DART qpos
    dart_q_orig = np.zeros(DART_QPOS_SIZE)
    idx = 0
    for jt in MASS_JOINT_TYPES:
        if jt == "Free":
            # Random rotation (small angle to avoid pi ambiguity)
            dart_q_orig[idx : idx + 3] = rng.uniform(-1.0, 1.0, 3)
            # Random translation
            dart_q_orig[idx + 3 : idx + 6] = rng.uniform(-2.0, 2.0, 3)
            idx += 6
        elif jt == "Ball":
            dart_q_orig[idx : idx + 3] = rng.uniform(-1.0, 1.0, 3)
            idx += 3
        elif jt == "Revolute":
            dart_q_orig[idx] = rng.uniform(-1.5, 1.5)
            idx += 1

    mj_q = dart_qpos_to_mujoco(dart_q_orig)
    dart_q_back = mujoco_qpos_to_dart(mj_q)

    # Compare via rotation matrices (expmap not unique for angle >= pi)
    idx_orig = 0
    idx_back = 0
    max_rot_err = 0.0
    max_trans_err = 0.0
    for jt in MASS_JOINT_TYPES:
        if jt == "Free":
            R_orig = Rotation.from_rotvec(dart_q_orig[idx_orig : idx_orig + 3]).as_matrix()
            R_back = Rotation.from_rotvec(dart_q_back[idx_back : idx_back + 3]).as_matrix()
            max_rot_err = max(max_rot_err, np.max(np.abs(R_orig - R_back)))
            max_trans_err = max(max_trans_err, np.max(np.abs(
                dart_q_orig[idx_orig + 3 : idx_orig + 6] - dart_q_back[idx_back + 3 : idx_back + 6]
            )))
            idx_orig += 6
            idx_back += 6
        elif jt == "Ball":
            R_orig = Rotation.from_rotvec(dart_q_orig[idx_orig : idx_orig + 3]).as_matrix()
            R_back = Rotation.from_rotvec(dart_q_back[idx_back : idx_back + 3]).as_matrix()
            max_rot_err = max(max_rot_err, np.max(np.abs(R_orig - R_back)))
            idx_orig += 3
            idx_back += 3
        elif jt == "Revolute":
            max_trans_err = max(max_trans_err, abs(dart_q_orig[idx_orig] - dart_q_back[idx_back]))
            idx_orig += 1
            idx_back += 1

    ok = max_rot_err < 1e-10 and max_trans_err < 1e-10
    if ok:
        print(f"  PASS (max rot error: {max_rot_err:.2e}, max trans/angle error: {max_trans_err:.2e})")
    else:
        print(f"  FAIL (max rot error: {max_rot_err:.2e}, max trans/angle error: {max_trans_err:.2e})")
        all_passed = False

    # --- Test 5: MuJoCo -> DART -> MuJoCo round-trip ---
    print("\n[Test 5] mujoco_qpos_to_dart <-> dart_qpos_to_mujoco round-trip")

    mj_q_orig = np.zeros(MJ_QPOS_SIZE)
    mi = 0
    for jt in MASS_JOINT_TYPES:
        if jt == "Free":
            mj_q_orig[mi : mi + 3] = rng.uniform(-2.0, 2.0, 3)
            # Generate valid quaternion
            q = Rotation.random(random_state=rng).as_quat()  # x,y,z,w
            mj_q_orig[mi + 3] = q[3]  # w
            mj_q_orig[mi + 4 : mi + 7] = q[:3]  # x,y,z
            mi += 7
        elif jt == "Ball":
            q = Rotation.random(random_state=rng).as_quat()
            mj_q_orig[mi] = q[3]
            mj_q_orig[mi + 1 : mi + 4] = q[:3]
            mi += 4
        elif jt == "Revolute":
            mj_q_orig[mi] = rng.uniform(-1.5, 1.5)
            mi += 1

    dart_q = mujoco_qpos_to_dart(mj_q_orig)
    mj_q_back = dart_qpos_to_mujoco(dart_q)

    # Compare quaternions: allow sign flip (q and -q are same rotation)
    mi_orig = 0
    mi_back = 0
    max_err = 0.0
    for jt in MASS_JOINT_TYPES:
        if jt == "Free":
            max_err = max(max_err, np.max(np.abs(mj_q_orig[mi_orig : mi_orig + 3] - mj_q_back[mi_back : mi_back + 3])))
            q1 = mj_q_orig[mi_orig + 3 : mi_orig + 7]
            q2 = mj_q_back[mi_back + 3 : mi_back + 7]
            # Quaternions q and -q are equivalent
            err_q = min(np.max(np.abs(q1 - q2)), np.max(np.abs(q1 + q2)))
            max_err = max(max_err, err_q)
            mi_orig += 7
            mi_back += 7
        elif jt == "Ball":
            q1 = mj_q_orig[mi_orig : mi_orig + 4]
            q2 = mj_q_back[mi_back : mi_back + 4]
            err_q = min(np.max(np.abs(q1 - q2)), np.max(np.abs(q1 + q2)))
            max_err = max(max_err, err_q)
            mi_orig += 4
            mi_back += 4
        elif jt == "Revolute":
            max_err = max(max_err, abs(mj_q_orig[mi_orig] - mj_q_back[mi_back]))
            mi_orig += 1
            mi_back += 1

    ok = max_err < 1e-10
    if ok:
        print(f"  PASS (max error: {max_err:.2e})")
    else:
        print(f"  FAIL (max error: {max_err:.2e})")
        all_passed = False

    # --- Test 6: Size sanity ---
    print("\n[Test 6] Size constants")
    assert DART_QPOS_SIZE == 56, f"DART_QPOS_SIZE should be 56, got {DART_QPOS_SIZE}"
    assert MJ_QPOS_SIZE == 71, f"MJ_QPOS_SIZE should be 71, got {MJ_QPOS_SIZE}"
    assert len(MASS_BODY_NAMES) == 23, f"Expected 23 bodies, got {len(MASS_BODY_NAMES)}"
    assert len(MASS_JOINT_TYPES) == 23, f"Expected 23 joint types, got {len(MASS_JOINT_TYPES)}"
    print(f"  PASS (DART_QPOS_SIZE={DART_QPOS_SIZE}, MJ_QPOS_SIZE={MJ_QPOS_SIZE}, bodies={len(MASS_BODY_NAMES)})")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    _test_roundtrip()
