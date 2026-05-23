"""Muscle model ported from MASS (external/MASS/core/Muscle.cpp).

This is a faithful Python port of the MASS hill-type muscle model.
All force curves and jacobian computations match the C++ implementation.

Usage:
    from rl_mass.engine import MuJoCoEngine
    from rl_mass.muscle import parse_muscle_xml

    engine = MuJoCoEngine("data/rl_mass/humanoid.xml", timestep=1/600)
    engine.forward()
    muscles = parse_muscle_xml("external/MASS/data/muscle284.xml", engine)
"""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from math import exp, sqrt

import numpy as np

from rl_mass.engine import PhysicsEngine


# ---------------------------------------------------------------------------
# Force curves (pure math, exact from Muscle.cpp)
# ---------------------------------------------------------------------------

def g_al(l_m_norm: float) -> float:
    """Active force-length curve. gamma=0.5"""
    return exp(-(l_m_norm - 1.0) ** 2 / 0.5)


def g_pl(l_m_norm: float) -> float:
    """Passive force-length curve. k_pe=4.0, e_mo=0.6"""
    if l_m_norm < 1.0:
        return 0.0
    return (exp(4.0 * (l_m_norm - 1.0) / 0.6) - 1.0) / (exp(4.0) - 1.0)


def g_t(e_t: float) -> float:
    """Tendon force curve. f_toe=0.33, k_toe=3.0, k_lin=51.878788, e_toe=0.02, e_t0=0.033"""
    if e_t <= 0.033:
        return 0.33 / (exp(3.0) - 1) * (exp(3.0 * e_t / 0.02) - 1)
    else:
        return 51.878788 * (e_t - 0.02) + 0.33


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MuscleAnchor:
    """A waypoint (anchor) of a muscle, potentially blended across bodies (LBS)."""
    body_ids: list[int]  # MuJoCo body indices
    local_positions: list[np.ndarray]  # (3,) in each body's local frame
    weights: list[float]  # LBS weights, sum to 1
    num_related_bodies: int = 0

    def __post_init__(self):
        self.num_related_bodies = len(self.body_ids)

    def get_point(self, engine: PhysicsEngine) -> np.ndarray:
        """Compute world-space position via LBS."""
        p = np.zeros(3)
        for i in range(self.num_related_bodies):
            bid = self.body_ids[i]
            R = engine.get_body_xmat(bid)
            t = engine.get_body_xpos(bid)
            p += self.weights[i] * (R @ self.local_positions[i] + t)
        return p


@dataclass
class Muscle:
    """Hill-type muscle model, ported from MASS Muscle.cpp."""
    name: str
    f0: float  # maximum isometric force
    l_m0: float  # optimal muscle fiber length
    l_t0: float  # tendon slack length
    pen_angle: float  # pennation angle (unused in MASS force computation)
    l_mt_max: float  # maximum muscle-tendon length

    anchors: list[MuscleAnchor] = field(default_factory=list)
    cached_anchor_positions: list[np.ndarray] = field(default_factory=list)

    num_related_dofs: int = 0
    related_dof_indices: list[int] = field(default_factory=list)

    activation: float = 0.0

    l_mt0: float = 0.0  # rest-pose muscle-tendon length (accumulated during loading)
    l_mt: float = 1.0  # normalized muscle-tendon length
    l_m: float = 0.0  # muscle fiber length (not normalized)

    # Constants (from Muscle.cpp constructor)
    f_toe: float = 0.33
    k_toe: float = 3.0
    k_lin: float = 51.878788
    e_toe: float = 0.02
    e_t0: float = 0.033
    k_pe: float = 4.0
    e_mo: float = 0.6
    gamma: float = 0.5

    def update(self, engine: PhysicsEngine) -> None:
        """Recompute anchor positions, l_mt (normalized), and l_m."""
        for i, anchor in enumerate(self.anchors):
            self.cached_anchor_positions[i] = anchor.get_point(engine)

        # Compute normalized muscle-tendon length
        # Matches Muscle::Getl_mt() in Muscle.cpp
        self.l_mt = 0.0
        for i in range(1, len(self.anchors)):
            self.l_mt += np.linalg.norm(
                self.cached_anchor_positions[i] - self.cached_anchor_positions[i - 1]
            )
        if self.l_mt0 > 0.0:
            self.l_mt /= self.l_mt0  # normalize
        else:
            self.l_mt = 1.0  # only 1 anchor, keep default

        # Muscle fiber length = total length - tendon slack (rigid tendon model)
        # Note: l_mt is normalized but l_t0 is absolute.
        # In MASS: l_m = l_mt - l_t0, where l_mt is the normalized value.
        self.l_m = self.l_mt - self.l_t0

    def get_f_A(self) -> float:
        """Active force: f0 * g_al(l_m / l_m0)."""
        return self.f0 * g_al(self.l_m / self.l_m0)

    def get_f_p(self) -> float:
        """Passive force: f0 * g_pl(l_m / l_m0)."""
        return self.f0 * g_pl(self.l_m / self.l_m0)

    def get_force(self) -> float:
        """Total muscle force: f_A * activation + f_p."""
        return self.get_f_A() * self.activation + self.get_f_p()

    def apply_force_to_body(self, engine: PhysicsEngine) -> None:
        """Apply muscle force along the muscle line to each anchor body.

        Matches Muscle::ApplyForceToBody in Muscle.cpp:
        - For each segment (i, i+1): apply force in direction (i+1 - i) to anchor i
        - For each segment (i-1, i): apply force in direction (i-1 - i) to anchor i
        Forces are applied via engine.apply_force_and_torque (force at body COM + torque
        from moment arm).
        """
        f = self.get_force()
        n = len(self.anchors)
        pos = self.cached_anchor_positions

        # Forward pass: anchor i pulls toward anchor i+1
        for i in range(n - 1):
            direction = pos[i + 1] - pos[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction /= norm
            force_vec = f * direction
            bid = self.anchors[i].body_ids[0]
            body_com = engine.get_body_com(bid)
            torque_vec = np.cross(pos[i] - body_com, force_vec)
            engine.apply_force_and_torque(bid, force_vec, torque_vec)

        # Backward pass: anchor i pulls toward anchor i-1
        for i in range(1, n):
            direction = pos[i - 1] - pos[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction /= norm
            force_vec = f * direction
            bid = self.anchors[i].body_ids[0]
            body_com = engine.get_body_com(bid)
            torque_vec = np.cross(pos[i] - body_com, force_vec)
            engine.apply_force_and_torque(bid, force_vec, torque_vec)

    def get_jacobian_transpose(self, engine: PhysicsEngine) -> np.ndarray:
        """Compute Jt of shape (nv, 3*n_anchors).

        For each anchor, use engine.get_linear_jacobian to get the
        translational Jacobian at the anchor's world-space position
        for the primary body.
        Matches Muscle::GetJacobianTranspose in Muscle.cpp.
        """
        nv = engine.nv
        n_anchors = len(self.anchors)
        Jt = np.zeros((nv, 3 * n_anchors))

        for i, anchor in enumerate(self.anchors):
            point = self.cached_anchor_positions[i]
            bid = anchor.body_ids[0]
            jacp = engine.get_linear_jacobian(point, bid)
            # jacp shape: (3, nv), Jt = jacp.T -> (nv, 3)
            Jt[:, i * 3 : (i + 1) * 3] = jacp.T

        return Jt

    def get_force_jacobian_and_passive(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute (A, p) each of shape (3*n_anchors,).

        A[i*3:(i+1)*3] = force_direction[i] * f_A
        p[i*3:(i+1)*3] = force_direction[i] * f_p

        force_direction is computed the same as in apply_force_to_body:
        each anchor accumulates directions from adjacent segments.
        Matches Muscle::GetForceJacobianAndPassive in Muscle.cpp.
        """
        f_a = self.get_f_A()
        f_p = self.get_f_p()
        n = len(self.anchors)
        pos = self.cached_anchor_positions

        force_dir = [np.zeros(3) for _ in range(n)]

        # Forward: anchor i -> direction toward i+1
        for i in range(n - 1):
            d = pos[i + 1] - pos[i]
            norm = np.linalg.norm(d)
            if norm > 1e-12:
                d /= norm
            force_dir[i] = force_dir[i] + d

        # Backward: anchor i -> direction toward i-1
        for i in range(1, n):
            d = pos[i - 1] - pos[i]
            norm = np.linalg.norm(d)
            if norm > 1e-12:
                d /= norm
            force_dir[i] = force_dir[i] + d

        A = np.zeros(3 * n)
        p = np.zeros(3 * n)
        for i in range(n):
            A[i * 3 : (i + 1) * 3] = force_dir[i] * f_a
            p[i * 3 : (i + 1) * 3] = force_dir[i] * f_p

        return A, p

    def get_related_jta(self, engine: PhysicsEngine) -> np.ndarray:
        """Compute JtA and extract related DOF indices.

        Returns compressed vector of length num_related_dofs.
        Matches Muscle::GetRelatedJtA in Muscle.cpp.
        """
        Jt = self.get_jacobian_transpose(engine)
        A, _ = self.get_force_jacobian_and_passive()
        JtA = Jt @ A

        result = np.zeros(self.num_related_dofs)
        for i, idx in enumerate(self.related_dof_indices):
            result[i] = JtA[idx]
        return result


# ---------------------------------------------------------------------------
# Vectorized muscle system
# ---------------------------------------------------------------------------


class MuscleSystem:
    """Vectorized manager for all muscles.

    Replaces per-muscle Python loops with batched NumPy operations for
    update(), force computation, and force application.

    Anchor positions are computed via LBS (linear blend skinning).
    All anchors are padded to max_bodies=2 so that both single-body and
    LBS anchors can be processed in one batch.

    Layout:
        - Each muscle has a contiguous range of anchors in the flat arrays.
        - muscle_anchor_start[m] / muscle_anchor_end[m] give the range.
        - Segment pairs (consecutive anchors) are precomputed for l_mt and
          force direction.
    """

    def __init__(self, muscles: list[Muscle]):
        n_muscles = len(muscles)
        self.n_muscles = n_muscles

        # --- Precompute per-muscle scalar data ---
        self.f0 = np.array([m.f0 for m in muscles], dtype=np.float64)
        self.l_m0 = np.array([m.l_m0 for m in muscles], dtype=np.float64)
        self.l_t0 = np.array([m.l_t0 for m in muscles], dtype=np.float64)
        self.l_mt0 = np.array([m.l_mt0 for m in muscles], dtype=np.float64)

        # --- Precompute anchor arrays (padded to 2 bodies) ---
        total_anchors = sum(len(m.anchors) for m in muscles)
        self.total_anchors = total_anchors

        # Per-anchor: 2 body IDs, 2 local positions, 2 weights
        anchor_body_ids = np.zeros((total_anchors, 2), dtype=np.int32)
        anchor_local_pos = np.zeros((total_anchors, 2, 3), dtype=np.float64)
        anchor_weights = np.zeros((total_anchors, 2), dtype=np.float64)

        # Per-anchor: primary body ID (for force application)
        anchor_primary_bid = np.zeros(total_anchors, dtype=np.int32)

        # Per-muscle: anchor range
        muscle_anchor_start = np.zeros(n_muscles, dtype=np.int32)
        muscle_anchor_end = np.zeros(n_muscles, dtype=np.int32)
        muscle_n_anchors = np.zeros(n_muscles, dtype=np.int32)

        idx = 0
        for m_i, muscle in enumerate(muscles):
            muscle_anchor_start[m_i] = idx
            for anchor in muscle.anchors:
                anchor_primary_bid[idx] = anchor.body_ids[0]
                nb = anchor.num_related_bodies
                for b in range(nb):
                    anchor_body_ids[idx, b] = anchor.body_ids[b]
                    anchor_local_pos[idx, b] = anchor.local_positions[b]
                    anchor_weights[idx, b] = anchor.weights[b]
                # Pad second slot if single-body (weight stays 0, pos stays 0)
                if nb == 1:
                    anchor_body_ids[idx, 1] = anchor.body_ids[0]  # dummy
                idx += 1
            muscle_anchor_end[m_i] = idx
            muscle_n_anchors[m_i] = len(muscle.anchors)

        self.anchor_body_ids = anchor_body_ids
        self.anchor_local_pos = anchor_local_pos
        self.anchor_weights = anchor_weights
        self.anchor_primary_bid = anchor_primary_bid
        self.muscle_anchor_start = muscle_anchor_start
        self.muscle_anchor_end = muscle_anchor_end
        self.muscle_n_anchors = muscle_n_anchors

        # --- Precompute segment pairs for l_mt and force directions ---
        # Segments connect consecutive anchors within each muscle.
        # seg_a[s], seg_b[s] are flat anchor indices for segment s.
        seg_a_list = []
        seg_b_list = []
        seg_muscle_list = []  # which muscle each segment belongs to
        for m_i in range(n_muscles):
            start = muscle_anchor_start[m_i]
            n = muscle_n_anchors[m_i]
            for k in range(n - 1):
                seg_a_list.append(start + k)
                seg_b_list.append(start + k + 1)
                seg_muscle_list.append(m_i)

        self.seg_a = np.array(seg_a_list, dtype=np.int32)
        self.seg_b = np.array(seg_b_list, dtype=np.int32)
        self.seg_muscle = np.array(seg_muscle_list, dtype=np.int32)
        self.n_segments = len(seg_a_list)

        # --- Anchor world positions (updated each step) ---
        self.anchor_world_pos = np.zeros((total_anchors, 3), dtype=np.float64)

        # --- Per-muscle state ---
        self.l_mt = np.ones(n_muscles, dtype=np.float64)
        self.l_m = np.zeros(n_muscles, dtype=np.float64)

        # --- Related DOFs (for get_muscle_torques) ---
        self.related_dof_indices = [m.related_dof_indices for m in muscles]
        self.num_related_dofs = [m.num_related_dofs for m in muscles]

        # --- Body-grouped anchor index for batched mj_jacBodyCom path ---
        # Each anchor lives on one primary body. Group anchors by body so we
        # call mj_jacBodyCom once per unique body (~tens) instead of mj_jac
        # per anchor (~thousands). Then reuse the (jacp_com, jacr) pair to
        # project all anchors on that body via the wrench-at-COM identity.
        anchor_muscle_id = np.zeros(total_anchors, dtype=np.int32)
        for m_i in range(n_muscles):
            s = muscle_anchor_start[m_i]
            n = muscle_n_anchors[m_i]
            anchor_muscle_id[s : s + n] = m_i
        self.anchor_muscle_id = anchor_muscle_id

        unique_bodies, inverse = np.unique(anchor_primary_bid, return_inverse=True)
        self.unique_bodies = unique_bodies.astype(np.int32)
        # body_anchor_lists[k] = anchor indices whose primary body == unique_bodies[k]
        self.body_anchor_lists = [
            np.where(inverse == k)[0].astype(np.int32)
            for k in range(len(unique_bodies))
        ]

        # related_dof_indices_arr[m] as np.int32 array for fast fancy indexing
        self.related_dof_indices_arr = [
            np.asarray(idx, dtype=np.int32) for idx in self.related_dof_indices
        ]

    def update_all(self, engine: PhysicsEngine) -> None:
        """Update all anchor world positions, l_mt, and l_m in batch."""
        data = engine.data  # type: ignore[attr-defined]

        # --- Batch anchor world positions via LBS ---
        # R[i] = xmat for body i, shape (nbody, 3, 3)
        # t[i] = xpos for body i, shape (nbody, 3)
        xmat_all = data.xmat.reshape(-1, 3, 3)  # (nbody, 3, 3)
        xpos_all = data.xpos  # (nbody, 3)

        # Gather transforms for each anchor's 2 body slots
        bids_0 = self.anchor_body_ids[:, 0]  # (total_anchors,)
        bids_1 = self.anchor_body_ids[:, 1]

        R0 = xmat_all[bids_0]  # (total_anchors, 3, 3)
        t0 = xpos_all[bids_0]  # (total_anchors, 3)
        R1 = xmat_all[bids_1]
        t1 = xpos_all[bids_1]

        lp0 = self.anchor_local_pos[:, 0, :]  # (total_anchors, 3)
        lp1 = self.anchor_local_pos[:, 1, :]

        w0 = self.anchor_weights[:, 0:1]  # (total_anchors, 1)
        w1 = self.anchor_weights[:, 1:2]

        # World pos = w0 * (R0 @ lp0 + t0) + w1 * (R1 @ lp1 + t1)
        # einsum for batched matmul: (N,3,3) @ (N,3) -> (N,3)
        wp0 = np.einsum('nij,nj->ni', R0, lp0) + t0
        wp1 = np.einsum('nij,nj->ni', R1, lp1) + t1
        self.anchor_world_pos[:] = w0 * wp0 + w1 * wp1

        # --- Batch l_mt computation ---
        # For each segment, compute distance between consecutive anchors
        pa = self.anchor_world_pos[self.seg_a]  # (n_segments, 3)
        pb = self.anchor_world_pos[self.seg_b]
        seg_lengths = np.linalg.norm(pb - pa, axis=1)  # (n_segments,)

        # Sum segment lengths per muscle using np.add.at
        self.l_mt[:] = 0.0
        np.add.at(self.l_mt, self.seg_muscle, seg_lengths)

        # Normalize by l_mt0 (avoid division by zero)
        valid = self.l_mt0 > 0.0
        self.l_mt[valid] /= self.l_mt0[valid]
        self.l_mt[~valid] = 1.0

        # Muscle fiber length
        self.l_m = self.l_mt - self.l_t0

    def apply_forces_all(
        self, engine: PhysicsEngine, activations: np.ndarray
    ) -> None:
        """Compute and apply all muscle forces at once.

        This replaces both get_force() and apply_force_to_body() for every
        muscle in a single vectorized pass.
        """
        data = engine.data  # type: ignore[attr-defined]
        xipos_all = data.xipos  # (nbody, 3) — body COMs

        # --- Compute forces per muscle ---
        l_m_norm = self.l_m / self.l_m0  # (n_muscles,)

        # Vectorized g_al: exp(-(x-1)^2 / 0.5)
        f_active = np.exp(-((l_m_norm - 1.0) ** 2) / 0.5)

        # Vectorized g_pl: piecewise
        f_passive = np.zeros(self.n_muscles, dtype=np.float64)
        mask = l_m_norm >= 1.0
        if np.any(mask):
            f_passive[mask] = (
                np.exp(4.0 * (l_m_norm[mask] - 1.0) / 0.6) - 1.0
            ) / (np.exp(4.0) - 1.0)

        # Total force per muscle: f0 * (g_al * activation + g_pl)
        muscle_force = self.f0 * (f_active * activations + f_passive)

        # --- Compute segment directions ---
        pa = self.anchor_world_pos[self.seg_a]  # (n_segments, 3)
        pb = self.anchor_world_pos[self.seg_b]
        diff = pb - pa  # (n_segments, 3)
        norms = np.linalg.norm(diff, axis=1, keepdims=True)  # (n_segments, 1)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        directions = diff / norms  # (n_segments, 3), unit vectors a->b

        # Force magnitude per segment (from the muscle it belongs to)
        seg_force = muscle_force[self.seg_muscle]  # (n_segments,)
        force_vecs = directions * seg_force[:, None]  # (n_segments, 3)

        # --- Apply forces to bodies ---
        # Clear the xfrc_applied buffer (caller should have already done this,
        # but we accumulate into it)
        # Forward pass: anchor seg_a[s] gets force in direction a->b
        # Backward pass: anchor seg_b[s] gets force in direction b->a

        # Get body IDs and COMs for force/torque computation
        bid_a = self.anchor_primary_bid[self.seg_a]  # (n_segments,)
        bid_b = self.anchor_primary_bid[self.seg_b]
        com_a = xipos_all[bid_a]  # (n_segments, 3)
        com_b = xipos_all[bid_b]
        pos_a = self.anchor_world_pos[self.seg_a]  # (n_segments, 3)
        pos_b = self.anchor_world_pos[self.seg_b]

        # Forward: force on anchor a in direction a->b
        torque_a = np.cross(pos_a - com_a, force_vecs)  # (n_segments, 3)

        # Backward: force on anchor b in direction b->a = -force_vecs
        neg_force_vecs = -force_vecs
        torque_b = np.cross(pos_b - com_b, neg_force_vecs)

        # Accumulate forces and torques per body using np.add.at
        np.add.at(data.xfrc_applied[:, :3], bid_a, force_vecs)
        np.add.at(data.xfrc_applied[:, 3:], bid_a, torque_a)
        np.add.at(data.xfrc_applied[:, :3], bid_b, neg_force_vecs)
        np.add.at(data.xfrc_applied[:, 3:], bid_b, torque_b)

    def get_all_muscle_torques(
        self, engine: PhysicsEngine
    ) -> np.ndarray:
        """Compute concatenated related JtA for all muscles.

        Body-grouped fast path. For each unique primary body of all anchors,
        call mj_jacBodyCom once (jacp at xipos, jacr). For any anchor at
        world point p on body b with applied force w, the JtA contribution is
            jac(p,b).T @ w  ==  jacp_com.T @ w + jacr.T @ (r × w)
        where r = p - xipos[b]. This avoids the per-anchor mj_jac call (and
        per-muscle explicit Jt matrix) of the old path. Reduces mj_jac-style
        calls from O(total_anchors × muscle_substeps) to O(unique_bodies ×
        muscle_substeps).
        """
        data = engine.data  # type: ignore[attr-defined]
        nv = engine.nv

        # --- Per-anchor force vector w_a = f_a[muscle(a)] * force_dir[a] ---
        # f_a (active force coefficient) is per muscle, same formula as
        # apply_forces_all / old get_all_muscle_torques.
        l_m_norm = self.l_m / self.l_m0
        f_a_muscle = self.f0 * np.exp(-((l_m_norm - 1.0) ** 2) / 0.5)

        # Per-anchor accumulated unit direction (forward outgoing + backward
        # incoming segment dirs). Same accumulation pattern as
        # _collect_muscle_tuple.force_dir_all.
        diff = self.anchor_world_pos[self.seg_b] - self.anchor_world_pos[self.seg_a]
        seg_norm = np.linalg.norm(diff, axis=1)
        seg_dir = diff.copy()
        nz = seg_norm > 1e-12
        seg_dir[nz] = diff[nz] / seg_norm[nz, None]
        force_dir_all = np.zeros((self.total_anchors, 3), dtype=np.float64)
        np.add.at(force_dir_all, self.seg_a, seg_dir)
        np.add.at(force_dir_all, self.seg_b, -seg_dir)

        f_a_per_anchor = f_a_muscle[self.anchor_muscle_id]
        wrench_F = force_dir_all * f_a_per_anchor[:, None]  # (total_anchors, 3)

        # --- Per-anchor torque about its primary body's COM (xipos) ---
        # r = anchor_world_pos - xipos[primary_bid]; T = r × F
        bids = self.anchor_primary_bid
        r = self.anchor_world_pos - data.xipos[bids]
        wrench_T = np.cross(r, wrench_F)  # (total_anchors, 3)

        # --- Accumulate JtA per muscle by iterating over unique bodies ---
        JtA_per_muscle = np.zeros((self.n_muscles, nv), dtype=np.float64)
        for k, bid in enumerate(self.unique_bodies):
            jacp_com, jacr = engine.get_body_com_jacobians(int(bid))  # (3, nv) each
            anc_idx = self.body_anchor_lists[k]  # anchors on this body
            # contrib[i,:] = wrench_F[anc].T @ jacp_com + wrench_T[anc].T @ jacr
            # i.e. for each anchor row, project wrench through this body's
            # jacobians. Vectorized over anchors on this body.
            contribs = wrench_F[anc_idx] @ jacp_com + wrench_T[anc_idx] @ jacr  # (k, nv)
            np.add.at(JtA_per_muscle, self.anchor_muscle_id[anc_idx], contribs)

        # --- Flatten to concatenated related-DOF result ---
        total_related = sum(self.num_related_dofs)
        result = np.empty(total_related, dtype=np.float64)
        idx = 0
        for m_i in range(self.n_muscles):
            rel = self.related_dof_indices_arr[m_i]
            n_rel = rel.size
            result[idx : idx + n_rel] = JtA_per_muscle[m_i, rel]
            idx += n_rel

        return result


# ---------------------------------------------------------------------------
# Muscle loading (from muscle284.xml)
# ---------------------------------------------------------------------------

def _compute_joint_frame_pos(engine: PhysicsEngine, body_id: int) -> np.ndarray:
    """Compute the joint frame position for a body.

    In DART: T = body.getTransform() * parentJoint.getTransformFromChildBodyNode()
    This gives the joint frame position in world space.

    For bodies with joints, we use the first joint's anchor position in
    world frame. For bodies without joints, we use the body position.
    """
    jnt_num = engine.body_jnt_num(body_id)

    if jnt_num == 0:
        return engine.get_body_xpos(body_id)

    jnt_id = engine.body_jnt_addr(body_id)
    return engine.get_joint_anchor(jnt_id)


def _add_anchor_single(
    muscle: Muscle,
    engine: PhysicsEngine,
    body_name: str,
    glob_pos: np.ndarray,
) -> None:
    """Add an anchor bound to a single body (first/last waypoint).

    Matches Muscle::AddAnchor(BodyNode* bn, const Vector3d& glob_pos).
    """
    bid = engine.body_id(body_name)
    if bid < 0:
        raise ValueError(f"Body '{body_name}' not found in model")

    R = engine.get_body_xmat(bid)
    t = engine.get_body_xpos(bid)
    local_pos = R.T @ (glob_pos - t)

    anchor = MuscleAnchor(
        body_ids=[bid],
        local_positions=[local_pos],
        weights=[1.0],
    )
    muscle.anchors.append(anchor)
    _finalize_anchor(muscle, engine)


def _add_anchor_lbs(
    muscle: Muscle,
    engine: PhysicsEngine,
    body_name: str,
    glob_pos: np.ndarray,
) -> None:
    """Add an anchor with LBS weighting (middle waypoints).

    Matches Muscle::AddAnchor(skel, bn, glob_pos, num_related_bodies).
    - Compute distance from glob_pos to each body's joint frame
    - Find nearest body by distance
    - If nearest < 0.08: LBS with nearest + its parent
    - Else: bind to specified body
    """
    nbody = engine.nbody

    # Compute distance to each body's joint frame and local positions
    distances = np.zeros(nbody)
    local_positions = [None] * nbody

    for i in range(nbody):
        R_i = engine.get_body_xmat(i)
        t_i = engine.get_body_xpos(i)
        # Local position in body i's frame (for potential LBS)
        local_positions[i] = R_i.T @ (glob_pos - t_i)

        # Distance to joint frame
        joint_pos = _compute_joint_frame_pos(engine, i)
        distances[i] = np.linalg.norm(glob_pos - joint_pos)

    # Sort by distance
    sorted_indices = np.argsort(distances)
    nearest_idx = sorted_indices[0]
    nearest_dist = distances[nearest_idx]

    if nearest_dist < 0.08:
        # LBS with nearest body + its parent
        lbs_body_ids = [nearest_idx]
        lbs_local_positions = [local_positions[nearest_idx]]
        lbs_weights = [1.0 / sqrt(nearest_dist) if nearest_dist > 1e-12 else 1e6]

        # Find parent body
        parent_bid = engine.body_parent_id(nearest_idx)
        if parent_bid > 0:  # has parent (not world)
            parent_dist = distances[parent_bid]
            lbs_body_ids.append(parent_bid)
            lbs_local_positions.append(local_positions[parent_bid])
            lbs_weights.append(
                1.0 / sqrt(parent_dist) if parent_dist > 1e-12 else 1e6
            )

        # Normalize weights
        total = sum(lbs_weights)
        lbs_weights = [w / total for w in lbs_weights]
    else:
        # Bind to specified body (single)
        bid = engine.body_id(body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        R = engine.get_body_xmat(bid)
        t = engine.get_body_xpos(bid)
        lbs_body_ids = [bid]
        lbs_local_positions = [R.T @ (glob_pos - t)]
        lbs_weights = [1.0]

    anchor = MuscleAnchor(
        body_ids=lbs_body_ids,
        local_positions=lbs_local_positions,
        weights=lbs_weights,
    )
    muscle.anchors.append(anchor)
    _finalize_anchor(muscle, engine)


def _finalize_anchor(muscle: Muscle, engine: PhysicsEngine) -> None:
    """Accumulate l_mt0 and recompute related DOFs after adding an anchor.

    Matches the tail of Muscle::AddAnchor in Muscle.cpp.
    """
    n = len(muscle.anchors)
    muscle.cached_anchor_positions = [np.zeros(3) for _ in range(n)]

    # Accumulate l_mt0 = sum of distances between consecutive anchors at rest
    if n > 1:
        p_new = muscle.anchors[n - 1].get_point(engine)
        p_prev = muscle.anchors[n - 2].get_point(engine)
        muscle.l_mt0 += np.linalg.norm(p_new - p_prev)

    # Update cached positions and lengths
    muscle.update(engine)

    # Compute related DOFs
    Jt = muscle.get_jacobian_transpose(engine)
    A, _ = muscle.get_force_jacobian_and_passive()
    JtA = Jt @ A

    muscle.num_related_dofs = 0
    muscle.related_dof_indices = []
    for i in range(JtA.shape[0]):
        if abs(JtA[i]) > 1e-3:
            muscle.num_related_dofs += 1
            muscle.related_dof_indices.append(i)


_PARSE_MUSCLE_CACHE: dict[tuple, list[Muscle]] = {}


def _engine_cache_key(engine: PhysicsEngine):
    # Use the id of the underlying model object — shared across worlds for
    # WarpEngine + WarpWorldView, and across reuses of the same MuJoCoEngine.
    model = getattr(engine, "model", None)
    if model is None:
        return None
    return id(model)


def parse_muscle_xml(
    xml_path: str, engine: PhysicsEngine, deepcopy_result: bool = True
) -> list[Muscle]:
    """Parse muscle definitions from MASS muscle XML.

    Matches Character::LoadMuscles in Character.cpp.

    Args:
        xml_path: Path to muscle284.xml
        engine: Physics engine (forward() must have been called).

    Returns:
        List of Muscle objects with anchors, l_mt0, and related DOFs computed.

    Results are cached per (xml_path mtime, engine.model identity). Repeated
    calls against the same engine model (e.g. batched WarpVecEnv constructing
    N MassEnvs over a single shared WarpEngine) reuse the parse and return a
    deep-copy so each caller still owns mutable per-world muscle state.
    """
    if os.environ.get("RL_MASS_DISABLE_MUSCLE_PARSE_CACHE"):
        return _do_parse_muscle_xml(xml_path, engine)
    try:
        mtime = os.path.getmtime(xml_path)
    except OSError:
        mtime = None
    key = (os.path.abspath(xml_path), mtime, _engine_cache_key(engine))
    cached = _PARSE_MUSCLE_CACHE.get(key)
    if cached is None:
        cached = _do_parse_muscle_xml(xml_path, engine)
        # Pre-update once against the (shared) initial engine state so cached
        # muscles already have cached_anchor_positions, l_mt and l_m filled.
        # Subsequent MassEnv constructions skip the per-env update loop.
        for m in cached:
            m.update(engine)
        _PARSE_MUSCLE_CACHE[key] = cached
    if not deepcopy_result:
        return cached
    return [copy.deepcopy(m) for m in cached]


def _do_parse_muscle_xml(xml_path: str, engine: PhysicsEngine) -> list[Muscle]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    muscles: list[Muscle] = []

    for unit in root.findall("Unit"):
        name = unit.attrib["name"]
        f0 = float(unit.attrib["f0"])
        lm = float(unit.attrib["lm"])
        lt = float(unit.attrib["lt"])
        pa = float(unit.attrib["pen_angle"])
        lmax = float(unit.attrib["lmax"])

        muscle = Muscle(
            name=name,
            f0=f0,
            l_m0=lm,
            l_t0=lt,
            pen_angle=pa,
            l_mt_max=lmax,
        )

        # Count waypoints
        waypoints = list(unit.findall("Waypoint"))
        num_waypoints = len(waypoints)

        for i, wp in enumerate(waypoints):
            body_name = wp.attrib["body"]
            p_str = wp.attrib["p"].strip().split()
            glob_pos = np.array([float(x) for x in p_str])

            if i == 0 or i == num_waypoints - 1:
                # First and last: bind to single body
                _add_anchor_single(muscle, engine, body_name, glob_pos)
            else:
                # Middle: LBS with distance-based weighting
                _add_anchor_lbs(muscle, engine, body_name, glob_pos)

        muscles.append(muscle)

    return muscles


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Basic smoke test: load muscles and print summary."""
    from rl_mass.engine import MuJoCoEngine

    print("=" * 60)
    print("Muscle Model Self-Test")
    print("=" * 60)

    engine = MuJoCoEngine("data/rl_mass/humanoid.xml", timestep=1.0 / 600)
    engine.forward()

    print(f"\nLoading muscles from external/MASS/data/muscle284.xml ...")
    muscles = parse_muscle_xml("external/MASS/data/muscle284.xml", engine)

    print(f"Loaded {len(muscles)} muscles")
    total_related_dofs = sum(m.num_related_dofs for m in muscles)
    print(f"Total related DOFs: {total_related_dofs}")

    # Print first 5 muscles
    print(f"\nFirst 5 muscles:")
    for m in muscles[:5]:
        print(
            f"  {m.name}: f0={m.f0:.1f}, l_m0={m.l_m0:.3f}, l_t0={m.l_t0:.3f}, "
            f"l_mt0={m.l_mt0:.4f}, anchors={len(m.anchors)}, "
            f"related_dofs={m.num_related_dofs}"
        )

    # Test force curves
    print(f"\nForce curve spot checks:")
    print(f"  g_al(1.0) = {g_al(1.0):.6f}  (expected: 1.0)")
    print(f"  g_al(0.5) = {g_al(0.5):.6f}  (expected: {exp(-0.25/0.5):.6f})")
    print(f"  g_pl(0.9) = {g_pl(0.9):.6f}  (expected: 0.0)")
    print(f"  g_pl(1.0) = {g_pl(1.0):.6f}  (expected: 0.0)")
    print(f"  g_pl(1.1) = {g_pl(1.1):.6f}")
    print(f"  g_t(0.0)  = {g_t(0.0):.6f}  (expected: 0.0)")
    print(f"  g_t(0.033)= {g_t(0.033):.6f}  (expected: ~2.42, continuous at e_t0)")

    # Test update and force for a muscle
    m0 = muscles[0]
    m0.activation = 1.0
    m0.update(engine)
    print(f"\nMuscle '{m0.name}' at rest pose with activation=1.0:")
    print(f"  l_mt (normalized) = {m0.l_mt:.6f}")
    print(f"  l_m               = {m0.l_m:.6f}")
    print(f"  l_m / l_m0        = {m0.l_m / m0.l_m0:.6f}")
    print(f"  f_A               = {m0.get_f_A():.4f}")
    print(f"  f_p               = {m0.get_f_p():.4f}")
    print(f"  total force       = {m0.get_force():.4f}")

    # Test Jacobian
    Jt = m0.get_jacobian_transpose(engine)
    print(f"  Jt shape          = {Jt.shape}")
    A, p = m0.get_force_jacobian_and_passive()
    JtA = Jt @ A
    print(f"  |JtA|             = {np.linalg.norm(JtA):.6f}")
    print(f"  related_dofs      = {m0.related_dof_indices}")

    jta_reduced = m0.get_related_jta(engine)
    print(f"  JtA_reduced shape = {jta_reduced.shape}")

    print("\n" + "=" * 60)
    print("SELF-TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
