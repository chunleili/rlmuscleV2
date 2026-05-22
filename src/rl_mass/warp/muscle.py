"""Batched muscle system on GPU via NVIDIA Warp kernels.

Mirrors `rl_mass.muscle.MuscleSystem` semantics but operates on N worlds in
parallel. Topology arrays (anchor body IDs, LBS weights, segment pairs,
unique-body grouping) are shared across worlds — only state is batched:

    anchor_world_pos : (nworld, total_anchors) vec3f
    l_mt             : (nworld, n_muscles)
    l_m              : (nworld, n_muscles)
    activations      : (nworld, n_muscles)

Built from a CPU `MuscleSystem` so the loader path stays unchanged. The
warp backend bypasses the CPU MuscleSystem at runtime (`vectorized_muscles=
False` on per-world MassEnv) and drives this batched system instead.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from rl_mass.muscle import MuscleSystem


@wp.kernel
def _update_anchors_lbs_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    anchor_bid0: wp.array(dtype=wp.int32),
    anchor_bid1: wp.array(dtype=wp.int32),
    anchor_lp0: wp.array(dtype=wp.vec3),
    anchor_lp1: wp.array(dtype=wp.vec3),
    anchor_w0: wp.array(dtype=wp.float32),
    anchor_w1: wp.array(dtype=wp.float32),
    out_world_pos: wp.array2d(dtype=wp.vec3),
):
    w, a = wp.tid()
    b0 = anchor_bid0[a]
    b1 = anchor_bid1[a]
    p0 = xmat[w, b0] * anchor_lp0[a] + xpos[w, b0]
    p1 = xmat[w, b1] * anchor_lp1[a] + xpos[w, b1]
    out_world_pos[w, a] = anchor_w0[a] * p0 + anchor_w1[a] * p1


@wp.kernel
def _zero_lmt_kernel(l_mt: wp.array2d(dtype=wp.float32)):
    w, m = wp.tid()
    l_mt[w, m] = 0.0


@wp.kernel
def _accumulate_seg_lengths_kernel(
    world_pos: wp.array2d(dtype=wp.vec3),
    seg_a: wp.array(dtype=wp.int32),
    seg_b: wp.array(dtype=wp.int32),
    seg_muscle: wp.array(dtype=wp.int32),
    l_mt: wp.array2d(dtype=wp.float32),
):
    w, s = wp.tid()
    pa = world_pos[w, seg_a[s]]
    pb = world_pos[w, seg_b[s]]
    d = pb - pa
    length = wp.sqrt(wp.dot(d, d))
    m = seg_muscle[s]
    wp.atomic_add(l_mt, w, m, length)


@wp.kernel
def _compute_muscle_force_kernel(
    l_m: wp.array2d(dtype=wp.float32),
    l_m0: wp.array(dtype=wp.float32),
    f0: wp.array(dtype=wp.float32),
    activations: wp.array2d(dtype=wp.float32),
    out_muscle_force: wp.array2d(dtype=wp.float32),
    out_f_active: wp.array2d(dtype=wp.float32),
):
    w, m = wp.tid()
    lmn = l_m[w, m] / l_m0[m]
    g_al = wp.exp(-((lmn - 1.0) * (lmn - 1.0)) / 0.5)
    if lmn >= 1.0:
        g_pl = (wp.exp(4.0 * (lmn - 1.0) / 0.6) - 1.0) / (wp.exp(4.0) - 1.0)
    else:
        g_pl = 0.0
    f = f0[m]
    out_muscle_force[w, m] = f * (g_al * activations[w, m] + g_pl)
    out_f_active[w, m] = f * g_al


@wp.kernel
def _zero_force_dir_kernel(out: wp.array2d(dtype=wp.vec3)):
    w, a = wp.tid()
    out[w, a] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _accumulate_force_dir_kernel(
    world_pos: wp.array2d(dtype=wp.vec3),
    seg_a: wp.array(dtype=wp.int32),
    seg_b: wp.array(dtype=wp.int32),
    force_dir: wp.array2d(dtype=wp.vec3),
):
    w, s = wp.tid()
    ai = seg_a[s]
    bi = seg_b[s]
    d = world_pos[w, bi] - world_pos[w, ai]
    n = wp.sqrt(wp.dot(d, d))
    if n < 1.0e-12:
        return
    u = d / n
    wp.atomic_add(force_dir, w, ai, u)
    wp.atomic_add(force_dir, w, bi, -u)


@wp.kernel
def _compute_wrenches_kernel(
    force_dir: wp.array2d(dtype=wp.vec3),
    f_active: wp.array2d(dtype=wp.float32),
    anchor_muscle_id: wp.array(dtype=wp.int32),
    anchor_primary_bid: wp.array(dtype=wp.int32),
    world_pos: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3),
    wrench_F: wp.array2d(dtype=wp.vec3),
    wrench_T: wp.array2d(dtype=wp.vec3),
):
    w, a = wp.tid()
    m = anchor_muscle_id[a]
    bid = anchor_primary_bid[a]
    F = force_dir[w, a] * f_active[w, m]
    r = world_pos[w, a] - xipos[w, bid]
    wrench_F[w, a] = F
    wrench_T[w, a] = wp.cross(r, F)


@wp.kernel
def _accumulate_jta_body_kernel(
    jacp: wp.array3d(dtype=wp.float32),     # (nworld, 3, nv)
    jacr: wp.array3d(dtype=wp.float32),
    anc_idx_on_body: wp.array(dtype=wp.int32),  # anchor indices on this body
    anchor_muscle_id: wp.array(dtype=wp.int32),
    wrench_F: wp.array2d(dtype=wp.vec3),
    wrench_T: wp.array2d(dtype=wp.vec3),
    jta_per_muscle: wp.array3d(dtype=wp.float32),  # (nworld, n_muscles, nv)
):
    w, k, dof = wp.tid()
    a = anc_idx_on_body[k]
    m = anchor_muscle_id[a]
    F = wrench_F[w, a]
    T = wrench_T[w, a]
    fx = F[0]; fy = F[1]; fz = F[2]
    tx = T[0]; ty = T[1]; tz = T[2]
    jp0 = jacp[w, 0, dof]
    jp1 = jacp[w, 1, dof]
    jp2 = jacp[w, 2, dof]
    jr0 = jacr[w, 0, dof]
    jr1 = jacr[w, 1, dof]
    jr2 = jacr[w, 2, dof]
    contrib = fx * jp0 + fy * jp1 + fz * jp2 + tx * jr0 + ty * jr1 + tz * jr2
    wp.atomic_add(jta_per_muscle, w, m, dof, contrib)


@wp.kernel
def _zero_jta_kernel(jta: wp.array3d(dtype=wp.float32)):
    w, m, dof = wp.tid()
    jta[w, m, dof] = 0.0


@wp.kernel
def _gather_xipos_for_body_kernel(
    xipos: wp.array2d(dtype=wp.vec3),
    bid: wp.int32,
    out_point: wp.array(dtype=wp.vec3),
):
    w = wp.tid()
    out_point[w] = xipos[w, bid]


@wp.kernel
def _gather_related_jta_kernel(
    jta_per_muscle: wp.array3d(dtype=wp.float32),  # (nworld, n_muscles, nv)
    related_dof: wp.array(dtype=wp.int32),         # (total_related,) flat
    related_muscle: wp.array(dtype=wp.int32),      # (total_related,) muscle of each entry
    out: wp.array2d(dtype=wp.float32),             # (nworld, total_related)
):
    w, i = wp.tid()
    out[w, i] = jta_per_muscle[w, related_muscle[i], related_dof[i]]


@wp.kernel
def _apply_segment_forces_kernel(
    world_pos: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3),
    seg_a: wp.array(dtype=wp.int32),
    seg_b: wp.array(dtype=wp.int32),
    seg_muscle: wp.array(dtype=wp.int32),
    anchor_primary_bid: wp.array(dtype=wp.int32),
    muscle_force: wp.array2d(dtype=wp.float32),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    w, s = wp.tid()
    ai = seg_a[s]
    bi = seg_b[s]
    pa = world_pos[w, ai]
    pb = world_pos[w, bi]
    d = pb - pa
    n = wp.sqrt(wp.dot(d, d))
    if n < 1.0e-12:
        return
    direction = d / n
    f_mag = muscle_force[w, seg_muscle[s]]
    fvec = direction * f_mag

    bid_a = anchor_primary_bid[ai]
    bid_b = anchor_primary_bid[bi]
    com_a = xipos[w, bid_a]
    com_b = xipos[w, bid_b]
    torque_a = wp.cross(pa - com_a, fvec)
    nfvec = -fvec
    torque_b = wp.cross(pb - com_b, nfvec)

    wa = wp.spatial_vector(fvec[0], fvec[1], fvec[2], torque_a[0], torque_a[1], torque_a[2])
    wb = wp.spatial_vector(nfvec[0], nfvec[1], nfvec[2], torque_b[0], torque_b[1], torque_b[2])
    wp.atomic_add(xfrc_applied, w, bid_a, wa)
    wp.atomic_add(xfrc_applied, w, bid_b, wb)


@wp.kernel
def _normalize_lmt_kernel(
    l_mt0: wp.array(dtype=wp.float32),
    l_t0: wp.array(dtype=wp.float32),
    l_mt: wp.array2d(dtype=wp.float32),
    l_m: wp.array2d(dtype=wp.float32),
):
    w, m = wp.tid()
    rest = l_mt0[m]
    if rest > 0.0:
        v = l_mt[w, m] / rest
    else:
        v = 1.0
    l_mt[w, m] = v
    l_m[w, m] = v - l_t0[m]


class WarpMuscleSystem:
    """GPU-batched muscle simulation over a shared `WarpEngine`.

    All state (anchor_world_pos, l_mt, l_m, activations) is `(nworld, ...)`.
    Topology arrays live on device but have no world dimension.
    """

    def __init__(self, cpu_system: MuscleSystem, engine):
        self._engine = engine
        self.n_muscles = int(cpu_system.n_muscles)
        self.total_anchors = int(cpu_system.total_anchors)
        self.n_segments = int(cpu_system.n_segments)
        self.num_worlds = engine.num_worlds
        self.device = engine.device

        with wp.ScopedDevice(self.device):
            self._init_topology(cpu_system)
            self._init_state()

    def _init_topology(self, cs: MuscleSystem) -> None:
        lp0 = cs.anchor_local_pos[:, 0, :].astype(np.float32)
        lp1 = cs.anchor_local_pos[:, 1, :].astype(np.float32)
        self.anchor_bid0 = wp.array(cs.anchor_body_ids[:, 0].astype(np.int32), dtype=wp.int32)
        self.anchor_bid1 = wp.array(cs.anchor_body_ids[:, 1].astype(np.int32), dtype=wp.int32)
        self.anchor_lp0 = wp.array(lp0, dtype=wp.vec3)
        self.anchor_lp1 = wp.array(lp1, dtype=wp.vec3)
        self.anchor_w0 = wp.array(cs.anchor_weights[:, 0].astype(np.float32), dtype=wp.float32)
        self.anchor_w1 = wp.array(cs.anchor_weights[:, 1].astype(np.float32), dtype=wp.float32)
        self.anchor_primary_bid = wp.array(cs.anchor_primary_bid.astype(np.int32), dtype=wp.int32)
        self.anchor_muscle_id = wp.array(cs.anchor_muscle_id.astype(np.int32), dtype=wp.int32)

        self.seg_a = wp.array(cs.seg_a.astype(np.int32), dtype=wp.int32)
        self.seg_b = wp.array(cs.seg_b.astype(np.int32), dtype=wp.int32)
        self.seg_muscle = wp.array(cs.seg_muscle.astype(np.int32), dtype=wp.int32)

        self.f0 = wp.array(cs.f0.astype(np.float32), dtype=wp.float32)
        self.l_m0 = wp.array(cs.l_m0.astype(np.float32), dtype=wp.float32)
        self.l_t0 = wp.array(cs.l_t0.astype(np.float32), dtype=wp.float32)
        self.l_mt0 = wp.array(cs.l_mt0.astype(np.float32), dtype=wp.float32)

        self.unique_bodies_np = cs.unique_bodies.astype(np.int32)
        self.body_anchor_lists_np = [arr.astype(np.int32) for arr in cs.body_anchor_lists]
        self.related_dof_indices_np = [arr.astype(np.int32) for arr in cs.related_dof_indices_arr]
        self.num_related_dofs_np = np.asarray(cs.num_related_dofs, dtype=np.int32)

    def _init_state(self) -> None:
        nw = self.num_worlds
        self.anchor_world_pos = wp.zeros((nw, self.total_anchors), dtype=wp.vec3)
        self.l_mt = wp.zeros((nw, self.n_muscles), dtype=wp.float32)
        self.l_m = wp.zeros((nw, self.n_muscles), dtype=wp.float32)
        self.activations = wp.zeros((nw, self.n_muscles), dtype=wp.float32)
        self.muscle_force = wp.zeros((nw, self.n_muscles), dtype=wp.float32)
        self.f_active = wp.zeros((nw, self.n_muscles), dtype=wp.float32)
        self.force_dir = wp.zeros((nw, self.total_anchors), dtype=wp.vec3)
        self.wrench_F = wp.zeros((nw, self.total_anchors), dtype=wp.vec3)
        self.wrench_T = wp.zeros((nw, self.total_anchors), dtype=wp.vec3)

        nv = self._engine.nv
        self.jta_per_muscle = wp.zeros((nw, self.n_muscles, nv), dtype=wp.float32)
        self.jacp_buf = wp.zeros((nw, 3, nv), dtype=wp.float32)
        self.jacr_buf = wp.zeros((nw, 3, nv), dtype=wp.float32)

        # Pre-stage per-body anchor index buffers on device.
        self.body_anchor_arrs = [
            wp.array(arr, dtype=wp.int32) for arr in self.body_anchor_lists_np
        ]
        self.body_bid_arrs = [
            wp.array(np.full(self.num_worlds, int(bid), dtype=np.int32), dtype=wp.int32)
            for bid in self.unique_bodies_np
        ]
        self.body_point_bufs = [
            wp.zeros(self.num_worlds, dtype=wp.vec3)
            for _ in self.unique_bodies_np
        ]

        # Flatten related-DOF gather indices.
        related_dof_flat = []
        related_muscle_flat = []
        for m_i, rel in enumerate(self.related_dof_indices_np):
            related_dof_flat.append(rel)
            related_muscle_flat.append(np.full(rel.size, m_i, dtype=np.int32))
        if related_dof_flat:
            related_dof_flat = np.concatenate(related_dof_flat).astype(np.int32)
            related_muscle_flat = np.concatenate(related_muscle_flat).astype(np.int32)
        else:
            related_dof_flat = np.zeros(0, dtype=np.int32)
            related_muscle_flat = np.zeros(0, dtype=np.int32)
        self.total_related = int(related_dof_flat.size)
        self.related_dof = wp.array(related_dof_flat, dtype=wp.int32)
        self.related_muscle = wp.array(related_muscle_flat, dtype=wp.int32)
        self.related_jta = wp.zeros((nw, self.total_related), dtype=wp.float32)

    def update_all(self) -> None:
        d = self._engine.data
        with wp.ScopedDevice(self.device):
            wp.launch(
                _update_anchors_lbs_kernel,
                dim=(self.num_worlds, self.total_anchors),
                inputs=[
                    d.xpos, d.xmat,
                    self.anchor_bid0, self.anchor_bid1,
                    self.anchor_lp0, self.anchor_lp1,
                    self.anchor_w0, self.anchor_w1,
                ],
                outputs=[self.anchor_world_pos],
            )
            wp.launch(
                _zero_lmt_kernel,
                dim=(self.num_worlds, self.n_muscles),
                inputs=[self.l_mt],
            )
            wp.launch(
                _accumulate_seg_lengths_kernel,
                dim=(self.num_worlds, self.n_segments),
                inputs=[
                    self.anchor_world_pos,
                    self.seg_a, self.seg_b, self.seg_muscle,
                    self.l_mt,
                ],
            )
            wp.launch(
                _normalize_lmt_kernel,
                dim=(self.num_worlds, self.n_muscles),
                inputs=[self.l_mt0, self.l_t0],
                outputs=[self.l_mt, self.l_m],
            )

    def apply_forces_all(self) -> None:
        """Compute muscle forces and accumulate into engine.data.xfrc_applied.

        Caller is responsible for clearing xfrc_applied before each control
        step (matches CPU MuscleSystem contract). `update_all()` must have
        been called this substep.
        """
        d = self._engine.data
        with wp.ScopedDevice(self.device):
            wp.launch(
                _compute_muscle_force_kernel,
                dim=(self.num_worlds, self.n_muscles),
                inputs=[self.l_m, self.l_m0, self.f0, self.activations],
                outputs=[self.muscle_force, self.f_active],
            )
            wp.launch(
                _apply_segment_forces_kernel,
                dim=(self.num_worlds, self.n_segments),
                inputs=[
                    self.anchor_world_pos, d.xipos,
                    self.seg_a, self.seg_b, self.seg_muscle,
                    self.anchor_primary_bid,
                    self.muscle_force,
                ],
                outputs=[d.xfrc_applied],
            )

    def get_all_muscle_torques(self) -> wp.array:
        """Return batched (nworld, total_related) JtA. Caller must have called
        update_all() and f_active must be current (call after apply_forces_all
        or compute it here)."""
        import mujoco_warp

        d = self._engine.data
        m_dev = self._engine.model
        nv = self._engine.nv

        with wp.ScopedDevice(self.device):
            # Ensure f_active is current — recompute (cheap).
            wp.launch(
                _compute_muscle_force_kernel,
                dim=(self.num_worlds, self.n_muscles),
                inputs=[self.l_m, self.l_m0, self.f0, self.activations],
                outputs=[self.muscle_force, self.f_active],
            )

            wp.launch(
                _zero_force_dir_kernel,
                dim=(self.num_worlds, self.total_anchors),
                inputs=[self.force_dir],
            )
            wp.launch(
                _accumulate_force_dir_kernel,
                dim=(self.num_worlds, self.n_segments),
                inputs=[
                    self.anchor_world_pos,
                    self.seg_a, self.seg_b,
                ],
                outputs=[self.force_dir],
            )
            wp.launch(
                _compute_wrenches_kernel,
                dim=(self.num_worlds, self.total_anchors),
                inputs=[
                    self.force_dir, self.f_active,
                    self.anchor_muscle_id, self.anchor_primary_bid,
                    self.anchor_world_pos, d.xipos,
                ],
                outputs=[self.wrench_F, self.wrench_T],
            )
            wp.launch(
                _zero_jta_kernel,
                dim=(self.num_worlds, self.n_muscles, nv),
                inputs=[self.jta_per_muscle],
            )

            for k, bid in enumerate(self.unique_bodies_np):
                wp.launch(
                    _gather_xipos_for_body_kernel,
                    dim=self.num_worlds,
                    inputs=[d.xipos, int(bid)],
                    outputs=[self.body_point_bufs[k]],
                )
                mujoco_warp.jac(
                    m_dev, d,
                    self.jacp_buf, self.jacr_buf,
                    self.body_point_bufs[k],
                    self.body_bid_arrs[k],
                )
                anc_idx = self.body_anchor_arrs[k]
                wp.launch(
                    _accumulate_jta_body_kernel,
                    dim=(self.num_worlds, int(anc_idx.shape[0]), nv),
                    inputs=[
                        self.jacp_buf, self.jacr_buf,
                        anc_idx, self.anchor_muscle_id,
                        self.wrench_F, self.wrench_T,
                    ],
                    outputs=[self.jta_per_muscle],
                )

            wp.launch(
                _gather_related_jta_kernel,
                dim=(self.num_worlds, self.total_related),
                inputs=[self.jta_per_muscle, self.related_dof, self.related_muscle],
                outputs=[self.related_jta],
            )

        return self.related_jta

    def set_activations(self, acts: np.ndarray) -> None:
        a = np.asarray(acts, dtype=np.float32)
        if a.shape != (self.num_worlds, self.n_muscles):
            raise ValueError(
                f"activations shape {a.shape} != "
                f"({self.num_worlds}, {self.n_muscles})"
            )
        self.activations.assign(a)

    def set_activations_tensor(self, t) -> None:
        """Device-to-device activation copy from a torch tensor on self.device.

        Avoids the H<->D round-trip incurred by numpy `assign()`.
        """
        import torch

        if not isinstance(t, torch.Tensor):
            raise TypeError("set_activations_tensor expects a torch.Tensor")
        if t.shape != (self.num_worlds, self.n_muscles):
            raise ValueError(
                f"activations tensor shape {tuple(t.shape)} != "
                f"({self.num_worlds}, {self.n_muscles})"
            )
        t32 = t.detach().to(dtype=torch.float32).contiguous()
        with wp.ScopedDevice(self.device):
            src = wp.from_torch(t32, dtype=wp.float32)
            wp.copy(self.activations, src)
