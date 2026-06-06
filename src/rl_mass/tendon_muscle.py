"""1D Hill muscles backed by MuJoCo spatial tendons.

Replaces the waypoint MuscleSystem for native-MJCF cases. Length and
moment-arms come from MuJoCo's tendon machinery (wrapping preserved);
rl_mass's Hill force-length curves (g_al / g_pl) are applied on top.

Per-muscle moment column matrix:
    R[:, m] = -moment_dense[m, :]          # (nv,) generalized force per +unit tension
Force scalars:
    f_active[m] = f0[m] * g_al(l_m_norm[m])
    f_passive[m] = f0[m] * g_pl(l_m_norm[m])
Applied generalized force for activations a:
    qfrc = R @ (f_active * a + f_passive)
Muscle-tuple matrices (active DOFs = all DOFs; no root):
    L[:, m] = R[:, m] * f_active[m]
    b       = R @ f_passive
    JtA     = concat over m of L[related_dofs(m), m]
"""
from __future__ import annotations

import numpy as np

from rl_mass.engine import MuJoCoEngine
# Hill force-length curves are inlined (vectorized) in update(); they mirror
# g_al / g_pl in rl_mass.muscle — keep the constants in sync if those change.


class TendonMuscleSystem:
    def __init__(self, engine: MuJoCoEngine):
        import mujoco

        self._mj = mujoco
        m = engine.model
        self.nv = int(m.nv)
        self.n_muscles = int(m.nu)

        # Per-muscle static params.
        self.f0 = m.actuator_gainprm[:, 2].copy().astype(np.float64)
        assert np.all(self.f0 > 0), "expected all muscle f0 > 0"
        self._LR = m.actuator_lengthrange.copy().astype(np.float64)   # (nu, 2)
        rng = m.actuator_gainprm[:, 0:2].copy().astype(np.float64)    # (nu, 2)
        assert np.all(self._LR[:, 1] > self._LR[:, 0]), "bad actuator_lengthrange"
        assert np.all(rng[:, 1] > rng[:, 0]), "bad gainprm operating range"
        self._rng0 = rng[:, 0]
        self._L0 = (self._LR[:, 1] - self._LR[:, 0]) / (rng[:, 1] - rng[:, 0])

        # Dense-moment scratch buffer.
        self._moment_dense = np.zeros((self.n_muscles, self.nv), dtype=np.float64)

        # State (filled by update()).
        self.R = np.zeros((self.nv, self.n_muscles), dtype=np.float64)
        self.l_m_norm = np.ones(self.n_muscles, dtype=np.float64)
        self.f_active = np.zeros(self.n_muscles, dtype=np.float64)
        self.f_passive = np.zeros(self.n_muscles, dtype=np.float64)

        # Related-DOF structure, computed once at rest pose.
        engine.forward()
        self._densify_moment(engine)
        R_rest = -self._moment_dense.T  # (nv, nu)
        # Related DOFs = DOFs a muscle's tendon path structurally depends on.
        # Computed ONCE at rest pose; num_total_related_dofs must stay fixed
        # (it sizes the MuscleNN JtA input). A wrapped spatial tendon has a
        # continuous moment function, so a structurally-spanned DOF is nonzero
        # at virtually all poses incl. neutral — the 1e-6 m floor keeps those
        # (important for small-moment-arm finger joints) while excluding pure
        # numerical noise. If a future motion needs a DOF that is exactly zero
        # at rest, build this union over several sampled reference poses instead.
        self.related_dof_indices = [
            np.where(np.abs(R_rest[:, m]) > 1e-6)[0].astype(np.intp)
            for m in range(self.n_muscles)
        ]
        self.num_related_dofs = [int(idx.size) for idx in self.related_dof_indices]
        self.num_total_related_dofs = int(sum(self.num_related_dofs))

    def _densify_moment(self, engine: MuJoCoEngine) -> None:
        d = engine.data
        self._mj.mju_sparse2dense(
            self._moment_dense, d.actuator_moment,
            d.moment_rownnz, d.moment_rowadr, d.moment_colind,
        )

    def update(self, engine: MuJoCoEngine) -> None:
        """Recompute R, normalized length, and force scalars at current pose."""
        d = engine.data
        self._densify_moment(engine)
        self.R[:] = -self._moment_dense.T  # (nv, nu)

        al = d.actuator_length
        # Normalized fiber length via MuJoCo's operating range. NOT clamped:
        # poses outside [LR0, LR1] yield l_m_norm outside the [rng0, rng1]
        # band, and g_al naturally decays there — that decay is the intended
        # force-length behavior (see Task 7 conditioning gate).
        self.l_m_norm[:] = self._rng0 + (al - self._LR[:, 0]) / self._L0
        # Vectorized g_al / g_pl (same math as rl_mass.muscle).
        x = self.l_m_norm
        self.f_active[:] = self.f0 * np.exp(-((x - 1.0) ** 2) / 0.5)
        fp = np.zeros_like(x)
        mask = x >= 1.0
        fp[mask] = (np.exp(4.0 * (x[mask] - 1.0) / 0.6) - 1.0) / (np.exp(4.0) - 1.0)
        self.f_passive[:] = self.f0 * fp

    def apply_forces(self, engine: MuJoCoEngine, activations: np.ndarray) -> None:
        """Set qfrc_applied from current R and the given activations.

        CONTRACT: assumes update() was already called this substep (uses the
        cached R / f_active / f_passive). Unlike get_muscle_torques and
        tuple_terms, this does NOT self-update — the env's step() calls
        update() right before apply_forces() every substep, and re-updating
        here would double-compute the dense moment in the hot loop.
        """
        a = np.asarray(activations, dtype=np.float64).ravel()
        muscle_force = self.f_active * a + self.f_passive  # (nu,)
        qfrc = self.R @ muscle_force                       # (nv,)
        engine.set_qfrc_applied(qfrc)

    def get_muscle_torques(self, engine: MuJoCoEngine) -> np.ndarray:
        """Concatenated related-DOF JtA = R[related, m] * f_active[m]."""
        self.update(engine)
        out = np.empty(self.num_total_related_dofs, dtype=np.float64)
        i = 0
        for m in range(self.n_muscles):
            idx = self.related_dof_indices[m]
            n = idx.size
            out[i:i + n] = self.R[idx, m] * self.f_active[m]
            i += n
        return out

    def tuple_terms(
        self, engine: MuJoCoEngine, activations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (L (nv,nu), b (nv,), JtA (num_total_related_dofs,)) at current pose."""
        self.update(engine)
        L = self.R * self.f_active[None, :]   # (nv, nu)
        b = self.R @ self.f_passive           # (nv,)
        out = np.empty(self.num_total_related_dofs, dtype=np.float64)
        i = 0
        for m in range(self.n_muscles):
            idx = self.related_dof_indices[m]
            n = idx.size
            out[i:i + n] = L[idx, m]
            i += n
        return L, b, out
