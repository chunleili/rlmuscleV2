"""Stable-PD torques for a fixed-base, all-hinge MuJoCo model.

All joints are 1-DOF (nq == nv), so positions subtract directly and no
root DOFs are special-cased. Formula matches DART/MASS stable PD:

    M_mod = M + dt*diag(Kv)
    qdqdt = q + dq*dt
    p_diff = -Kp*(qdqdt - q_des)
    v_diff = -Kv*dq
    ddq = M_mod^{-1} @ (-C_G + p_diff + v_diff + C_f)
    tau = p_diff + v_diff - dt*Kv*ddq
"""
from __future__ import annotations

import numpy as np

from rl_mass.engine import MuJoCoEngine


def native_spd_forces(
    engine: MuJoCoEngine,
    q_des: np.ndarray,
    kp: float,
    kv: float,
) -> np.ndarray:
    nv = engine.nv
    dt = engine.timestep
    Kp = np.full(nv, kp, dtype=np.float64)
    Kv = np.full(nv, kv, dtype=np.float64)

    q = engine.get_qpos()
    dq = engine.get_qvel()
    M = engine.get_mass_matrix()
    M_mod = M + dt * np.diag(Kv)
    M_inv = np.linalg.inv(M_mod)

    qdqdt = q + dq * dt
    p_diff = -Kp * (qdqdt - q_des)
    v_diff = -Kv * dq
    cg = engine.get_coriolis_gravity()
    cf = engine.get_constraint_forces()
    ddq = M_inv @ (-cg + p_diff + v_diff + cf)
    tau = p_diff + v_diff - dt * Kv * ddq
    return tau
