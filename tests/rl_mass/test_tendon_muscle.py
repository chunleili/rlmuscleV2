"""TendonMuscleSystem: dims, sign cross-check vs MuJoCo, tuple terms."""
import numpy as np
from rl_mass.engine import MuJoCoEngine
from rl_mass.tendon_muscle import TendonMuscleSystem

XML = "/home/chunleli/Dev/musclemimic_models/musclemimic_models/model/arm/myoarm_bimanual.xml"


def main():
    eng = MuJoCoEngine(XML, timestep=1.0 / 600)
    eng.forward()
    ms = TendonMuscleSystem(eng)
    assert ms.n_muscles == 126, ms.n_muscles
    assert ms.f0.shape == (126,) and np.all(ms.f0 > 0)
    assert ms.nv == 76

    ms.update(eng)
    # R is the per-muscle moment column matrix (nv, n_muscles).
    assert ms.R.shape == (76, 126), ms.R.shape

    # --- Sign cross-check against MuJoCo's own actuator path ---
    # If we feed F (>=0 tension), our generalized force must equal MuJoCo's
    # qfrc_actuator when its actuator_force == -F. Use a fresh data with
    # actuation ENABLED to read qfrc_actuator for a chosen ctrl.
    import mujoco
    m2 = mujoco.MjModel.from_xml_path(XML)
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    # Drive activation of one muscle via ctrl; read resulting actuator_force + qfrc.
    d2.ctrl[:] = 0.0
    d2.ctrl[0] = 1.0
    for _ in range(50):  # let activation dynamics settle
        mujoco.mj_forward(m2, d2)
        d2.act[:] = d2.ctrl  # shortcut: force act to ctrl for steady state
    mujoco.mj_forward(m2, d2)
    F_signed = d2.actuator_force.copy()           # (nu,), muscle force (<=0)
    qfrc_mj = d2.qfrc_actuator.copy()             # (nv,)
    # Our convention: qfrc = R @ F where F is +tension; MuJoCo F_signed = -F.
    qfrc_ours = ms.R @ (-F_signed)
    assert np.allclose(qfrc_ours, qfrc_mj, atol=1e-6), np.abs(qfrc_ours - qfrc_mj).max()

    # --- Tuple terms shapes ---
    a = np.zeros(126)
    L, b, JtA = ms.tuple_terms(eng, a)
    assert L.shape == (76, 126), L.shape
    assert b.shape == (76,), b.shape
    assert JtA.shape == (ms.num_total_related_dofs,), JtA.shape
    print("OK", ms.n_muscles, ms.num_total_related_dofs)


if __name__ == "__main__":
    main()
