"""The muscle basis must span the SPD desired torque on the INDEPENDENT DOFs.

Constraint-dependent shoulder-rhythm DOFs are excluded (driven by MuJoCo
equality constraints, not muscles). On the remaining independent DOFs the
moment matrix is full rank, so the least-squares torque-matching residual
should be ~0.
"""
import numpy as np
from rl_mass.finger.bimanual_env import BimanualEnv

CFG = "data/rl_mass_bimanual/config.json"


def main():
    env = BimanualEnv(CFG)
    act = env.active_dof
    residuals = []
    fracs_in_band = []
    for k in range(8):
        env.reset(rsi=False, t=k * env.ref.max_time / 8.0)
        env.set_action(np.zeros(env.num_action))
        tau_des = env.get_desired_torques()  # already on independent DOFs (num_action,)
        L, b, _ = env.muscle_system.tuple_terms(env.engine, np.zeros(env.num_muscles))
        La = L[act]
        rhs = tau_des - b[act]
        a, *_ = np.linalg.lstsq(La, rhs, rcond=None)
        resid = np.linalg.norm(La @ a - rhs) / (np.linalg.norm(tau_des) + 1e-9)
        residuals.append(resid)
        x = env.muscle_system.l_m_norm
        fracs_in_band.append(float(np.mean((x > 0.4) & (x < 1.6))))
    print("relative LS residual on independent DOFs (mean):", round(float(np.mean(residuals)), 4))
    print("l_m_norm fraction in [0.4,1.6] (mean):", round(float(np.mean(fracs_in_band)), 3))
    assert np.mean(residuals) < 0.2, np.mean(residuals)
    print("OK")


if __name__ == "__main__":
    main()
