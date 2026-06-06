"""Can the muscle basis span the SPD desired torque along the reference?"""
import numpy as np
from rl_mass.bimanual_env import BimanualEnv

CFG = "data/rl_mass_bimanual/config.json"


def main():
    env = BimanualEnv(CFG)
    residuals = []
    fracs_in_band = []
    for k in range(8):
        env.reset(rsi=False, t=k * env.ref.max_time / 8.0)
        env.set_action(np.zeros(env.num_action))
        tau_des = env.get_desired_torques()
        L, b, _ = env.muscle_system.tuple_terms(env.engine, np.zeros(env.num_muscles))
        # Solve min_a>=0-ish (unconstrained LS is enough to gauge span) for L a = tau_des - b
        a, *_ = np.linalg.lstsq(L, tau_des - b, rcond=None)
        resid = np.linalg.norm(L @ a + b - tau_des) / (np.linalg.norm(tau_des) + 1e-9)
        residuals.append(resid)
        x = env.muscle_system.l_m_norm
        fracs_in_band.append(float(np.mean((x > 0.4) & (x < 1.6))))
    print("relative LS residual (mean):", round(float(np.mean(residuals)), 4))
    print("l_m_norm fraction in [0.4,1.6] (mean):", round(float(np.mean(fracs_in_band)), 3))
    # Soft gate: report, then assert the basis is not degenerate.
    assert np.mean(residuals) < 0.5, np.mean(residuals)
    print("OK")


if __name__ == "__main__":
    main()
