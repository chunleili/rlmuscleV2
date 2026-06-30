"""Arm26 BimanualEnv builds from config: 2 DOF, 6 muscles, state/step run."""
import numpy as np
from rl_mass.env_factory import make_env

CFG = "data/rl_mass_arm26/config.json"

def main():
    env = make_env(CFG)
    assert env.num_action == 2, env.num_action       # no equality constraints
    assert env.num_muscles == 6, env.num_muscles
    s = env.reset(rsi=False)
    assert s.shape[0] == env.num_state
    # one control step with zero action + zero activation runs without NaN
    env.set_action(np.zeros(env.num_action))
    env.set_activation_levels(np.zeros(env.num_muscles))
    env.steps_at_once()
    assert not np.any(np.isnan(env.engine.get_qpos()))
    r = env.get_reward()
    assert 0.0 <= r <= 1.0, r
    print("OK")

if __name__ == "__main__":
    main()
