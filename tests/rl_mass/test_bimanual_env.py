"""BimanualEnv: interface attrs, reset, step, reward, muscle tuple shapes."""
import numpy as np
from rl_mass.bimanual_env import BimanualEnv

CFG = "data/rl_mass_bimanual/config.json"


def main():
    env = BimanualEnv(CFG)
    assert env.num_action == 54, env.num_action
    assert env.num_muscles == 126, env.num_muscles
    assert env.use_muscle is True
    assert env.num_state == 153, env.num_state
    assert env.num_total_related_dofs > 0

    s = env.reset(rsi=False)
    assert s.shape == (153,), s.shape
    r = env.get_reward()
    assert 0.0 <= r <= 1.0 + 1e-6, r

    # One control step with zero action; collect a muscle tuple.
    env.set_action(np.zeros(env.num_action))
    env.set_activation_levels(np.zeros(env.num_muscles))
    env.steps_at_once()
    assert len(env.muscle_tuples) >= 1
    mt = env.muscle_tuples[-1]
    assert mt.L.shape == (54 * 126,), mt.L.shape
    assert mt.b.shape == (54,), mt.b.shape
    assert mt.tau_des.shape == (54,), mt.tau_des.shape
    assert mt.JtA.shape == (env.num_total_related_dofs,), mt.JtA.shape

    dt = env.get_desired_torques()
    assert dt.shape == (54,), dt.shape
    mtq = env.get_muscle_torques()
    assert mtq.shape == (env.num_total_related_dofs,), mtq.shape
    assert not env.is_end_of_episode()
    print("OK", env.num_state, env.num_total_related_dofs, round(r, 4))


if __name__ == "__main__":
    main()
