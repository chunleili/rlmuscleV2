"""make_env returns BimanualEnv for the native config; serial vec env runs."""
from rl_mass.env_factory import make_env
from rl_mass.bimanual_env import BimanualEnv
from rl_mass.vec_env import make_vec_env
import numpy as np

CFG = "data/rl_mass_bimanual/config.json"


def main():
    env = make_env(CFG)
    assert isinstance(env, BimanualEnv)

    vec = make_vec_env(CFG, num_envs=2, mode="serial")
    assert vec.num_action == 76
    assert vec.num_muscles == 126
    states = vec.get_states()
    assert states.shape == (2, 153), states.shape
    vec.set_actions(np.zeros((2, 76)))
    vec.set_activation_levels(np.zeros((2, 126)))
    vec.steps_at_once()
    tuples = vec.gather_muscle_tuples()
    assert len(tuples) >= 2, len(tuples)
    print("OK", states.shape, len(tuples))


if __name__ == "__main__":
    main()
