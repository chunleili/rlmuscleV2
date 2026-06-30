"""Regression: default config still resolves to MassEnv via the factory."""
from rl_mass.env_factory import make_env
from rl_mass.env import MassEnv


def main():
    # main moved per-task configs into data/rl_mass/configs/; walk has no
    # skeleton_format key, so it must route to the DART MassEnv path.
    env = make_env("data/rl_mass/configs/walk.json")
    assert isinstance(env, MassEnv)
    print("OK")


if __name__ == "__main__":
    main()
