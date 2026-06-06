"""Regression: default config still resolves to MassEnv via the factory."""
from rl_mass.env_factory import make_env
from rl_mass.env import MassEnv


def main():
    env = make_env("data/rl_mass/config.json")
    assert isinstance(env, MassEnv)
    print("OK")


if __name__ == "__main__":
    main()
