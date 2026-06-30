"""Smoke test: bimanual MJCF loads via MuJoCoEngine with expected dims."""
import json
from rl_mass.engine import MuJoCoEngine


def main():
    with open("data/rl_mass_bimanual/config.json") as f:
        cfg = json.load(f)
    eng = MuJoCoEngine(cfg["skeleton_file"], timestep=1.0 / cfg["simulation_hz"])
    eng.forward()
    assert eng.nq == 76, eng.nq
    assert eng.nv == 76, eng.nv
    assert eng.model.nu == 126, eng.model.nu
    assert eng.model.ntendon == 134, eng.model.ntendon
    print("OK", eng.nq, eng.nv, eng.model.nu, eng.model.ntendon)


if __name__ == "__main__":
    main()
