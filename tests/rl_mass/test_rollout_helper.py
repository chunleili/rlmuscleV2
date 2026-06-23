# tests/rl_mass/test_rollout_helper.py
"""Shared rollout helper: shapes line up on Arm26 with random-init nets."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from rl_mass.env_factory import make_env
from rl_mass.model import MuscleNN, SimulationNN
from tools._rollout import rollout

CFG = "data/rl_mass_arm26/config.json"

def main():
    env = make_env(CFG)
    model = SimulationNN(env.num_state, env.num_action)
    mm = MuscleNN(env.num_total_related_dofs, env.num_action, env.num_muscles)
    gx, gm, acts, qpos, rews = rollout(env, model, mm, max_steps=20, deterministic=True)
    T = gx.shape[0]
    assert acts.shape == (T, env.num_muscles), acts.shape
    assert qpos.shape[1] == env.engine.nq
    assert gm.shape[1:] == (env.model.ngeom, 3, 3)
    assert len(rews) <= T - 1  # one reward per non-terminal step
    assert np.all(acts >= 0.0)
    print("OK")

if __name__ == "__main__":
    main()
