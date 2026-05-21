"""GPU vectorized backend for rl_mass via mujoco_warp + NVIDIA Warp kernels.

Mirrors the structure of rl_mass/ at the file level:

    rl_mass/engine.py      <-> rl_mass/warp/engine.py
    rl_mass/env.py         <-> rl_mass/warp/env.py
    rl_mass/muscle.py      <-> rl_mass/warp/muscle.py
    rl_mass/vec_env.py     <-> rl_mass/warp/vec_env.py
    rl_mass/motion.py      <-> rl_mass/warp/motion.py
    rl_mass/joint_utils.py <-> rl_mass/warp/joint_utils.py

Custom kernels live under rl_mass/warp/kernels/.

The CPU rl_mass.* code is the bit-for-bit reference; the warp backend
trades fp64 exactness for batched GPU throughput (~hundreds of envs per
step). Regression tests in tests/test_warp_vs_cpu.py compare results.
"""

from rl_mass.warp.vec_env import WarpVecEnv

__all__ = ["WarpVecEnv"]
