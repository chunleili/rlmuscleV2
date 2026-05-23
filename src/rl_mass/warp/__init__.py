"""GPU batched backend for rl_mass via mujoco_warp + NVIDIA Warp kernels.

`WarpGPUEnv` is a single env over N worlds — all state lives as `(N, ...)`
tensors and every op is one batched op. It is intentionally NOT a vec-env
(no list of `MassEnv` to loop over per step).
"""

from rl_mass.warp.gpu_env import WarpGPUEnv

__all__ = ["WarpGPUEnv"]
