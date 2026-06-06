"""Select the env class by config `skeleton_format`.

Default (absent or "dart") -> MassEnv (humanoid/BVH/DART path).
"mujoco_native" -> BimanualEnv (native-MJCF fixed-base path).
"""
from __future__ import annotations

import json


def make_env(config_path: str):
    with open(config_path) as f:
        fmt = json.load(f).get("skeleton_format", "dart")
    if fmt == "mujoco_native":
        from rl_mass.bimanual_env import BimanualEnv
        return BimanualEnv(config_path)
    if fmt == "dart":
        from rl_mass.env import MassEnv
        return MassEnv(config_path)
    raise ValueError(f"unknown skeleton_format: {fmt!r}")
