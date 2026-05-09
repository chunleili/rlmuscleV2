"""Warp-based renderer for muscle simulation (OpenGL interactive + USD offline)."""

import numpy as np
import warp as wp
import warp.render


class WarpRenderer:
    """Wraps wp.render.OpenGLRenderer or wp.render.UsdRenderer."""

    def __init__(self, mode: str = "human", stage_path: str = "muscle_sim.usd"):
        self.mode = mode
        self.renderer = None
        if mode == "human":
            self.renderer = wp.render.OpenGLRenderer(vsync=True)
        elif mode == "usd":
            self.renderer = wp.render.UsdRenderer(stage_path)
        self.sim_time = 0.0

    def is_running(self) -> bool:
        if self.renderer is None:
            return True
        if self.mode == "human":
            return self.renderer.is_running()
        return True

    def update(self, sim_time: float, pos_np: np.ndarray, tri_indices: np.ndarray,
               bone_pos_np: np.ndarray = None, bone_indices: np.ndarray = None):
        if self.renderer is None:
            return
        self.sim_time = sim_time

        self.renderer.begin_frame(sim_time)
        self.renderer.render_mesh(
            name="muscle",
            points=pos_np,
            indices=tri_indices,
            colors=(0.8, 0.3, 0.2),
        )
        if bone_pos_np is not None and bone_indices is not None and len(bone_indices) > 0:
            self.renderer.render_mesh(
                name="bone",
                points=bone_pos_np,
                indices=bone_indices,
                colors=(0.9, 0.9, 0.85),
            )
        self.renderer.end_frame()

    def save(self):
        if self.renderer is not None and self.mode == "usd":
            self.renderer.save()

    def register_key_press_callback(self, callback):
        if self.mode == "human" and self.renderer is not None:
            self.renderer.register_key_press_callback(callback)
