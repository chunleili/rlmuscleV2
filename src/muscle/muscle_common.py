"""Shared pure-Python utilities and base class for muscle simulators.

Both the Taichi (muscle.py) and Warp (muscle_warp.py) backends import from here
to avoid code duplication.
"""

from pathlib import Path

import numpy as np

from muscle.config import SimConfig
from muscle.constraints import ConstraintBuilderMixin
from muscle.mesh_io import load_mesh, load_bone_mesh


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_bbox(pos):
    return np.array([pos.min(axis=0), pos.max(axis=0)])


def compute_fiber_stretches(pos, tet_indices, rest_matrices, fiber_dirs):
    """Compute per-tet fiber stretch from deformed positions (vectorized).

    Args:
        pos: Current vertex positions (N, 3).
        tet_indices: Tet connectivity (M, 4). pts[3] is reference vertex.
        rest_matrices: Inverse rest-pose matrices (M, 3, 3).
        fiber_dirs: Per-tet fiber directions (M, 3).

    Returns:
        Per-tet fiber stretch values (M,).
    """
    i0, i1, i2, i3 = (tet_indices[:, k] for k in range(4))
    # Edge vectors: (M, 3, 3) — columns are [p0-p3, p1-p3, p2-p3]
    Ds = np.stack([pos[i0] - pos[i3], pos[i1] - pos[i3], pos[i2] - pos[i3]], axis=-1)
    # Deformation gradient projected onto fiber: F @ fiber_dir
    F = np.einsum('mij,mjk->mik', Ds, rest_matrices)
    Fd = np.einsum('mij,mj->mi', F, fiber_dirs)
    return np.maximum(np.linalg.norm(Fd, axis=1), 1e-8)


def activation_ramp(t: float) -> float:
    """Piecewise activation over normalized time [0,1]: 0→0.5→1.0→0.7→0.3→0."""
    if t <= 0.2:
        return 0.0
    elif t <= 0.3:
        return 0.5
    elif t <= 0.5:
        return 1.0
    elif t <= 0.7:
        return 0.7
    elif t <= 0.8:
        return 0.3
    else:
        return 0.0


# ---------------------------------------------------------------------------
# MuscleSimBase — shared simulation logic (pure Python)
# ---------------------------------------------------------------------------

class MuscleSimBase(ConstraintBuilderMixin):
    """Base class for muscle simulation backends (Taichi / Warp).

    Subclasses must implement:
        _init_backend, _allocate_fields, _init_fields, _precompute_rest,
        _build_surface_tris, build_constraints, _create_bone_fields,
        integrate, update_velocities, clear, solve_constraints,
        apply_dP, update_attach_targets, reset, calc_vol_error
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._init_backend()

        self.constraint_configs = self.cfg.constraints if self.cfg.constraints else []

        print("Loading mesh from:", cfg.geo_path)
        mesh_data = load_mesh(cfg.geo_path)
        self.pos0_np, self.tet_np, self.v_fiber_np, self.v_tendonmask_np, self.geo = mesh_data
        self.n_verts = self.pos0_np.shape[0]
        print(f"Loaded mesh: {self.n_verts} vertices, {self.tet_np.shape[0]} tetrahedra.")

        print("Loading bone mesh:", cfg.bone_geo_path)
        self._load_bone(cfg.bone_geo_path)

        print("Allocating&initializing fields...")
        self._allocate_fields()
        self._init_fields()
        self._precompute_rest()
        self._build_surface_tris()
        # GPU parallelism: auto-enable Jacobi to avoid race conditions on CUDA
        is_gpu = self.pos.device.is_cuda
        self.use_jacobi = getattr(self.cfg, 'use_jacobi', is_gpu)
        self.use_colored_gs = getattr(self.cfg, 'use_colored_gs', False)
        self.build_constraints()
        self.contraction_ratio = getattr(self.cfg, 'contraction_ratio', 0.4)
        self.fiber_stiffness_scale = getattr(self.cfg, 'fiber_stiffness_scale', 10000.0)
        self.has_compressstiffness = getattr(self.cfg, 'HAS_compressstiffness', False)
        self.dt = self.cfg.dt / self.cfg.num_substeps
        self.step_cnt = 0

        self._init_renderer()
        print("All initialization done.")

    # -- hooks for subclasses -------------------------------------------------

    def _init_backend(self):
        raise NotImplementedError

    def _allocate_fields(self):
        raise NotImplementedError

    def _init_fields(self):
        raise NotImplementedError

    def _precompute_rest(self):
        raise NotImplementedError

    def _build_surface_tris(self):
        raise NotImplementedError

    # build_constraints: provided by subclass

    def _create_bone_fields(self):
        """Create backend-specific GPU fields from bone numpy data."""
        raise NotImplementedError

    def _init_renderer(self):
        """Optional: set up visualization backend."""
        pass

    def integrate(self):
        raise NotImplementedError

    def update_velocities(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def solve_constraints(self):
        raise NotImplementedError

    def apply_dP(self):
        raise NotImplementedError

    def update_attach_targets(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def calc_vol_error(self):
        raise NotImplementedError

    # -- shared pure-Python methods -------------------------------------------

    def _load_bone(self, target_path):
        """Load bone mesh data and create GPU fields."""
        if Path(target_path).exists():
            self.bone_geo, self.bone_pos, self.bone_indices_np, self.bone_muscle_ids = \
                load_bone_mesh(target_path)
            self._create_bone_fields()
        else:
            self.bone_geo = None
            self.bone_pos = np.zeros((0, 3), dtype=np.float32)
            self.bone_indices_np = np.zeros(0, dtype=np.int32)
            self.bone_muscle_ids = {}

    def load_bone_geo(self, target_path):
        """Return (bone_geo, bone_pos). Loads once, then returns cached data.

        Called by constraint builders (create_attach_constraints, etc.).
        """
        if not hasattr(self, 'bone_pos'):
            self._load_bone(target_path)
        return self.bone_geo, self.bone_pos

    def step(self):
        self.update_attach_targets()
        for _ in range(self.cfg.num_substeps):
            self.clear_forces()
            self.accumulate_active_fiber_force()
            self.integrate()
            self.clear()
            self.solve_constraints()
            if self.use_jacobi:
                self.apply_dP()
            self.post_smooth()
            self.freeze_near_inverted()
            self.repair_inverted_tets()
            self.update_velocities()

    def get_fps(self):
        if not hasattr(self, 'step_start_time') or not hasattr(self, 'step_end_time'):
            return 0.0
        dur = self.step_end_time - self.step_start_time
        return 1.0 / dur if dur != 0 else 0.0
