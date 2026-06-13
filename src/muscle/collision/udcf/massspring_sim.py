"""MassSpringSim — Gaia S01 dynamic MassSpring as an XpbdEngine program.

Composes the validated MassSpringTet material (collision/udcf/massspring.py)
with boundary + DCD collision, reproducing Gaia runStepCPU substep order:
  integrate -> spring -> volume(hard) -> boundary -> inversion -> DCD -> update_vel
Output is bit-identical to the pre-merge bespoke driver (see
tests/golden/capture_engine_goldens.py::capture_massspring)."""
from __future__ import annotations

import numpy as np
import warp as wp

from muscle.muscle_warp import XpbdEngine
from muscle.collision.udcf.massspring import MassSpringTet, MassSpringParams
from muscle.collision.udcf.solver import UdcfDcdCollider, UdcfParams
from muscle.collision.udcf.kernel import box_boundary_kernel
from muscle.util.geo_io import GeoFile
from muscle.util.mesh_io import build_surface_tris


class MassSpringSim(XpbdEngine):
    def __init__(self, ms, dcd, n_substeps, bounds_lo, bounds_hi,
                 mu_s, mu_d, no_inversion=False, device="cuda:0"):
        self.ms = ms
        self.dcd = dcd
        self.n_substeps = int(n_substeps)
        self.bounds_lo = bounds_lo
        self.bounds_hi = bounds_hi
        self.mu_s = float(mu_s)
        self.mu_d = float(mu_d)
        self.no_inversion = bool(no_inversion)
        self.device = device
        self.build_program()

    # expose engine-standard handles
    @property
    def pos(self):
        return self.ms.pos

    @property
    def vel(self):
        return self.ms.vel

    def build_program(self):
        s = self
        def integrate_phase():
            s.ms.integrate()
            s.ms.solve_material()                       # spring -> volume(hard)
            wp.launch(box_boundary_kernel, dim=s.ms.n_verts,
                      inputs=[s.ms.pos, s.ms.pprev, s.bounds_lo, s.bounds_hi,
                              wp.float32(s.mu_s), wp.float32(s.mu_d)],
                      device=s.device)
        self.constraint_solvers = []
        self.constraint_solvers.append(
            lambda: (None if s.no_inversion else s.ms.solve_inversion()))
        if s.dcd is not None:
            self.constraint_solvers.append(lambda: s.dcd.solve(0))
        def collide_phase():
            for fn in s.constraint_solvers:
                fn()
        def finalize_phase():
            s.ms.update_velocities()
        self.program = [integrate_phase, collide_phase, finalize_phase]

    def step(self):
        for _ in range(self.n_substeps):
            for phase in self.program:
                phase()

    @classmethod
    def u_drop(cls, geo_path, *, dt=1.0 / 60.0, n_substeps=20, density=3.0,
               gravity=-9.8, init_vel_y=-5.0, translate_y=0.2,
               spring_compliance=1e-4, spring_damping=0.01,
               mu_s=0.5, mu_d=0.4, thickness=1e-3, no_dcd=False,
               no_inversion=False, init_verts=None, init_vel=None,
               device="cuda:0"):
        """Build the S01 U-drop exactly as the bespoke driver did.

        When ``init_verts`` is provided (a recovery-pose vertex array), those
        verts become the MassSpringTet REST state — reproducing the pre-refactor
        warm-start semantics where the recovery pose was passed directly as the
        ``verts`` argument so there is zero pre-strain.  ``translate_y`` is
        ignored in that case.  When ``init_vel`` is provided it overrides
        ``init_vel_y``.
        """
        g = GeoFile(geo_path)
        tets = np.asarray(g.tet_indices, np.int32)
        if init_verts is not None:
            # Warm-start: recovery pose is the rest state (no pre-strain).
            verts = np.asarray(init_verts, np.float32).copy()
        else:
            verts = np.asarray(g.positions, np.float32).copy()
            verts[:, 1] += translate_y
        n_v = verts.shape[0]
        msp = MassSpringParams(spring_compliance=spring_compliance,
                               spring_damping=spring_damping,
                               density=density, gravity=gravity)
        ms = MassSpringTet(verts, tets, dt / n_substeps, params=msp, device=device)
        if init_vel is not None:
            ms.vel.assign(np.asarray(init_vel, np.float32))
        elif init_vel_y != 0.0:
            v0 = np.zeros((n_v, 3), np.float32); v0[:, 1] = init_vel_y
            ms.vel.assign(v0)
        surf = build_surface_tris(tets, verts).astype(np.int32)
        dcd = None
        if not no_dcd:
            mp = max(65536, surf.shape[0] * (surf.shape[0] - 1) // 2)
            dv = np.array([[1e4, 1e4, 1e4], [1e4 + 1, 1e4, 1e4],
                           [1e4, 1e4, 1e4 + 1]], np.float32)
            dcd = UdcfDcdCollider(ms.pos, ms.inv_mass, surface_tris=surf,
                                  static_verts=dv, static_tris=np.array([[0, 1, 2]], np.int32),
                                  params=UdcfParams(allow_self=True, max_pairs=mp,
                                                    thickness=thickness, jacobi_avg=False,
                                                    gauss_seidel=True, max_push=0.0),
                                  device=device)
        lo = wp.vec3(-2.0, 0.0, -2.0); hi = wp.vec3(2.0, 5.0e4, 2.0)
        sim = cls(ms, dcd, n_substeps, lo, hi, mu_s, mu_d,
                  no_inversion=no_inversion, device=device)
        sim.surf = surf
        return sim
