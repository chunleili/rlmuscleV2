"""MassSpringSim: tet-edge mass-spring + hard-volume XPBD (no fibers)."""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.render

from muscle.util.util import SimConfig
from muscle.util.mesh_io import build_surface_tris, load_mesh, load_bone_mesh
from muscle.defs import ConstraintType
from muscle.engine import XpbdEngine, build_unique_edges


class MassSpringSim(XpbdEngine):
    """Gaia S01 dynamic MassSpring as an XpbdEngine program.

    The Gaia U-drop material — stiff edge springs, a hard det(F)=1 volume
    constraint and an inverted-tet untangle — expressed entirely with engine
    primitives: edges are ``ConstraintType.DISTANCE`` constraints (reusing the engine distance
    XPBD math), volume/inversion are the migrated ``ConstraintType.TETVOLUMEHARD`` /
    ``ConstraintType.TETINVERSION`` constraint types, and ``integrate`` / ``update_velocities``
    / graph-colored Gauss-Seidel are the shared ``XpbdEngine`` machinery.
    Boundary + DCD self-collision run as program phases between the material
    and finalize passes, reproducing Gaia's substep order::

      integrate -> spring -> volume(hard) -> boundary -> inversion -> DCD -> update_vel
    """

    def __init__(self, verts, tets, *, dt=1.0 / 60.0, n_substeps=20,
                 density=3.0, gravity=-9.8, spring_compliance=1e-4,
                 spring_damping=0.01, inversion_multiplier=0.02,
                 exp_vel_damping=0.95, const_vel_damping=0.02,
                 mu_s=0.5, mu_d=0.4, bounds_lo=(-2.0, 0.0, -2.0),
                 bounds_hi=(2.0, 5.0e4, 2.0), thickness=1e-3,
                 no_inversion=False, no_dcd=False, init_vel=None,
                 init_vel_y=0.0, device="cuda:0"):
        from types import SimpleNamespace
        from muscle.collision.udcf.kernel import box_boundary_kernel
        from muscle.collision.udcf.solver import UdcfDcdCollider, UdcfParams
        from muscle.util.mesh_io import build_surface_tris

        self.device = device
        dt_sub = dt / n_substeps
        self.dt = float(dt_sub)
        self.cfg = SimpleNamespace(veldamping=0.0, gravity=gravity,
                                   num_substeps=int(n_substeps))
        self._gravity_vec = wp.vec3(0.0, gravity, 0.0)
        self.use_jacobi = False
        self.use_colored_gs = True
        self.has_compressstiffness = False
        self.no_inversion = bool(no_inversion)
        self.ms_inversion_mult = float(inversion_multiplier)
        # Gaia velocity damping (consumed by update_velocities).
        self._vel_exp_damp = float(exp_vel_damping ** dt_sub)
        self._vel_const_damp = float(const_vel_damping * dt_sub)

        verts = np.asarray(verts, np.float32).copy()
        tets = np.asarray(tets, np.int32).copy()
        n_v = verts.shape[0]
        self.n_verts = n_v
        self.tet_np = tets

        # Gaia winds tets positive (det(Dm) > 0); U.geo winds the other way.
        # Flip orientation so rest volume is positive and inversion doesn't
        # fire on every rest tet (see fascia/houdini-lefthand-winding note).
        v = verts
        Dm0 = np.stack([v[tets[:, 1]] - v[tets[:, 0]],
                        v[tets[:, 2]] - v[tets[:, 0]],
                        v[tets[:, 3]] - v[tets[:, 0]]], axis=2)
        if np.median(np.linalg.det(Dm0)) < 0:
            tets[:, [2, 3]] = tets[:, [3, 2]]

        Dm = np.stack([v[tets[:, 1]] - v[tets[:, 0]],
                       v[tets[:, 2]] - v[tets[:, 0]],
                       v[tets[:, 3]] - v[tets[:, 0]]], axis=2)
        DmInv = np.linalg.inv(Dm)
        rest_vol = np.linalg.det(Dm) / 6.0
        mass = np.zeros(n_v, np.float64)
        for k in range(4):
            np.add.at(mass, tets[:, k], 0.25 * rest_vol * density)
        inv_mass = np.where(mass > 0, 1.0 / np.where(mass > 0, mass, 1.0), 0.0)

        edges = build_unique_edges(tets)
        rest_len = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)

        dev = device
        self.pos = wp.array(v, dtype=wp.vec3, device=dev)
        self.pprev = wp.array(v, dtype=wp.vec3, device=dev)
        self.vel = wp.zeros(n_v, dtype=wp.vec3, device=dev)
        self.force = wp.zeros(n_v, dtype=wp.vec3, device=dev)
        self.mass = wp.array(mass.astype(np.float32), dtype=wp.float32, device=dev)
        self.inv_mass = wp.array(inv_mass.astype(np.float32), dtype=wp.float32, device=dev)
        self.stopped = wp.zeros(n_v, dtype=wp.int32, device=dev)
        self.dP = wp.zeros(n_v, dtype=wp.vec3, device=dev)
        self.dPw = wp.zeros(n_v, dtype=wp.float32, device=dev)
        self.ms_dm_inv = wp.array(DmInv.astype(np.float32), dtype=wp.mat33, device=dev)
        self.ms_rest_vol = wp.array(rest_vol.astype(np.float32), dtype=wp.float32, device=dev)

        if init_vel is not None:
            self.vel.assign(np.asarray(init_vel, np.float32))
        elif init_vel_y != 0.0:
            v0 = np.zeros((n_v, 3), np.float32)
            v0[:, 1] = init_vel_y
            self.vel.assign(v0)

        # --- constraints: springs (ConstraintType.DISTANCE) + hard volume + inversion -------
        spring_stiffness = 1.0 / spring_compliance   # engine alpha = 1/(k dt^2)
        all_constraints = []
        for e in range(edges.shape[0]):
            all_constraints.append(dict(
                type=ConstraintType.DISTANCE, pts=[int(edges[e, 0]), int(edges[e, 1]), -1, -1],
                stiffness=spring_stiffness, dampingratio=float(spring_damping),
                tetid=-1, L=[0.0, 0.0, 0.0], restlength=float(rest_len[e]),
                restvector=[0.0, 0.0, 0.0, 0.0], restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0))
        for t in range(tets.shape[0]):
            pts = [int(tets[t, 0]), int(tets[t, 1]), int(tets[t, 2]), int(tets[t, 3])]
            common = dict(pts=pts, stiffness=1.0, dampingratio=0.0, tetid=t,
                          L=[0.0, 0.0, 0.0], restlength=0.0,
                          restvector=[0.0, 0.0, 0.0, 0.0], restdir=[0.0, 0.0, 0.0],
                          compressionstiffness=-1.0)
            all_constraints.append(dict(type=ConstraintType.TETVOLUMEHARD, **common))
            all_constraints.append(dict(type=ConstraintType.TETINVERSION, **{**common}))

        self.n_cons = len(all_constraints)
        for i, c in enumerate(all_constraints):
            c['cidx'] = i
        self._build_colored_gs(all_constraints)   # builds self.c_* arrays + color_type_ranges

        # Per-color ranges split by type.  Gaia sweeps ALL springs, then ALL
        # volume tets, then (after boundary) ALL inversion tets — so each gets
        # its own full colored pass rather than interleaving per color.
        def _ranges_of(ctype):
            return [{ctype: tr[ctype]} for tr in self.color_type_ranges if ctype in tr]
        self._spring_ranges = _ranges_of(ConstraintType.DISTANCE)
        self._volume_ranges = _ranges_of(ConstraintType.TETVOLUMEHARD)
        self._inversion_ranges = _ranges_of(ConstraintType.TETINVERSION)

        # --- boundary + DCD self-collision -----------------------------------
        self._box_boundary_kernel = box_boundary_kernel
        self.bounds_lo = wp.vec3(*bounds_lo)
        self.bounds_hi = wp.vec3(*bounds_hi)
        self.mu_s = float(mu_s)
        self.mu_d = float(mu_d)
        self.surf = build_surface_tris(tets, v).astype(np.int32)
        self.dcd = None
        if not no_dcd:
            mp = max(65536, self.surf.shape[0] * (self.surf.shape[0] - 1) // 2)
            self.dcd = UdcfDcdCollider(
                self.pos, self.inv_mass, surface_tris=self.surf,
                params=UdcfParams(allow_self=True, max_pairs=mp,
                                  thickness=thickness, jacobi_avg=False,
                                  gauss_seidel=True, max_push=0.0),
                device=device)
        self.build_program()

    def build_program(self):
        s = self

        def integrate_phase():
            s.integrate()
            s.clear()                       # reset spring XPBD lambda each substep

        def material_phase():
            for tr in s._spring_ranges:        # all springs (Gaia projectConstraints)
                s._dispatch_constraints(tr)
            for tr in s._volume_ranges:        # then all hard-volume tets
                s._dispatch_constraints(tr)

        def boundary_phase():
            wp.launch(s._box_boundary_kernel, dim=s.n_verts,
                      inputs=[s.pos, s.pprev, s.bounds_lo, s.bounds_hi,
                              wp.float32(s.mu_s), wp.float32(s.mu_d)],
                      device=s.device)

        def inversion_phase():
            if s.no_inversion:
                return
            for tr in s._inversion_ranges:
                s._dispatch_constraints(tr)

        def dcd_phase():
            if s.dcd is not None:
                s.dcd.solve(0)

        def finalize_phase():
            s.update_velocities()

        self.program = [integrate_phase, material_phase, boundary_phase,
                        inversion_phase, dcd_phase, finalize_phase]

    def step(self):
        self.run_program()

    @classmethod
    def u_drop(cls, geo_path, *, dt=1.0 / 60.0, n_substeps=20, density=3.0,
               gravity=-9.8, init_vel_y=-5.0, translate_y=0.2,
               spring_compliance=1e-4, spring_damping=0.01,
               mu_s=0.5, mu_d=0.4, thickness=1e-3, no_dcd=False,
               no_inversion=False, init_verts=None, init_vel=None,
               device="cuda:0"):
        """Build the Gaia S01 U-drop.  ``init_verts`` (a recovery pose) becomes
        the rest state (zero pre-strain), in which case ``translate_y`` is
        ignored; ``init_vel`` overrides ``init_vel_y``."""
        from muscle.util.geo_io import GeoFile
        g = GeoFile(geo_path)
        tets = np.asarray(g.tet_indices, np.int32)
        if init_verts is not None:
            verts = np.asarray(init_verts, np.float32).copy()
        else:
            verts = np.asarray(g.positions, np.float32).copy()
            verts[:, 1] += translate_y
        return cls(verts, tets, dt=dt, n_substeps=n_substeps, density=density,
                   gravity=gravity, spring_compliance=spring_compliance,
                   spring_damping=spring_damping, mu_s=mu_s, mu_d=mu_d,
                   thickness=thickness, no_inversion=no_inversion, no_dcd=no_dcd,
                   init_vel=init_vel, init_vel_y=init_vel_y, device=device)
