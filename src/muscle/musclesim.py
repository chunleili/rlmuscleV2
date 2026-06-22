"""MuscleSim: Warp XPBD muscle solver (fiber + Millard/DGF active force)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import warp as wp
import warp.render

from muscle.util.util import SimConfig
from muscle.util.mesh_io import build_surface_tris, load_mesh, load_bone_mesh
from muscle.fiber.millard import MillardCurve, MillardCurves
from muscle.defs import ConstraintType, FemFlag
from muscle.engine import (
    XpbdEngine,
    WarpRenderer,
    accumulate_active_fiber_force_kernel,
    calc_vol_error_kernel,
    count_inverted_tets_kernel,
    _clamp_force_by_accel_kernel,
    clear_reaction_kernel,
    compute_cell_tendon_mask_kernel,
    fill_float_kernel,
    integrate_kernel,
    lbs_apply_kernel,
    precompute_rest_kernel,
    update_attach_targets_kernel,
)
from muscle.constraint.solve_kernels import solve_pintotarget_kernel


class MuscleSim(XpbdEngine):
    """Warp-backed muscle simulator.

    Holds init + a hook-free step loop alongside all Warp kernels.
    Bone-coupled scenarios (LBS predictor → constraint corrector) compose
    their own per-substep loop using the public methods (integrate,
    solve_constraints, …) — see mg/mg.py (MultiGridSim).
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._init_backend()

        self.constraint_configs = self.cfg.constraints if self.cfg.constraints else []

        print("Loading mesh from:", cfg.geo_path)
        mesh_data = load_mesh(
            cfg.geo_path,
            prim_path=getattr(cfg, "geo_prim_path", None),
            prim_root=getattr(cfg, "geo_prim_root", None),
        )
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
        # Millard fiber constraint uses its OWN stiffness scale (separate from the shared
        # fiber_stiffness_scale used by fibernorm/dgf). Default smaller to reduce belly distortion.
        self.millard_stiffness_scale = getattr(self.cfg, 'millard_stiffness_scale', 1.0)
        self.has_compressstiffness = getattr(self.cfg, 'HAS_compressstiffness', False)
        self.dt = self.cfg.dt / self.cfg.num_substeps
        self.step_cnt = 0

        self._init_renderer()
        print("All initialization done.")

    def _load_bone(self, target_path):
        """Load bone mesh data and create GPU fields."""
        if Path(target_path).exists():
            bone_root = getattr(self.cfg, "bone_root", "/character/bone")
            self.bone_geo, self.bone_pos, self.bone_indices_np, self.bone_muscle_ids = \
                load_bone_mesh(target_path, bone_root=bone_root)
            self._create_bone_fields()
        else:
            self.bone_geo = None
            self.bone_pos = np.zeros((0, 3), dtype=np.float32)
            self.bone_indices_np = np.zeros(0, dtype=np.int32)
            self.bone_muscle_ids = {}

    def load_bone_geo(self, target_path):
        """Return (bone_geo, bone_pos). Loads once, then returns cached data."""
        if not hasattr(self, 'bone_pos'):
            self._load_bone(target_path)
        return self.bone_geo, self.bone_pos

    def build_program(self):
        """Default program: reproduces the standalone substep order exactly
        (see git history of step()). Phases are closures over self."""
        s = self
        def integrate_phase():
            s.clear_forces()
            s.accumulate_active_fiber_force()
            s.integrate()
            if hasattr(s, "apply_pintotarget"):
                s.apply_pintotarget()
            s.clear()
        def solve_phase():
            s.solve_constraints(run_extras=False)
            if s.use_jacobi:
                s.apply_dP()
            s.solve_extras()
        def finalize_phase():
            s.post_smooth()
            s.freeze_near_inverted()
            s.repair_inverted_tets()
            if hasattr(s, "apply_pintotarget"):
                s.apply_pintotarget()
            s.update_velocities()
        self.program = [integrate_phase, solve_phase, finalize_phase]
        if bool(getattr(self.cfg, "debug_per_substep", False)):
            def debug_phase():
                s._debug_check_substep(s._program_sub, s.cfg.num_substeps)
            self.program.append(debug_phase)

    def step(self):
        """Run one frame. Standalone (no bone coupling) substep loop, driven
        by the phase program built in build_program()."""
        self.update_attach_targets()
        if not hasattr(self, "program"):
            self.build_program()
        self.run_program()

    def _debug_check_substep(self, sub_i, n_sub):
        """Per-substep mesh-quality + state logging. cfg.debug_per_substep."""
        import logging
        from muscle.util.mesh_utils import (
            check_mesh_quality, MeshDistortionError,
        )
        log = logging.getLogger(__name__)
        pos_np = self.pos.numpy()
        # rest_matrix was built against tet_np[:, [3,0,1,2]] ordering — reuse
        # so det(F) here matches the solver's internal F.
        tet_idx = self.tet_np[:, [3, 0, 1, 2]]
        tet_poses = self.rest_matrix.numpy()
        step_id = getattr(self, "step_cnt", -1)
        tag = f"step={step_id} sub={sub_i + 1}/{n_sub}"
        try:
            detF = check_mesh_quality(pos_np, tet_idx, tet_poses, step=None)
            detF_lo = float(detF.min())
            detF_hi = float(detF.max())
        except MeshDistortionError as exc:
            log.error(f"{tag} MeshDistortionError: {exc}")
            return
        centroid = pos_np.mean(axis=0)
        try:
            vol_err = float(self.calc_vol_error())
        except Exception:
            vol_err = float("nan")
        log.info(
            f"{tag}  centroid=({centroid[0]:+.4f},{centroid[1]:+.4f},{centroid[2]:+.4f})  "
            f"vol_err={vol_err:+.4f}  detF=[{detF_lo:.4f},{detF_hi:.4f}]"
        )

    def get_fps(self):
        if not hasattr(self, 'step_start_time') or not hasattr(self, 'step_end_time'):
            return 0.0
        dur = self.step_end_time - self.step_start_time
        return 1.0 / dur if dur != 0 else 0.0

    @classmethod
    def from_procedural(cls, vertices, tets, fiber_dirs_per_tet,
                        bone_targets=None, *, constraint_configs,
                        dts, num_substeps=1, device="cpu",
                        density=1060.0, veldamping=0.02, gravity=0.0,
                        sigma0=0.0, lambda_opt=1.0,
                        contraction_ratio=0.0, fiber_stiffness_scale=1.0,
                        v_max_norm=0.0, d_damp=0.0):
        """Create a MuscleSim from procedural mesh data (no file loading).

        Args:
            vertices: Vertex positions (N, 3) float.
            tets: Tet connectivity (M, 4) int.
            fiber_dirs_per_tet: Per-tet fiber directions (M, 3) float.
            bone_targets: Optional attachment target positions (K, 3) float.
            constraint_configs: List of constraint config dicts, e.g.
                [{"type": "volume", "stiffness": 1e6, "dampingratio": 0.1}, ...].
            dts: Frame timestep [s] (substep dt = dts / num_substeps).
            num_substeps: Number of substeps per frame.
            device: Warp device string ("cpu" or "cuda:0").
            density: Material density [kg/m^3].
            veldamping: Velocity damping coefficient.
            gravity: Gravity magnitude [m/s^2].
            sigma0: Peak isometric stress [Pa] for explicit active fiber force.
            lambda_opt: Optimal fiber stretch ratio (l_opt / L_ref).
            contraction_ratio: Initial contraction ratio.
            fiber_stiffness_scale: Scaling factor for fiber constraint stiffness.
            v_max_norm: Normalizer for fiber velocity (V_max * lambda_opt).
            d_damp: Linear damping coefficient for fiber velocity term.

        Returns:
            Initialized MuscleSim instance.
        """
        from types import SimpleNamespace

        vertices = np.asarray(vertices, dtype=np.float32)
        tets = np.asarray(tets, dtype=np.int32)
        n_v = len(vertices)

        # Average per-tet fiber dirs to per-vertex
        fiber_dirs_per_tet = np.asarray(fiber_dirs_per_tet, dtype=np.float32)
        vf = np.zeros((n_v, 3), dtype=np.float32)
        vc = np.zeros(n_v, dtype=np.float32)
        for e, t in enumerate(tets):
            for vi in t:
                vf[vi] += fiber_dirs_per_tet[e]; vc[vi] += 1.0
        vc = np.maximum(vc, 1.0)
        vf /= vc[:, None]
        vf /= np.maximum(np.linalg.norm(vf, axis=1, keepdims=True), 1e-8)

        cfg = SimpleNamespace(
            geo_path="<procedural>", bone_geo_path="<none>",
            gui=False, render_mode="none",
            constraints=list(constraint_configs),
            dt=dts, num_substeps=num_substeps,
            gravity=gravity, density=density, veldamping=veldamping,
            contraction_ratio=contraction_ratio,
            fiber_stiffness_scale=fiber_stiffness_scale,
            HAS_compressstiffness=False, arch=device,
            save_image=False, pause=False, reset=False,
            show_wireframe=False,
            render_fps=24, color_bones=False, color_muscles="tendonmask",
            activation=0.0, nsteps=1,
            sigma0=sigma0, lambda_opt=lambda_opt,
            v_max_norm=v_max_norm, d_damp=d_damp,
        )

        wp.set_device(device)
        sim = object.__new__(cls)
        sim.cfg = cfg
        sim.constraint_configs = cfg.constraints

        sim.pos0_np = vertices
        sim.tet_np = tets
        sim.v_fiber_np = vf
        sim.v_tendonmask_np = None
        sim.geo = SimpleNamespace()
        sim.n_verts = n_v

        if bone_targets is not None:
            sim.bone_pos = np.asarray(bone_targets, dtype=np.float32)
        else:
            sim.bone_pos = np.zeros((0, 3), dtype=np.float32)
        sim.bone_geo = None
        sim.bone_indices_np = np.zeros(0, dtype=np.int32)
        sim.bone_muscle_ids = {}

        wp.init()
        sim._init_backend()
        sim._allocate_fields()
        sim._init_fields()
        sim._precompute_rest()
        sim._build_surface_tris()
        sim._create_bone_fields()

        is_gpu = not device.startswith("cpu")
        sim.use_jacobi = is_gpu
        sim.use_colored_gs = False
        sim.contraction_ratio = contraction_ratio
        sim.fiber_stiffness_scale = fiber_stiffness_scale
        sim.has_compressstiffness = False
        sim.dt = dts / num_substeps
        sim.step_cnt = 0
        sim.renderer = None

        sim.build_constraints()
        return sim

    def rebuild_constraints(self, extra_constraints=None):
        """Rebuild constraint arrays, optionally appending extra constraints.

        Useful for adding ConstraintType.ATTACH constraints after initial build.

        Args:
            extra_constraints: Optional list of constraint dicts to append.
        """
        all_cons = list(self.raw_constraints)
        if extra_constraints:
            all_cons.extend(extra_constraints)

        all_cons.sort(key=lambda c: c["type"])
        n = len(all_cons)
        self.n_cons = n
        self.cons_ranges = {}
        if n == 0:
            self._upload_cons([])
            self.reaction_accum = wp.zeros(1, dtype=wp.vec3)
            return

        prev, start = None, 0
        for i, c in enumerate(all_cons):
            c["cidx"] = i
            t = c["type"]
            if t != prev:
                if prev is not None:
                    self.cons_ranges[prev] = (start, i - start)
                start, prev = i, t
        self.cons_ranges[prev] = (start, n - start)

        self._upload_cons(all_cons)
        self.reaction_accum = wp.zeros(max(n, 1), dtype=wp.vec3)

    def _init_backend(self):
        wp.init()

    def _build_surface_tris(self):
        self.surface_tris = build_surface_tris(self.tet_np)
        self.n_tris = self.surface_tris.shape[0]
        self.surface_tris_flat = self.surface_tris.reshape(-1).astype(np.int32)

    def _allocate_fields(self):
        n_v = self.n_verts
        n_tet = self.tet_np.shape[0]

        self.pos = wp.zeros(n_v, dtype=wp.vec3)
        self.pprev = wp.zeros(n_v, dtype=wp.vec3)
        self.pos0 = wp.zeros(n_v, dtype=wp.vec3)
        self.vel = wp.zeros(n_v, dtype=wp.vec3)
        self.force = wp.zeros(n_v, dtype=wp.vec3)
        self.mass = wp.zeros(n_v, dtype=wp.float32)
        self.stopped = wp.zeros(n_v, dtype=wp.int32)
        self.v_fiber_dir = wp.zeros(n_v, dtype=wp.vec3)
        self.dP = wp.zeros(n_v, dtype=wp.vec3)
        self.dPw = wp.zeros(n_v, dtype=wp.float32)
        self.tet_indices = wp.zeros(n_tet, dtype=wp.vec4i)

        self.rest_volume = wp.zeros(n_tet, dtype=wp.float32)
        self.rest_matrix = wp.zeros(n_tet, dtype=wp.mat33)
        self.activation = wp.zeros(n_tet, dtype=wp.float32)

        # LBS optional: filled by LBSDriver via set_lbs_T_rel / apply_lbs.
        self._lbs_T_rel_field = None

    def _create_bone_fields(self):
        if self.bone_pos.shape[0] > 0:
            self.bone_pos_field = wp.from_numpy(self.bone_pos.astype(np.float32), dtype=wp.vec3)
            if self.bone_indices_np.shape[0] > 0:
                self.bone_indices_field = wp.from_numpy(self.bone_indices_np.astype(np.int32), dtype=wp.int32)
            else:
                self.bone_indices_field = None
        else:
            self.bone_pos_field = None
            self.bone_indices_field = None

    def build_constraints(self):
        print("Building constraints...")
        all_constraints, _dt_collect = self._collect_raw_constraints()

        # Sort by type for per-type kernel dispatch
        all_constraints.sort(key=lambda c: c['type'])

        n_cons = len(all_constraints)
        self.n_cons = n_cons
        self.cons_ranges = {}
        if n_cons > 0:
            prev_type = None
            start_idx = 0
            for i, c in enumerate(all_constraints):
                c['cidx'] = i
                ctype = c['type']
                if ctype != prev_type:
                    if prev_type is not None:
                        self.cons_ranges[prev_type] = (start_idx, i - start_idx)
                    start_idx = i
                    prev_type = ctype
            if prev_type is not None:
                self.cons_ranges[prev_type] = (start_idx, n_cons - start_idx)

            self._upload_cons(all_constraints)
        else:
            self._upload_cons([])
            self.raw_constraints = []

        print(f"Built {n_cons} constraints total. [{_dt_collect*1000:.0f}ms]")

        self.reaction_accum = wp.zeros(max(n_cons, 1), dtype=wp.vec3)

        # Initialize Millard curve arrays if needed (for constraints or active force)
        needs_millard = (ConstraintType.TETFIBERMILLARD in self.cons_ranges
                         or getattr(self.cfg, 'sigma0', 0.0) > 0)
        if needs_millard:
            self._init_millard_curves()

        # Initialize explicit active fiber force if sigma0 is configured
        sigma0 = getattr(self.cfg, 'sigma0', 0.0)
        if sigma0 > 0:
            self.sigma0 = sigma0
            self.lambda_opt = getattr(self.cfg, 'lambda_opt', 1.0)
            self.v_max_norm = getattr(self.cfg, 'v_max_norm', 0.0)
            self.d_damp = getattr(self.cfg, 'd_damp', 0.0)
            self._has_active_fiber_force = True
            print(f"  Active fiber force: sigma0={sigma0:.0f} Pa, "
                  f"lambda_opt={self.lambda_opt:.3f}, "
                  f"v_max_norm={self.v_max_norm:.3f}, d_damp={self.d_damp:.2f}")
        else:
            self.v_max_norm = 0.0
            self.d_damp = 0.0
            self._has_active_fiber_force = False

        # Build colored constraint groups for parallel Gauss-Seidel
        if self.use_colored_gs and n_cons > 0:
            self._build_colored_gs(all_constraints)

    def _init_millard_curves(self):
        """Initialize Millard 2012 curve coefficient arrays for GPU evaluation."""

        mc = MillardCurves()
        self._millard_curves = mc  # keep reference for CPU-side equilibrium solve

        def _upload_curve(curve):
            """Upload MillardCurve data to Warp arrays."""
            n_seg = len(curve.segments)
            x_coeffs = np.zeros((n_seg, 6), dtype=np.float32)
            y_coeffs = np.zeros((n_seg, 6), dtype=np.float32)
            F_coeffs = np.zeros((n_seg, 11), dtype=np.float32)
            seg_bounds = np.zeros(n_seg + 1, dtype=np.float32)

            for i, seg in enumerate(curve.segments):
                x_coeffs[i] = seg.x_coeffs.astype(np.float32)
                y_coeffs[i] = seg.y_coeffs.astype(np.float32)
                F_coeffs[i] = seg.F_coeffs.astype(np.float32)
                seg_bounds[i] = np.float32(seg.x_start)
            seg_bounds[n_seg] = np.float32(curve.segments[-1].x_end)

            return {
                'x_coeffs': wp.from_numpy(x_coeffs.flatten(), dtype=wp.float32),
                'y_coeffs': wp.from_numpy(y_coeffs.flatten(), dtype=wp.float32),
                'F_coeffs': wp.from_numpy(F_coeffs.flatten(), dtype=wp.float32),
                'seg_bounds': wp.from_numpy(seg_bounds, dtype=wp.float32),
                'n_segments': n_seg,
                'x_lo': curve.x_lo,
                'x_hi': curve.x_hi,
                'y_lo': curve.y_lo,
                'y_hi': curve.y_hi,
                'dydx_lo': curve.dydx_lo,
                'dydx_hi': curve.dydx_hi,
            }

        fl = _upload_curve(mc.fl)
        self._millard_fl_x_coeffs = fl['x_coeffs']
        self._millard_fl_y_coeffs = fl['y_coeffs']
        self._millard_fl_F_coeffs = fl['F_coeffs']
        # Energy offset: -eval_integral(0) so that Psi(lm) + offset > 0 for all lm > 0
        self._millard_fl_energy_offset = float(-mc.fl.eval_integral(0.0))
        self._millard_fl_seg_bounds = fl['seg_bounds']
        self._millard_fl_n_segments = fl['n_segments']
        self._millard_fl_x_lo = fl['x_lo']
        self._millard_fl_x_hi = fl['x_hi']
        self._millard_fl_y_lo = fl['y_lo']
        self._millard_fl_y_hi = fl['y_hi']
        self._millard_fl_dydx_lo = fl['dydx_lo']
        self._millard_fl_dydx_hi = fl['dydx_hi']

        fpe = _upload_curve(mc.fpe)
        self._millard_fpe_x_coeffs = fpe['x_coeffs']
        self._millard_fpe_y_coeffs = fpe['y_coeffs']
        self._millard_fpe_F_coeffs = fpe['F_coeffs']
        self._millard_fpe_seg_bounds = fpe['seg_bounds']
        self._millard_fpe_n_segments = fpe['n_segments']
        self._millard_fpe_x_lo = fpe['x_lo']
        self._millard_fpe_x_hi = fpe['x_hi']
        self._millard_fpe_y_lo = fpe['y_lo']
        self._millard_fpe_y_hi = fpe['y_hi']
        self._millard_fpe_dydx_lo = fpe['dydx_lo']
        self._millard_fpe_dydx_hi = fpe['dydx_hi']

        print(f"  Millard curves initialized: f_L ({mc.fl.segments.__len__()} seg), "
              f"f_PE ({mc.fpe.segments.__len__()} seg)")

    def _init_fields(self):
        self.pos0 = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.pos = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.vel.zero_()
        self.force.zero_()
        self.mass.zero_()
        self.stopped.zero_()
        self.tet_indices = wp.from_numpy(self.tet_np.astype(np.int32), dtype=wp.vec4i)

        if self.v_fiber_np is None:
            self.v_fiber_np = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (self.n_verts, 1))
        self.v_fiber_dir = wp.from_numpy(self.v_fiber_np.astype(np.float32), dtype=wp.vec3)

        n_tet = self.tet_np.shape[0]
        self.tendonmask = wp.zeros(n_tet, dtype=wp.float32)
        if self.v_tendonmask_np is not None:
            v_tendonmask_wp = wp.from_numpy(self.v_tendonmask_np.astype(np.float32), dtype=wp.float32)
            wp.launch(compute_cell_tendon_mask_kernel, dim=n_tet,
                      inputs=[self.tet_indices, v_tendonmask_wp, self.tendonmask])

        self.activation.zero_()
        self.total_rest_volume = wp.zeros(1, dtype=wp.float32)

        print("Initialized fields done.")

    def _rest_density(self):
        """Material density fed to ``precompute_rest_kernel`` (mass = density*V0).

        Seam for subclasses whose ``cfg`` carries total mass instead of an
        explicit density (e.g. the pin-to-target OneWaySim, which derives
        density = total_mass / sum(rest_volume)). Default reads ``cfg.density``.
        """
        return self.cfg.density

    def _precompute_rest(self):
        n_tet = self.tet_np.shape[0]
        self.total_rest_volume.zero_()
        self.mass.zero_()
        wp.launch(precompute_rest_kernel, dim=n_tet,
                  inputs=[self.pos0, self.tet_indices, self.rest_volume,
                          self.rest_matrix, self.mass, float(self._rest_density()),
                          self.total_rest_volume])
        # Clamp vertex mass to prevent instability from degenerate tets.
        mass_floor = getattr(self.cfg, 'mass_floor', 0.0)
        if mass_floor <= 0.0 and getattr(self.cfg, 'sigma0', 0.0) > 0.0:
            mass_np = self.mass.numpy()
            mass_floor = float(np.median(mass_np[mass_np > 0])) * 0.01
        if mass_floor > 0.0:
            mass_np = self.mass.numpy()
            n_clamped = int((mass_np < mass_floor).sum())
            if n_clamped > 0:
                mass_np = np.maximum(mass_np, mass_floor)
                self.mass = wp.from_numpy(mass_np, dtype=wp.float32)
                print(f"  Mass floor applied: {mass_floor:.2e} kg ({n_clamped} vertices clamped)")

    def _init_renderer(self):
        self.renderer = None
        if self.cfg.gui and self.cfg.render_mode in ("human", "usd"):
            if self.cfg.render_mode == "usd":
                stage_path = getattr(self.cfg, "output_path", None) or "output/bicep_oneway.usd"
                os.makedirs(os.path.dirname(stage_path) or ".", exist_ok=True)
            else:
                stage_path = None
            self.renderer = WarpRenderer(mode=self.cfg.render_mode, stage_path=stage_path)

    def reset(self):
        self.pos = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.pprev = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.vel.zero_()
        self.force.zero_()
        self.activation.zero_()
        self.clear()
        self.step_cnt = 1

    def update_attach_targets(self):
        if (
            hasattr(self, "bone_pos_field")
            and self.bone_pos_field is not None
            and self.bone_pos_field.shape[0] > 0
            and self.n_cons > 0
            and (
                len(getattr(self, "attach_constraints", []) or []) > 0
                or len(getattr(self, "pintotarget_constraints", []) or []) > 0
                or len(getattr(self, "distanceline_constraints", []) or []) > 0
            )
        ):
            wp.launch(update_attach_targets_kernel, dim=self.n_cons,
                      inputs=[self.c_type, self.c_pts, self.c_restvector,
                              self.bone_pos_field, self.n_cons])

    def clear_forces(self):
        """Zero the per-vertex force accumulator."""
        self.force.zero_()

    def accumulate_active_fiber_force(self):
        """Compute explicit active fiber force per tet, scatter to vertex force field.

        Uses tau = sigma0 * (a * f_FL(r) * f_V(v_norm) + d_damp * v_norm)
        where r = lambda_f / lambda_opt, v_norm = dlm_dt / v_max_norm.
        See docs/notes/xpbd_fem_fiber_lm.md for derivation.
        """
        if not getattr(self, '_has_active_fiber_force', False):
            return
        n_tet = self.tet_np.shape[0]
        wp.launch(accumulate_active_fiber_force_kernel, dim=n_tet, inputs=[
            self.tet_indices, self.pos, self.vel, self.force,
            self.rest_matrix, self.rest_volume,
            self.v_fiber_dir, self.activation,
            self.sigma0, self.lambda_opt,
            self.v_max_norm, self.d_damp,
            self._millard_fl_x_coeffs, self._millard_fl_y_coeffs,
            self._millard_fl_seg_bounds, self._millard_fl_n_segments,
            self._millard_fl_x_lo, self._millard_fl_x_hi,
            self._millard_fl_y_lo, self._millard_fl_y_hi,
            self._millard_fl_dydx_lo, self._millard_fl_dydx_hi,
        ])
        # Clamp per-vertex acceleration to prevent instability on degenerate tets.
        max_accel = getattr(self.cfg, 'max_accel', 2000.0)
        if max_accel > 0.0:
            wp.launch(_clamp_force_by_accel_kernel, dim=self.n_verts,
                      inputs=[self.force, self.mass, wp.float32(max_accel)])

    def save_predicted_positions(self, vertex_ids):
        """Save predicted positions after integrate() and before solve()."""
        self._predicted_pos = self.pos.numpy()[vertex_ids].copy()
        self._predicted_ids = vertex_ids

    def get_insertion_force(self, insertion_vids, dts):
        """Get net muscle force on bone from insertion constraint correction.

        Requires save_predicted_positions() called after integrate().
        """
        pos_np = self.pos.numpy()
        corrected = pos_np[insertion_vids]
        predicted = self._predicted_pos

        mass_np = self.mass.numpy()
        masses = mass_np[insertion_vids]
        dts2 = dts * dts

        corrections = corrected.astype(np.float64) - predicted.astype(np.float64)
        forces = masses[:, None].astype(np.float64) * corrections / dts2

        return (-forces.sum(axis=0)).astype(np.float32)

    def set_lbs_T_rel(self, T_rel_flat_np):
        """Upload (n_bones * 12,) float32 row-major T_rel buffer to GPU.

        Allocates the Warp array on first call.
        """
        if getattr(self, "_lbs_T_rel_field", None) is None:
            self._lbs_T_rel_field = wp.from_numpy(
                T_rel_flat_np.astype(np.float32), dtype=wp.float32)
        else:
            self._lbs_T_rel_field.assign(T_rel_flat_np.astype(np.float32))

    def apply_lbs(self, vertex_bone_field, blend: float = 1.0):
        """Overwrite pos with LBS target = pos0 @ T_rel[vertex_bone[v]],
        lerped from current pos by ``blend``.

        Caller must have called ``set_lbs_T_rel`` first.
        """
        if self._lbs_T_rel_field is None:
            return
        wp.launch(lbs_apply_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pos0, vertex_bone_field,
                          self._lbs_T_rel_field, float(blend)])

    def calc_vol_error(self):
        total_vol = wp.zeros(1, dtype=wp.float32)
        n_tet = self.tet_np.shape[0]
        wp.launch(calc_vol_error_kernel, dim=n_tet,
                  inputs=[self.tet_indices, self.pos, self.rest_volume, total_vol])
        total_vol_np = total_vol.numpy()[0]
        total_rest_np = self.total_rest_volume.numpy()[0]
        if total_rest_np == 0.0:
            return 0.0
        return (total_vol_np - total_rest_np) / total_rest_np

    def count_inverted(self) -> int:
        """Number of tets with det(F) <= 0 (inverted) at the current pos."""
        cnt = wp.zeros(1, dtype=wp.int32)
        wp.launch(count_inverted_tets_kernel, dim=self.tet_np.shape[0],
                  inputs=[self.tet_indices, self.pos, self.rest_matrix, cnt])
        return int(cnt.numpy()[0])

    def clear_reaction(self):
        if self.n_cons > 0:
            wp.launch(clear_reaction_kernel, dim=max(self.n_cons, 1),
                      inputs=[self.reaction_accum])

    def _mark_pintotarget_stopped(self):
        """Mark all pintotarget source verts as `stopped=1`.

        Effect: integrate_kernel skips them (no velocity, no pos drift), and
        constraint solvers see inv_mass=0 (they act as kinematic anchors).
        Position is then driven *only* by solve_pintotarget_kernel each substep.
        Without this, apply_pintotarget creates huge artificial velocities
        (pinned vert jumps several cm per substep) which integrate then over-
        shoots — fatal under GMG (restrict feeds the overshoot into coarse).
        """
        src_idx = np.array([c["pts"][0] for c in self.pintotarget_constraints],
                           dtype=np.int32)
        if src_idx.size == 0:
            return
        stopped_np = self.stopped.numpy()
        stopped_np[src_idx] = 1
        self.stopped = wp.from_numpy(stopped_np, dtype=wp.int32)

    def _capturable_substep(self):
        """One muscle PBD substep — all Warp kernels, no CPU sync.

        Safe for CUDA graph capture: no .numpy(), no wp.from_numpy(), no
        wp.array() allocations. All conditionals (use_jacobi, use_colored_gs,
        post_smooth_iters, etc.) are resolved at capture time.
        """
        self.clear_forces()
        # self.accumulate_active_fiber_force()
        self.integrate()
        self.clear()
        self.update_attach_targets()
        # self.clear_reaction()
        self.solve_constraints(run_extras=False)
        if self.use_jacobi:
            self.apply_dP()
        # Extras (collision) AFTER apply_dP so ARAP/volume settles first;
        # otherwise the stale pre-solve ARAP gradient gets added on top of
        # the collision push and pulls penetrated verts back inward.
        self.solve_extras()
        self.post_smooth()
        # self.freeze_near_inverted()
        # self.repair_inverted_tets()
        self.update_velocities()

    def render(self):
        if self.renderer is None:
            return
        pos_np = self.pos.numpy()
        bone_pos_np = None
        bone_idx = None
        if hasattr(self, 'bone_pos_field') and self.bone_pos_field is not None:
            bone_pos_np = self.bone_pos_field.numpy()
            bone_idx = self.bone_indices_np if hasattr(self, 'bone_indices_np') else None
        sim_time = self.step_cnt * self.cfg.dt
        self.renderer.update(sim_time, pos_np, self.surface_tris_flat, bone_pos_np, bone_idx)

    def run(self):
        self.step_cnt = 1

        if self.renderer is not None:
            def on_key(symbol, modifiers):
                import pyglet
                if symbol == pyglet.window.key.R:
                    self.reset()
                    return pyglet.event.EVENT_HANDLED
                if symbol == pyglet.window.key.SPACE:
                    self.cfg.pause = not self.cfg.pause
                    return pyglet.event.EVENT_HANDLED
            self.renderer.register_key_press_callback(on_key)

        while self.step_cnt <= self.cfg.nsteps:
            if self.renderer is not None and not self.renderer.is_running():
                break
            if self.cfg.reset:
                self.reset()
                self.step_cnt = 1
                self.cfg.reset = False
            if not self.cfg.pause:
                wp.launch(fill_float_kernel, dim=self.activation.shape[0],
                          inputs=[self.activation, self.cfg.activation])
                self.step_start_time = time.perf_counter()
                self.step()
                self.step_cnt += 1
                self.step_end_time = time.perf_counter()
            self.render()

        if self.renderer is not None:
            self.renderer.save()
