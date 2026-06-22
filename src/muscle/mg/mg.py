"""Warp ARAP-HPBD one-way pin-to-target solver.

Originally ported from realtimeMuscle lion sim; generic over any
restpos.geo + cons_<frame>.geo pipeline (lion, minotaur, ...).

Algorithm-equivalent (not strict bit-identical) to
``realtimeMuscle/engine/arap_hpbd.py`` running the lion_nomg.json config:
quasi-static, no gravity, no multigrid, ``nsubsteps=10``,
``fine_iterations=1``, SOR ``omega=0.1``, ``mu=1e9``.

This module is intentionally self-contained — it does NOT inherit from or
modify ``muscle.engine.MuscleSim`` so the fiber/Millard/DGF pipeline
stays untouched. The only shared symbol is ``ssvd_fn`` (oriented SVD)
imported from ``muscle.engine``.
"""
from __future__ import annotations

import warp as wp

from .kernel import *



# ─── Python-side class glue ────────────────────────────────────────────────

import json
import os
import sys
from pathlib import Path

import numpy as np

from muscle.musclesim import MuscleSim
from muscle.defs import ConstraintType
from muscle.util.geo_io import GeoFile, read_cons_frame
from muscle.util.util import _check_unzipped



class _CoarseEngine(MuscleSim):
    """SoA sub-solver on the multigrid coarse cage.

    Reuses the full MuscleSim / XpbdEngine XPBD pipeline so the (optional, off by
    default) coarse smoother runs the SAME dispatch as the fine level — ARAP-SVD
    + optional volume + an optional per-vertex attach-to-projection anchor — with
    no HPBD ``pos_mid`` path. Built only when ``coarse_iterations > 0``.

    ``attach_compliance_np`` (per coarse vert) adds an ATTACH constraint anchoring
    each vert toward a target set per substep via :meth:`set_anchor` (the bone-
    following Galerkin projection); pass ``None`` for a free ARAP relax.
    """

    def __init__(self, pos_np, tet_np, total_mass, mu, dt, use_volume,
                 volume_stiffness, attach_compliance_np=None, device=None):
        self.cfg = {}
        self.total_mass = float(total_mass)
        self.mu = float(mu)
        self.dt = float(dt)
        self._device = device
        self.use_jacobi = True
        self.use_colored_gs = False
        self.has_compressstiffness = False
        self.fiber_stiffness_scale = 1.0
        self.millard_stiffness_scale = 1.0
        self.n_verts = int(pos_np.shape[0])
        self.pos0_np = np.ascontiguousarray(pos_np, np.float32)
        self.tet_np = np.asarray(tet_np, dtype=np.int64)
        self.v_fiber_np = None
        self.v_tendonmask_np = None
        self._use_volume = bool(use_volume)
        self._volume_stiffness = float(volume_stiffness)
        self._attach_comp = (np.asarray(attach_compliance_np, np.float32)
                             if attach_compliance_np is not None else None)
        self._attach_offset = None

        self._allocate_fields()
        self._build_coarse_constraints()
        self._init_fields()
        self._precompute_rest()
        self.pprev = wp.clone(self.pos)

    def _rest_density(self):
        _, vols, valid = self._batch_compute_tet_rest_matrices()
        return self.total_mass / float(max(vols[valid].sum(), 1e-30))

    def integrate(self):
        wp.copy(self.pprev, self.pos)

    def _build_coarse_constraints(self):
        cons = self.create_tet_arap_constraints(
            {'stiffness': self.mu}, ctype=ConstraintType.TETARAPSVD)
        if self._use_volume:
            cons += self.create_tet_volume_constraint(
                {'stiffness': self._volume_stiffness})
        if self._attach_comp is not None:
            for v in range(self.n_verts):
                k = 1.0 / float(max(self._attach_comp[v], 1e-30))
                cons.append(dict(
                    type=ConstraintType.ATTACH, pts=[v, -1, -1, -1],
                    stiffness=k, compressionstiffness=-1.0, dampingratio=0.0,
                    tetid=-1, L=[0.0, 0.0, 0.0], restlength=0.0,
                    restvector=[0.0, 0.0, 0.0, 1.0], restdir=[0.0, 0.0, 0.0]))
        # Finalize like MuscleSim.build_constraints: sort by type -> per-type
        # ranges -> upload. Attach verts stay in vertex order within their range.
        cons.sort(key=lambda c: c['type'])
        self.n_cons = len(cons)
        self.cons_ranges = {}
        prev = None
        start = 0
        for i, c in enumerate(cons):
            c['cidx'] = i
            t = c['type']
            if t != prev:
                if prev is not None:
                    self.cons_ranges[prev] = (start, i - start)
                start = i
                prev = t
        if prev is not None:
            self.cons_ranges[prev] = (start, self.n_cons - start)
        self._upload_cons(cons)
        self.reaction_accum = wp.zeros(max(self.n_cons, 1), dtype=wp.vec3)
        if ConstraintType.ATTACH in self.cons_ranges:
            self._attach_offset = self.cons_ranges[ConstraintType.ATTACH][0]

    def set_anchor(self, target_wp):
        """Point the attach constraints at the current projection target
        (coarse_target): writes c_restvector[off:off+NV, :3]. Host roundtrip —
        the coarse smoother is experimental / off by default."""
        if self._attach_offset is None:
            return
        off = self._attach_offset
        rv = self.c_restvector.numpy()
        rv[off:off + self.n_verts, 0:3] = target_wp.numpy()
        self.c_restvector.assign(rv)

    def smooth(self, pos_wp, n_iters):
        """Relax the cage from pos_wp for n_iters XPBD sweeps; write back to pos_wp."""
        wp.copy(self.pos, pos_wp)
        self.integrate()
        for _ in range(int(n_iters)):
            self.clear()
            self.solve_constraints(run_extras=False)
            self.apply_dP()
        wp.copy(pos_wp, self.pos)


class MultiGridSim(MuscleSim):
    """Per-frame ARAP-HPBD driver for the pin-to-target workflow.

    Lifecycle (one frame):
        sim.load_frame_targets(frame)  # reads cons_<frame>.geo target_pos
        sim.apply_pintotarget()        # one-shot hard overwrite (per-frame)
        for _ in range(nsubsteps):     # 10 for lion
            sim.substep()              # ARAP, fine_iters=1 for lion
    """

    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg = cfg
        self.mu = float(cfg["mu"])
        self.omega = float(cfg["omega"])
        self.dt_frame = float(cfg["delta_t"])
        self.nsubsteps = int(cfg["nsubsteps"])
        self.fine_iters = int(cfg["fine_iterations"])
        self.total_mass = float(cfg["total_mass"])
        self.geo_dir = str(cfg["geo_dir"])
        self.cons_dir = str(cfg.get("cons_dir", self.geo_dir))
        self.rest_path = str(cfg.get("rest_path", str(Path(self.geo_dir) / "restpos.geo")))
        # Master switch. When False the fascia layer is fully disabled and the
        # solver is byte-for-byte the pre-fascia pin-to-target sim (no merged
        # state, no pin_active mask, no anchor/glue). Defaults to False so any
        # config lacking the key reproduces the original behaviour exactly.
        self.use_fascia = bool(cfg.get("use_fascia", False))
        self.fascia_path = cfg.get("fascia_path") if self.use_fascia else None
        if self.fascia_path is not None and not Path(self.fascia_path).is_absolute():
            self.fascia_path = str(Path.cwd() / self.fascia_path)
        self.fascia_rest_scale = float(cfg.get("fascia_rest_scale", 0.95))
        self.fascia_glue_iters = int(cfg.get("fascia_glue_iterations", self.fine_iters))
        # Soft-pin stiffness for FREED muscle verts (fraction toward bone target
        # per substep). Keeps freed verts tracking gross body motion so they
        # don't tear loose when the per-frame target jumps, while still letting
        # the fascia glue pull them inward. 0 disables (pure free).
        self.fascia_soft_pin = float(cfg.get("fascia_soft_pin", 0.5))
        self.dt = self.dt_frame / self.nsubsteps
        self.end_frame = int(cfg.get("end_frame", 90))
        self.start_frame = int(cfg.get("start_frame", 1))
        self._device = device

        self.use_multigrid = bool(cfg.get("use_multigrid", False))
        self.coarse_iters = int(cfg.get("coarse_iterations", 0))
        # Accept either `coarse_rest_path` (preferred) or legacy
        # `coarse_model_path` (matches realtimeMuscle scene JSONs).
        self.coarse_rest_path = cfg.get("coarse_rest_path",
                                        cfg.get("coarse_model_path", None))
        if self.coarse_rest_path is not None:
            self.coarse_rest_path = str(self.coarse_rest_path)
        # Conforming per-muscle voxel cage baked by tools/gen_coarse_cage.py.
        # When set it REPLACES coarse_rest_path: a true enclosing cage (every fine
        # vert inside a same-muscle coarse tet, prolong weights in [0,1]) instead
        # of the surface-fit lowres (61.8% of fine verts extrapolated -> spikes).
        self.coarse_npz_path = cfg.get("coarse_npz_path", None)
        if self.coarse_npz_path is not None:
            self.coarse_npz_path = str(self.coarse_npz_path)
        # Soft anchor strength for the coarse ARAP/volume smoother: each coarse
        # iter pulls the cage a fraction back toward its bone-following projection,
        # so the free smoother can't drift / blow up the stiff coarse volume term.
        self.coarse_anchor = float(cfg.get("coarse_anchor", 0.0))
        # Coarse attach compliances: stiff at tendons (bone anchor), soft at belly.
        self.coarse_attach_tendon_compliance = float(
            cfg.get("coarse_attach_tendon_compliance", 1.0e-7))
        self.coarse_attach_belly_compliance = float(
            cfg.get("coarse_attach_belly_compliance",
                    cfg.get("pin_attach_compliance", 1.0e-1)))
        # Opt-in convergence probe: when true, the global volume-weighted ARAP
        # residual RMS is appended to self.residual_log after each fine iter
        # (and once at iter 0 = the post-pre_phase initial guess). Used to
        # compare multigrid vs non-MG convergence; off by default (adds a sync).
        self.debug_residual = bool(cfg.get("debug_residual", False))
        self.residual_log = []
        self._resid_buf = None

        # Hydrostatic volume constraint (incompressibility). ARAP-SVD matches
        # shape only; the fine loop adds a det(F)=1 XPBD constraint at stiffness
        # volume_stiffness. ON by default — incompressibility is intrinsic to
        # muscle/soft tissue; set "use_volume": false to recover pure ARAP-HPBD.
        self.use_volume = bool(cfg.get("use_volume", True))
        self.volume_stiffness = float(cfg.get("volume_stiffness", 1.0e10))

        # Optional PASSIVE fiber constraint. The humanV2 asset (restpos.geo)
        # carries the per-vertex rest fiber direction in the ``materialW`` point
        # attribute, so the two passive Houdini fiber models are supported:
        #   "fibernorm"  -> solve_fibernorm_kernel  (linear TETFIBERNORM)
        #   "aniso_arap" -> solve_aniso_arap_kernel (quadratic HOBAK fiber)
        # The active "fibermillard" model still needs per-tet activation (from an
        # RL trajectory) that the asset lacks, so it remains gated off.
        self.use_fiber = bool(cfg.get("use_fiber", False))
        self.fiber_model = str(cfg.get("fiber_model", "fibernorm"))
        self.fiber_stiffness = float(cfg.get("fiber_stiffness", 1.0e8))
        self.contraction_ratio = float(cfg.get("contraction_ratio", 0.4))
        if self.use_fiber and self.fiber_model not in ("fibernorm", "aniso_arap"):
            raise NotImplementedError(
                f"fiber_model={self.fiber_model!r} not supported. The OneWayGmg "
                "solver implements the PASSIVE Houdini fiber models 'fibernorm' "
                "and 'aniso_arap' (both read the materialW fiber field). The "
                "active 'fibermillard' model needs per-tet activation, which the "
                "humanV2 asset does not carry — bake it in to enable that."
            )

        _check_unzipped(self.rest_path)
        rest = GeoFile(self.rest_path)
        self.NV = self.n_verts = int(rest.positions.shape[0])
        self.NT = int(rest.tet_indices.shape[0])

        # Time-invariant per-vertex muscle attributes, cached for USD baking
        # (materialW = fiber direction, tendonmask, optional Cd) and named-muscle
        # targeting (muscle_id). Stored as numpy.
        self.materialW_np = (rest.point_attr("materialW").astype(np.float32)
                             if rest.has_point_attr("materialW") else None)
        self.tendonmask_np = (rest.point_attr("tendonmask").astype(np.float32)
                              if rest.has_point_attr("tendonmask") else None)
        self.cd_np = (rest.point_attr("Cd").astype(np.float32)
                      if rest.has_point_attr("Cd") else None)
        self.muscle_id_np = (rest.prim_attr("muscle_id")
                             if rest.has_prim_attr("muscle_id") else None)
        self.muscle_id_strings = (rest.prim_attr_strings("muscle_id")
                                  if rest.has_prim_attr("muscle_id") else [])

        # ── SoA constraint pipeline (inherited from MuscleSim / XpbdEngine) ──
        # The HPBD per-tet state (pos_mid, B, alpha, L, inv_mass, vol_*, fiber_*)
        # is gone; the solver now runs the colored/Jacobi XPBD path on dP/dPw +
        # the c_* constraint table. Feed the builder + _allocate_fields the data
        # they read, then let build_constraints() upload the table.
        self.pos0_np = rest.positions.astype(np.float32)
        self.tet_np = np.asarray(rest.tet_indices, dtype=np.int64)
        self.v_fiber_np = (self.materialW_np.copy()
                           if (self.use_fiber and self.materialW_np is not None)
                           else None)
        self.v_tendonmask_np = self.tendonmask_np

        self.use_jacobi = bool(cfg.get("use_jacobi", True))
        self.use_colored_gs = bool(cfg.get("use_colored_gs", False))
        self.has_compressstiffness = False
        self.fiber_stiffness_scale = 1.0
        self.millard_stiffness_scale = 1.0

        if self.use_fiber:
            if self.materialW_np is None:
                raise ValueError(
                    "use_fiber=true but restpos.geo has no 'materialW' point "
                    "attribute (the fiber direction field).")
            if self.materialW_np.shape[0] != self.NV:
                raise ValueError(
                    f"materialW has {self.materialW_np.shape[0]} verts, "
                    f"expected NV={self.NV}.")
            if self.fiber_model not in ("fibernorm", "aniso_arap"):
                raise NotImplementedError(
                    f"fiber_model={self.fiber_model!r} not wired to the SoA "
                    "dispatch; supported: 'fibernorm' (ConstraintType.TETFIBERNORM) "
                    "and 'aniso_arap' (ConstraintType.TETARAPANISO).")

        # Constraint set: ARAP-SVD shape match (always) + optional volume + fiber.
        # ARAP stiffness = mu (the old HPBD ARAP stiffness); volume / fiber take
        # their configured stiffness. Targets / pins stay kinematic (apply_pintotarget).
        self.constraint_configs = [
            dict(type="tetarapsvd", name="arap", stiffness=self.mu),
        ]
        if self.use_volume:
            self.constraint_configs.append(
                dict(type="volume", name="volume", stiffness=self.volume_stiffness))
        if self.use_fiber:
            # fibernorm -> linear TETFIBERNORM; aniso_arap -> quadratic HOBAK
            # along-fiber ARAP (TETARAPANISO). Both read the materialW fiber field.
            _ftype = "aniso_arap" if self.fiber_model == "aniso_arap" else "fiber"
            self.constraint_configs.append(
                dict(type=_ftype, name="fiber", stiffness=self.fiber_stiffness))

        # Allocate base SoA fields, build + upload the constraint table, then
        # fill geometry via the reused MuscleSim helpers: _init_fields sets
        # pos0/pos/tet_indices/v_fiber_dir/tendonmask/activation, _precompute_rest
        # fills rest_matrix + per-vertex mass (density via the _rest_density seam).
        self._allocate_fields()
        self.build_constraints()
        self._init_fields()
        self._precompute_rest()
        self.pprev = wp.clone(self.pos)
        self._fiber_nt = self.NT  # muscle NT (fascia is a separate sub-solver)

        # Back-compat aliases for the humanV2 driver / USD exporters. They expect
        # a (NT,4) tet-index array `tet_idx` and a per-vertex `inv_mass`; the SoA
        # core stores tets as `tet_indices` (vec4i) and per-vertex `mass`. `nvm`
        # (muscle-only vertex count) == NV here: fascia is a SEPARATE sub-solver,
        # so its verts never enter this table.
        self.tet_idx = self.tet_indices
        self.nvm = self.NV
        _m = self.mass.numpy()
        self.inv_mass = wp.from_numpy(
            np.where(_m > 0.0, 1.0 / _m, 0.0).astype(np.float32),
            dtype=wp.float32, device=device)

        # Pin-to-target: topology (pts) is constant across all cons_<frame>.geo
        # files; only target_pos varies.
        cons0_path = str(Path(self.cons_dir) / f"cons_{self.start_frame}.geo")
        _check_unzipped(cons0_path)
        cons0 = GeoFile(cons0_path)
        self.NCONS = int(cons0.pts.shape[0])
        self.pts = wp.array(cons0.pts, dtype=wp.int32, device=device)
        self._pin_pts_np = np.asarray(cons0.pts, dtype=np.int64)
        self.pin_active = None  # rebuilt by Fascia (glue-ring free) when fascia is active
        rest_offset_np = (rest.positions[cons0.pts] - cons0.target_pos).astype(np.float32)
        self.rest_offset = wp.array(rest_offset_np, dtype=wp.vec3, device=device)
        self.target_pos = wp.array(cons0.target_pos.copy(), dtype=wp.vec3, device=device)

        # Optional PARTIAL pinning. humanV2 is fully pinned (pts == all NV): every
        # vert is hard-overwritten to a per-frame target each substep, and those
        # targets self-overlap, so the belly is forced through neighbouring
        # muscles and no collision push can hold. With tendon_pin_threshold set,
        # only verts whose tendonmask exceeds it (the tendon attachment regions)
        # stay HARD-pinned; the belly is freed to a soft pull toward target
        # (pin_soft_stiffness) so it tracks gross body motion while yielding to
        # collision. Reuses the fascia free/soft-pin path (mutually exclusive with
        # fascia, which rebuilds pin_active from its own glue ring below).
        self.tendon_pin_threshold = cfg.get("tendon_pin_threshold", None)
        if self.tendon_pin_threshold is not None:
            tmask = rest.point_attr("tendonmask").astype(np.float32)
            active = (tmask[self._pin_pts_np]
                      > float(self.tendon_pin_threshold)).astype(np.int32)
            self.pin_active = wp.array(active, dtype=wp.int32, device=device)
            self.fascia_soft_pin = float(
                cfg.get("pin_soft_stiffness", self.fascia_soft_pin))
            print(f"[partial-pin] tendonmask>{self.tendon_pin_threshold}: "
                  f"{int(active.sum())}/{len(active)} hard pins, "
                  f"rest soft (stiffness={self.fascia_soft_pin})")

        # Belly attach mode: tendons (pin_active==1) stay hard-pinned; the belly
        # (pin_active==0) is tied to its bone target by a compliant XPBD point
        # constraint solved in the iteration loop (attach_target_kernel) instead
        # of the position-blend soft pin. Force-based, so collision/fascia can
        # compete with it. Requires pin_active (set via tendon_pin_threshold).
        self.pin_belly_attach = bool(cfg.get("pin_belly_attach", False))
        self.pin_attach_compliance = float(cfg.get("pin_attach_compliance", 1e-6))
        self.attach_L = None
        if self.pin_belly_attach:
            if self.pin_active is None:
                raise ValueError("pin_belly_attach=true needs tendon_pin_threshold "
                                 "(to define the hard tendon vs belly split)")
            # MG + attach: free coarse ARAP would drift off the (unanchored)
            # restricted tendon positions and the prolonged belly — which the
            # fine level no longer hard-pins — accumulates that drift → NaN.
            # Fix (pre_phase): coarse is FULL-pinned to the bone target each
            # substep (restrict the dense fine target onto coarse, copy it in,
            # skip coarse ARAP). Coarse becomes a bone-driven scaffold; prolong
            # hands the belly a coarse-smoothed bone-following start, then the
            # fine attach + collision deviate it. See _init_coarse / pre_phase.
            print(f"[belly-attach] XPBD point attach for belly verts "
                  f"(compliance={self.pin_attach_compliance:g}); soft-blend off")

        if self.use_multigrid:
            if self.coarse_npz_path is not None:
                # The conforming voxel cage is built for the full-pin scaffold
                # path (restrict via nearest-vert gather, no coarse ARAP). The
                # free-ARAP path would need the tet barycentric c2f map, which
                # the npz doesn't carry.
                if not self.pin_belly_attach:
                    raise ValueError(
                        "coarse_npz_path requires pin_belly_attach=true "
                        "(the cage feeds the full-pin scaffold path)")
                self._init_coarse_cage()
            elif self.coarse_rest_path is None:
                raise ValueError(
                    "use_multigrid=true but no coarse_rest_path / "
                    "coarse_model_path in config"
                )
            else:
                self._init_coarse(rest.positions)

        # Fascia: a standalone shell sub-solver (its own MuscleSim: ARAP + volume)
        # coupled by anchor + two-way glue. It builds itself in __init__ (loads the
        # fascia geo, pairs glue/anchor, frees the glued muscle verts from the pin).
        # The muscle SoA table is left untouched — fascia is NOT merged in.
        self.fascia = None
        if self.fascia_path:
            from muscle.fascia.fascia import Fascia
            self.fascia = Fascia(self)

        # Fine belly-attach (built last so pin_active reflects tendon_pin / fascia):
        # each freed belly vert gets a compliant ATTACH toward its per-frame bone
        # target, appended to the SoA constraint table + refreshed each frame.
        if self.pin_belly_attach:
            self._build_fine_belly_attach()
            self._update_belly_attach_targets()


    # ------------------------------------------------------------------

    def _finalize_constraints(self, cons):
        """Sort a constraint dict list by type, rebuild cons_ranges, upload c_*
        arrays + reaction_accum, and (if enabled) the colored GS groups. Mirrors
        the tail of MuscleSim.build_constraints so the table can be re-finalized
        after appending belly-attach constraints."""
        cons.sort(key=lambda c: c['type'])
        self.n_cons = len(cons)
        self.cons_ranges = {}
        prev = None
        start = 0
        for i, c in enumerate(cons):
            c['cidx'] = i
            t = c['type']
            if t != prev:
                if prev is not None:
                    self.cons_ranges[prev] = (start, i - start)
                start = i
                prev = t
        if prev is not None:
            self.cons_ranges[prev] = (start, self.n_cons - start)
        self.raw_constraints = cons
        self._upload_cons(cons)
        self.reaction_accum = wp.zeros(max(self.n_cons, 1), dtype=wp.vec3)
        if self.use_colored_gs and self.n_cons > 0:
            self._build_colored_gs(cons)

    def _build_fine_belly_attach(self):
        """Append a per-belly-vert ATTACH constraint (target filled per frame)."""
        active = (self.pin_active.numpy() if self.pin_active is not None
                  else np.ones(self.NCONS, np.int32))
        k = 1.0 / float(max(self.pin_attach_compliance, 1e-30))
        belly = np.where(active == 0)[0].astype(np.int64)
        self._belly_cons_idx = belly                 # cons-frame index per attach
        cons = list(self.raw_constraints)
        for i in belly:
            v = int(self._pin_pts_np[i])
            cons.append(dict(
                type=ConstraintType.ATTACH, pts=[v, -1, -1, -1], stiffness=k,
                compressionstiffness=-1.0, dampingratio=0.0, tetid=-1,
                L=[0.0, 0.0, 0.0], restlength=0.0,
                restvector=[0.0, 0.0, 0.0, 1.0], restdir=[0.0, 0.0, 0.0]))
        self._finalize_constraints(cons)
        self._belly_attach_off = (self.cons_ranges[ConstraintType.ATTACH][0]
                                  if ConstraintType.ATTACH in self.cons_ranges
                                  else None)

    def _update_belly_attach_targets(self):
        """Refresh the belly ATTACH targets = target_pos + rest_offset (per frame)."""
        off = getattr(self, "_belly_attach_off", None)
        if off is None or len(self._belly_cons_idx) == 0:
            return
        idx = self._belly_cons_idx
        tgt = self.target_pos.numpy()[idx] + self.rest_offset.numpy()[idx]
        rv = self.c_restvector.numpy()
        rv[off:off + len(idx), 0:3] = tgt
        self.c_restvector.assign(rv)

    def _rest_density(self):
        """Density = total_mass / sum(rest volume). The pin-to-target cfg carries
        a total mass, not an explicit density (the MuscleSim default)."""
        _, volumes, valid = self._batch_compute_tet_rest_matrices()
        return self.total_mass / float(max(volumes[valid].sum(), 1e-30))

    def integrate(self):
        """Quasi-static predict: no gravity / inertia. Each substep just snapshots
        the start positions into pprev so the XPBD velocity-damping term vanishes
        and the solve is a pure compliance projection (pin-to-target is kinematic,
        applied separately). Overrides XpbdEngine.integrate (which reads
        cfg.gravity — absent from this dict cfg)."""
        wp.copy(self.pprev, self.pos)

    # ------------------------------------------------------------------

    def _init_coarse(self, fine_rest_positions: np.ndarray) -> None:
        """Load coarse mesh, allocate per-coarse fields, build f↔c barycentric map.

        Both mappings are computed in rest pose: fine vert i → containing
        coarse tet (cage_idx, uvw) for prolongation; coarse vert i →
        containing fine tet (cage_idx_c2f, uvw_c2f) for restriction.
        """
        device = self._device
        _check_unzipped(self.coarse_rest_path)
        coarse = GeoFile(self.coarse_rest_path)
        self.coarse_NV = int(coarse.positions.shape[0])
        self.coarse_NT = int(coarse.tet_indices.shape[0])

        self.coarse_pos     = wp.array(coarse.positions.copy(), dtype=wp.vec3, device=device)
        self.coarse_tet_idx = wp.array(coarse.tet_indices, dtype=wp.vec4i, device=device)

        # Coarse SoA smoother engine (free ARAP relax, no anchor) — only when the
        # optional smoother is enabled. The default projection-only V-cycle
        # (coarse_iters=0) needs none of the elastic fields.
        self._coarse = None
        if self.coarse_iters > 0:
            self._coarse = _CoarseEngine(
                coarse.positions, coarse.tet_indices, self.total_mass, self.mu,
                self.dt, use_volume=False, volume_stiffness=self.volume_stiffness,
                attach_compliance_np=None, device=device)

        # Rest-pose barycentric maps (one-shot brute force).
        fine_pos_rest = wp.array(fine_rest_positions.copy(), dtype=wp.vec3, device=device)

        # fine → coarse: each fine vert i lives in some coarse tet
        self.cage_idx = wp.zeros(self.NV, dtype=wp.int32, device=device)
        self.uvw      = wp.zeros(self.NV, dtype=wp.vec3,  device=device)
        wp.launch(
            compute_barycentric_kernel,
            dim=self.NV,
            inputs=[fine_pos_rest, self.coarse_pos, self.coarse_tet_idx,
                    self.cage_idx, self.uvw],
            device=device,
        )

        # coarse → fine: each coarse vert i lives in some fine tet
        self.cage_idx_c2f = wp.zeros(self.coarse_NV, dtype=wp.int32, device=device)
        self.uvw_c2f      = wp.zeros(self.coarse_NV, dtype=wp.vec3,  device=device)
        wp.launch(
            compute_barycentric_kernel,
            dim=self.coarse_NV,
            inputs=[self.coarse_pos, fine_pos_rest, self.tet_indices,
                    self.cage_idx_c2f, self.uvw_c2f],
            device=device,
        )
        wp.synchronize()

        # Capture the muscle fine vertex count for multigrid prolongation. self.NV
        # is muscle-only throughout (fascia is a separate sub-solver, never merged),
        # so mg_fine_nv == self.NV; kept as its own field for clarity at the prolong
        # launch (the cage_idx/uvw maps cover exactly these verts).
        self.mg_fine_nv = self.NV

        # Coarse full-pin scaffold (only used when pin_belly_attach): the dense
        # fine bone target restricted onto the coarse mesh each substep, copied
        # straight into coarse_pos so coarse never drifts off the bone.
        self.coarse_target = wp.zeros(self.coarse_NV, dtype=wp.vec3, device=device)
        self.fine_target_dense = wp.zeros(self.mg_fine_nv, dtype=wp.vec3, device=device)

    def _init_coarse_cage(self) -> None:
        """Load the conforming per-muscle cage + baked transfer maps from the npz
        produced by tools/gen_coarse_cage.py (--method inflate, the default).
        Restrict is the Galerkin least-squares projection (P^T P)^-1 P^T; prolong
        is the f->c barycentric. This projection-only path (coarse_iterations=0)
        is the production default: it hands the fine solve a smooth bone-following
        start with no measurable downside.

        The coarse ARAP (+volume) elastic fields below are built UNCONDITIONALLY so
        the optional coarse smoother (coarse_iterations>0, in pre_phase) can run,
        but that smoother is OFF by default and currently a NET LOSS: relaxing the
        cage pulls it off the clean Galerkin projection toward its own rest shape,
        corrupting the prolonged start (measured: any coarse_iters>0 blows the fine
        ARAP residual up ~100x; see docs/humanV2_cage_multigrid.md). The fields and
        kernels are kept for a future fiber-aware coarse solve that may want to
        relax the cage along the muscle fiber direction rather than just project."""
        device = self._device
        # Auto-bake on first run: the cage npz is a gitignored asset, so a fresh
        # checkout won't have it. Rather than fail, invoke tools/gen_coarse_cage.py
        # once (inflate cage, ~90s fTetWild) against the SAME muscle restpos the
        # sim loaded, then load the result. Subsequent runs hit the cached npz.
        if not os.path.exists(self.coarse_npz_path):
            import subprocess
            script = Path(__file__).resolve().parents[3] / "tools" / "gen_coarse_cage.py"
            print(f"[coarse-cage] {self.coarse_npz_path} missing — baking now "
                  f"(inflate, ~90s fTetWild); cached after first run...")
            os.makedirs(os.path.dirname(self.coarse_npz_path) or ".", exist_ok=True)
            subprocess.run(
                [sys.executable, str(script),
                 "--out", self.coarse_npz_path, "--rest", self.rest_path],
                check=True)
            if not os.path.exists(self.coarse_npz_path):
                raise RuntimeError(
                    f"coarse cage bake reported success but {self.coarse_npz_path} "
                    f"still missing; run {script} manually")
        d = np.load(self.coarse_npz_path)
        npz_fine_nv = int(d["mg_fine_nv"])
        if npz_fine_nv != self.NV:
            raise ValueError(
                f"coarse cage built for fine NV={npz_fine_nv} but this sim has "
                f"NV={self.NV}; regenerate tools/gen_coarse_cage.py against the "
                f"same restpos.geo")
        self.coarse_NV = int(d["coarse_pos"].shape[0])
        self.coarse_NT = int(d["coarse_tet"].shape[0])
        self.coarse_pos = wp.array(d["coarse_pos"], dtype=wp.vec3, device=device)
        self.coarse_tet_idx = wp.array(d["coarse_tet"], dtype=wp.vec4i, device=device)

        # Coarse SoA smoother engine (ARAP + optional volume + attach-to-projection)
        # — only when the optional smoother is enabled. The default projection-only
        # Galerkin V-cycle (coarse_iters=0) needs none of the elastic fields. The
        # attach compliance geometrically blends stiff (tendon) to soft (belly) by
        # the baked tendon fraction, mirroring the fine pin/attach split.
        self._coarse = None
        if self.coarse_iters > 0:
            ct = (d["coarse_tendon"] if "coarse_tendon" in d
                  else np.zeros(self.coarse_NV, np.float32)).astype(np.float64)
            comp = np.exp((1.0 - ct) * np.log(self.coarse_attach_belly_compliance)
                          + ct * np.log(self.coarse_attach_tendon_compliance))
            self._coarse = _CoarseEngine(
                d["coarse_pos"], d["coarse_tet"], self.total_mass, self.mu,
                self.dt, use_volume=self.use_volume,
                volume_stiffness=self.volume_stiffness,
                attach_compliance_np=comp.astype(np.float32), device=device)
        # f->c prolongation (per fine muscle vert).
        self.cage_idx = wp.array(d["cage_idx"], dtype=wp.int32, device=device)
        self.uvw = wp.array(d["uvw"], dtype=wp.vec3, device=device)
        # c->f restriction: Galerkin projection Cdef=(P^T P)^-1 P^T T (smooth, low
        # strain). Measured ~6x less injected strain and 3-6x faster convergence
        # than a nearest-vert gather; gather was removed after that comparison.
        self.minv_flat = wp.array(d["minv_flat"], dtype=wp.float32, device=device)
        self.blk_base = wp.array(d["blk_base"], dtype=wp.int32, device=device)
        self.blk_dim = wp.array(d["blk_dim"], dtype=wp.int32, device=device)
        self.blk_matoff = wp.array(d["blk_matoff"], dtype=wp.int32, device=device)
        self.coarse_b = wp.zeros(self.coarse_NV, dtype=wp.vec3, device=device)
        self.mg_fine_nv = self.NV
        self.coarse_target = wp.zeros(self.coarse_NV, dtype=wp.vec3, device=device)
        self.fine_target_dense = wp.zeros(self.mg_fine_nv, dtype=wp.vec3, device=device)
        print(f"[coarse-cage] loaded {self.coarse_npz_path}: "
              f"coarse NV={self.coarse_NV} NT={self.coarse_NT}; "
              f"Galerkin lsq restrict, coarse_iters={self.coarse_iters}"
              f"{' (relax: experimental/off by default)' if self.coarse_iters > 0 else ''}"
              f"{'+vol' if self.use_volume else ''}")

    # ------------------------------------------------------------------

    def tet_mask_for_muscles(self, names) -> np.ndarray:
        """Boolean per-(muscle)-tet mask selecting tets in the named muscles.

        ``names`` is a string or list of substrings matched against
        ``muscle_id_strings`` (e.g. ``"bicepsBrachialis"`` matches both L_/R_).
        Length = muscle NT (= ``_fiber_nt``), aligned with ``fiber_activation``.
        """
        if isinstance(names, str):
            names = [names]
        if self.muscle_id_np is None:
            raise ValueError("restpos.geo has no 'muscle_id' prim attribute.")
        sel_ids = [i for i, s in enumerate(self.muscle_id_strings)
                   if any(n in s for n in names)]
        return np.isin(self.muscle_id_np, sel_ids)

    def set_fiber_activation(self, acti_np) -> None:
        """Upload a per-(muscle)-tet activation array (length = ``_fiber_nt``).

        Writes the SoA per-tet ``activation`` field read by the TETFIBERNORM
        dispatch. Muscle tets are the first ``_fiber_nt`` entries."""
        if not self.use_fiber:
            raise RuntimeError("set_fiber_activation requires use_fiber=True.")
        acti = np.ascontiguousarray(acti_np, np.float32)
        if acti.shape[0] == self._fiber_nt and self._fiber_nt != self.NT:
            full = self.activation.numpy()
            full[:self._fiber_nt] = acti
            self.activation.assign(full)
        else:
            self.activation.assign(acti)

    def preload_targets(self, frames=None) -> None:
        """Parse every frame's cons target_pos once and hold it in RAM.

        Per-frame ``.geo`` is large JSON; parsing it inside the step loop is the
        dominant cost. Preloading lifts all of it out of the loop (and, via the
        ``.npy`` cache, off disk after the first ever run), so stepping is pure
        GPU work — independent of any warm/cold cache state.
        """
        if frames is None:
            frames = range(self.start_frame, self.end_frame + 1)
        self._target_stack = {f: read_cons_frame(self.cons_dir, f) for f in frames}

    def load_frame_targets(self, frame: int) -> None:
        stack = getattr(self, "_target_stack", None)
        if stack is None or frame not in stack:
            # lazy fallback for callers that did not preload (also memoizes)
            if stack is None:
                stack = self._target_stack = {}
            stack[frame] = read_cons_frame(self.cons_dir, frame)
        # In-place copy preserves the GPU buffer identity / device.
        self.target_pos.assign(stack[frame])
        # Belly ATTACH targets follow the per-frame bone target.
        self._update_belly_attach_targets()

    def apply_pintotarget(self) -> None:
        if getattr(self, "pin_active", None) is not None:
            wp.launch(
                pintotarget_overwrite_masked_kernel,
                dim=self.NCONS,
                inputs=[self.pts, self.target_pos, self.rest_offset,
                        self.pin_active, self.pos],
                device=self._device,
            )
            # Soft-pull the freed verts toward their bone target so they track
            # gross body motion (per-frame target jumps) while still yielding
            # inward to the fascia squeeze. Skipped under belly-attach mode: the
            # belly is handled by the compliant XPBD attach in the iter loop
            # instead (a position blend here would double-pull + overwrite it).
            if self.fascia_soft_pin > 0.0 and not self.pin_belly_attach:
                wp.launch(
                    pintotarget_soft_kernel,
                    dim=self.NCONS,
                    inputs=[self.pts, self.target_pos, self.rest_offset,
                            self.pin_active, self.fascia_soft_pin, self.pos],
                    device=self._device,
                )
        else:
            wp.launch(
                pintotarget_overwrite_kernel,
                dim=self.NCONS,
                inputs=[self.pts, self.target_pos, self.rest_offset, self.pos],
                device=self._device,
            )

    # ─── Multigrid V-cycle: restrict → (coarse smooth, off) → prolong ────────

    def _restrict(self) -> None:
        """Fine → coarse transfer, writing ``coarse_pos``.

        Two modes:
          * belly-attach: coarse is a bone-driven scaffold. Scatter the dense
            fine bone target, Galerkin least-squares restrict it onto the cage
            (Cdef=(P^T P)^-1 P^T T — smoothest cage deformation reproducing the
            target), copy straight in. No coarse ARAP, so it can't drift and
            feed the unpinned belly a divergent start.
          * direct V-cycle: restrict the current fine positions onto coarse for
            the free ARAP relax that follows.
        """
        s = self
        if s.pin_belly_attach:
            wp.launch(
                scatter_target_kernel,
                dim=s.NCONS,
                inputs=[s.pts, s.target_pos, s.rest_offset, s.fine_target_dense],
                device=s._device,
            )
            if getattr(s, "minv_flat", None) is not None:
                # b = P^T T (atomic scatter), then per-muscle block solve
                # Cdef = Minv b. (Legacy lowres path used restrict_direct_kernel
                # on the tet-barycentric c2f map — removed, ~6x more strain.)
                s.coarse_b.zero_()
                wp.launch(
                    restrict_galerkin_scatter_kernel,
                    dim=s.mg_fine_nv,
                    inputs=[s.cage_idx, s.uvw, s.coarse_tet_idx,
                            s.fine_target_dense, s.coarse_b],
                    device=s._device,
                )
                wp.launch(
                    block_solve_kernel,
                    dim=s.coarse_NV,
                    inputs=[s.minv_flat, s.blk_base, s.blk_dim,
                            s.blk_matoff, s.coarse_b, s.coarse_target],
                    device=s._device,
                )
            wp.copy(s.coarse_pos, s.coarse_target)
        else:
            wp.launch(
                restrict_direct_kernel,
                dim=s.coarse_NV,
                inputs=[s.pos, s.tet_indices,
                        s.cage_idx_c2f, s.uvw_c2f, s.coarse_pos],
                device=s._device,
            )

    def _coarse_smooth(self) -> None:
        """Optional coarse cage relax — OFF by default (coarse_iterations=0) and
        currently a NET LOSS, kept for a future fiber-aware solve.

        Runs the SAME XPBD dispatch as the fine level via the nested
        ``_CoarseEngine`` (ARAP-SVD + optional volume; in belly-attach it also
        anchors each iter back to the Galerkin projection ``coarse_target``).

        Idea: relax incompressibility cheaply on the small cage, prolong the
        correction to the belly. Measured reality: relaxing drags the cage off
        the clean Galerkin projection toward its own (inflated) rest shape, so
        any coarse_iters>0 corrupts the prolonged start and blows the fine ARAP
        residual up ~100x. Do not enable without a fiber-direction-aware coarse
        energy. See docs/humanV2_cage_multigrid.md.
        """
        s = self
        if getattr(s, "_coarse", None) is None or s.coarse_iters <= 0:
            return
        if s.pin_belly_attach:
            # Anchor target is the Galerkin projection; meaningless without it.
            if getattr(s, "minv_flat", None) is None:
                return
            s._coarse.set_anchor(s.coarse_target)
        s._coarse.smooth(s.coarse_pos, s.coarse_iters)

    def _prolong(self) -> None:
        """Coarse → fine transfer: blend each fine vert from its containing
        coarse tet (barycentric prolong P), writing ``pos``."""
        wp.launch(
            prolong_kernel,
            dim=getattr(self, "mg_fine_nv", self.NV),
            inputs=[self.coarse_pos, self.coarse_tet_idx,
                    self.cage_idx, self.uvw, self.pos],
            device=self._device,
        )

    def build_program(self):
        s = self

        def pre_phase():
            if s.use_multigrid:
                # V-cycle: restrict fine→coarse, optionally relax the cage
                # (off by default), prolong coarse→fine, then re-pin kinematically.
                s._restrict()
                s._coarse_smooth()
                s._prolong()
                s.apply_pintotarget()
            elif s.pin_active is not None:
                # Partial pin without multigrid: still re-apply pin every substep
                # so the hard tendon anchors hold and freed belly verts get their
                # soft pull (the multigrid branch above does this via prolong).
                s.apply_pintotarget()
            # Fascia anchor + glue run in post_phase (s.fascia.substep), AFTER the
            # muscle solve, so the shell tracks the just-updated muscle and the
            # two-way glue squeezes the relaxed belly.
            # XPBD Lagrange / Jacobi accumulators are reset by s.clear() inside
            # iter_phase each fine iter — no per-substep HPBD L.zero_() needed.


        def iter_phase():
            # SoA XPBD sweep. Quasi-static integrate snapshots pprev=pos; each
            # fine iter clears the Jacobi accumulators + constraint Lagrange,
            # dispatches the built constraint table (ARAP-SVD + optional volume /
            # fiber / belly-attach — parallel-Jacobi over constraint types),
            # applies the averaged correction, then re-pins kinematically.
            s.integrate()
            for it in range(s.fine_iters):
                s.clear()
                s.solve_constraints(run_extras=False)
                if s.use_jacobi:
                    s.apply_dP()
                s.apply_pintotarget()
                # Per-iteration collision hook (UDCF DCD): interleaves contact
                # resolution with ARAP/volume the way the reference PBD loop
                # interleaves narrowPhase() with projectConstraints().
                for fn in getattr(s, "collision_solvers_iter", None) or []:
                    fn(it)

        def post_phase():
            # Fascia shell substep: anchor (follow muscle) -> relax (ARAP+volume)
            # interleaved with the two-way glue squeeze on the muscle belly.
            if s.fascia is not None:
                s.fascia.substep()
            # Optional collision pass (e.g. muscle vs rigid SDF). Runs AFTER ARAP /
            # glue each substep so it pushes against the relaxed geometry; pinned
            # verts are protected by the collider's own inv_mass=0 mask.
            for fn in getattr(s, "collision_solvers", None) or []:
                fn()

        self.program = [pre_phase, iter_phase, post_phase]

    def substep(self) -> None:
        if not hasattr(self, "program"):
            self.build_program()
        for phase in self.program:
            phase()

    def step_frame(self, frame: int) -> None:
        self.load_frame_targets(frame)
        self.apply_pintotarget()
        # Fascia rides the skeleton: advect the shell by this frame's bone motion
        # ONCE (target_pos is constant across the frame's substeps) before solving.
        if self.fascia is not None:
            self.fascia.advect_bone()
        for _ in range(self.nsubsteps):
            self.substep()

    def positions_np(self) -> np.ndarray:
        return self.pos.numpy()


def load_config(config_path: str) -> dict:
    """Load a JSON scene config and resolve relative paths against cwd."""
    cfg = json.load(open(config_path))
    for key in ("geo_dir", "cons_dir", "rest_path", "fascia_path"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(Path.cwd() / cfg[key])
    return cfg
