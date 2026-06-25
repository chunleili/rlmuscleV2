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
from muscle.util.geo_io import GeoFile, read_cons_frame, read_geo_attr_cached
from muscle.util.util import _check_unzipped


def find_constraint(cfg: dict, name: str) -> dict | None:
    """Look up a constraint config block by ``name`` in the cfg ``constraints``
    array (bicep.json style: every constraint type carries its own stiffness /
    dampingratio / extra params). Returns None when absent OR when the entry sets
    ``"enable": false`` — so a constraint can be toggled off in place without
    deleting its tuned params. ``enable`` defaults to true. For the optional
    constraints (muscle volume, fiber) absence/disabled == off; for the mandatory
    ones (arap, belly_attach) disabling raises at the call site."""
    for c in cfg.get("constraints", []):
        if c.get("name") == name:
            return c if c.get("enable", True) else None
    return None


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
        # Quasi-static muscle: no inertia. Engine-level flag (XpbdEngine.integrate
        # / update_velocities honor it); gravity vec feeds the same path (humanV2
        # runs gravity-free, so this is [0,0,0] but stays config-driven).
        self.quasi_static = bool(cfg.get("quasi_static", False))
        self._gravity_vec = wp.vec3(*[float(x) for x in cfg.get("gravity", [0.0, 0.0, 0.0])])
        # ── Per-constraint params (bicep.json style: cfg["constraints"]) ──────
        # Each constraint type owns its stiffness / dampingratio / extra params.
        # arap is mandatory (the muscle ARAP-SVD shape match, stiffness = the old
        # `mu`); muscle volume / fiber are optional (absence == disabled).
        _arap = find_constraint(cfg, "arap")
        if _arap is None:
            raise ValueError("constraints[] must define an 'arap' entry "
                             "(the muscle ARAP-SVD shape match)")
        self.mu = float(_arap["stiffness"])
        self._arap_damp = float(_arap.get("dampingratio", 0.0))
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
        # Fascia constraint params resolved from the constraints[] array (so the
        # standalone Fascia sub-solver reads them off the parent, not raw cfg):
        #   fascia_arap   -> shell ARAP-SVD shape (stiffness<0 == use geo attr),
        #                    plus its shrink rest_scale (the squeeze target);
        #   fascia_volume -> shell incompressibility;
        #   fascia_glue   -> muscle<->fascia glue (stiffness_scale, iterations).
        _f_arap = find_constraint(cfg, "fascia_arap") or {}
        _f_vol = find_constraint(cfg, "fascia_volume") or {}
        _f_glue = find_constraint(cfg, "fascia_glue") or {}
        self.fascia_rest_scale = float(_f_arap.get("rest_scale", 0.95))
        self.fascia_arap_stiffness = float(_f_arap.get("stiffness", -1.0))
        self.fascia_arap_damp = float(_f_arap.get("dampingratio", 0.0))
        self.fascia_volume_stiffness = float(_f_vol.get("stiffness", 1.0e10))
        self.fascia_volume_damp = float(_f_vol.get("dampingratio", 0.0))
        self.fascia_glue_scale = float(_f_glue.get("stiffness_scale", 0.001))
        self.fascia_glue_stiffness = _f_glue.get("stiffness", None)
        self.fascia_glue_damp = float(_f_glue.get("dampingratio", 0.0))
        self.fascia_glue_iters = int(_f_glue.get("iterations", self.fine_iters))
        # Fascia coupling mode (Phase-1 Han-style redesign):
        #   "squeeze" -> legacy: bone-anchored shrunk-ARAP shell + one-way glue
        #                drags the muscle belly inward (the active squeeze; this
        #                folds muscle tets -> inversions, super-additive with MG).
        #   "ride"    -> Han 2024: bind each fascia vert barycentrically to the
        #                closest MUSCLE-surface triangle and follow it each
        #                substep (no inward push -> ~zero muscle-tet inversions).
        #                Constitutive = SNH (volume-preserving), no rest_scale.
        _f_snh = find_constraint(cfg, "fascia_snh") or {}
        self.fascia_mode = str(cfg.get("fascia_mode", "squeeze"))
        self.fascia_snh_mu = float(_f_snh.get("mu", 1.0e3))
        self.fascia_snh_lam = float(_f_snh.get("lam", 1.0e8))
        self.fascia_snh_damp = float(_f_snh.get("dampingratio", 0.0))
        self.fascia_bind_max_dist = float(cfg.get("fascia_bind_max_dist", 0.05))
        self.dt = self.dt_frame / self.nsubsteps
        self.end_frame = int(cfg.get("end_frame", 90))
        self.start_frame = int(cfg.get("start_frame", 1))
        self._device = device

        self.use_multigrid = bool(cfg.get("use_multigrid", False))
        # Conforming per-muscle cage baked by tools/gen_coarse_cage.py: a true
        # enclosing cage (every fine vert inside a same-muscle coarse tet, prolong
        # weights in [0,1]). The ONLY multigrid coarse source — the legacy
        # surface-fit lowres geo (61.8% of fine verts extrapolated -> spikes) is gone.
        # Projection-only V-cycle: restrict the dense bone target onto the cage
        # (Galerkin lsq) and prolong it back as the fine solve's start. No coarse
        # ARAP relax — the cage is a pure bone-following scaffold.
        self.coarse_npz_path = cfg.get("coarse_npz_path", None)
        if self.coarse_npz_path is not None:
            self.coarse_npz_path = str(self.coarse_npz_path)
        # Opt-in convergence probe: when true, the global volume-weighted ARAP
        # residual RMS is appended to self.residual_log after each fine iter
        # (and once at iter 0 = the post-pre_phase initial guess). Used to
        # compare multigrid vs non-MG convergence; off by default (adds a sync).
        self.debug_residual = bool(cfg.get("debug_residual", False))
        self.residual_log = []
        self._resid_buf = None

        # Hydrostatic volume constraint (incompressibility). ARAP-SVD matches
        # shape only; the fine loop adds a det(F)=1 XPBD constraint. Enabled by
        # adding a 'volume' entry to constraints[] (absence == pure ARAP-HPBD).
        _vol = find_constraint(cfg, "volume")
        self.use_volume = _vol is not None
        self.volume_stiffness = float(_vol["stiffness"]) if _vol else 1.0e10
        self.volume_damp = float(_vol.get("dampingratio", 0.0)) if _vol else 0.0

        # Optional PASSIVE fiber constraint. The humanV2 asset (restpos.geo)
        # carries the per-vertex rest fiber direction in the ``materialW`` point
        # attribute, so the two passive Houdini fiber models are supported:
        #   "fibernorm"  -> solve_fibernorm_kernel  (linear TETFIBERNORM)
        #   "aniso_arap" -> solve_aniso_arap_kernel (quadratic HOBAK fiber)
        # The active "fibermillard" model still needs per-tet activation (from an
        # RL trajectory) that the asset lacks, so it remains gated off.
        _fib = find_constraint(cfg, "fiber")
        self.use_fiber = _fib is not None
        self.fiber_model = str((_fib or {}).get("fiber_model", "fibernorm"))
        self.fiber_stiffness = float((_fib or {}).get("stiffness", 1.0e8))
        self.fiber_damp = float((_fib or {}).get("dampingratio", 0.0))
        self.contraction_ratio = float((_fib or {}).get("contraction_ratio", 0.4))
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
            dict(type="tetarapsvd", name="arap", stiffness=self.mu,
                 dampingratio=self._arap_damp),
        ]
        if self.use_volume:
            self.constraint_configs.append(
                dict(type="volume", name="volume", stiffness=self.volume_stiffness,
                     dampingratio=self.volume_damp))
        if self.use_fiber:
            # fibernorm -> linear TETFIBERNORM; aniso_arap -> quadratic HOBAK
            # along-fiber ARAP (TETARAPANISO). Both read the materialW fiber field.
            _ftype = "aniso_arap" if self.fiber_model == "aniso_arap" else "fiber"
            self.constraint_configs.append(
                dict(type=_ftype, name="fiber", stiffness=self.fiber_stiffness,
                     dampingratio=self.fiber_damp))

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

        # Per-frame target source. Two modes:
        #   "points" — read the baked per-vertex traction target from
        #              cons_<frame>.geo (the target_pos attribute).
        #   "mesh"   — bind each muscle vert to the closest point on the bone
        #              SURFACE at the start frame, then each frame the target rides
        #              that surface through the bone_<frame>.geo sequence, so the
        #              muscle follows the actual skeleton mesh motion instead of
        #              pre-baked traction points. `use_sliding` (mesh-only) lets the
        #              binding re-project tangentially within slide_range each
        #              slide_interval frames; off = pure follow (no drift).
        # Either way the downstream machinery only sees `target_pos` (per muscle
        # vert) + a constant `rest_offset`; the belly-attach goal is their sum.
        self.target_mode = str(cfg.get("target_mode", "points"))
        self.use_sliding = bool(cfg.get("use_sliding", False))
        if self.use_sliding and self.target_mode != "mesh":
            raise ValueError("use_sliding=true requires target_mode='mesh' "
                             "(sliding has no meaning for the 'points' source)")
        self.bone_dir = cfg.get("bone_dir")
        self.slide_range = float(cfg.get("slide_range", 0.02))      # max tangential drift (m)
        self.slide_interval = int(cfg.get("slide_interval", 1))     # re-project every N frames
        if self.target_mode == "mesh":
            self._build_bone_slide_binding(rest, device)
        else:
            # Pin-to-target: topology (pts) is constant across all cons_<frame>.geo
            # files; only target_pos varies.
            cons0_path = str(Path(self.cons_dir) / f"cons_{self.start_frame}.geo")
            _check_unzipped(cons0_path)
            cons0 = GeoFile(cons0_path)
            self.NCONS = int(cons0.pts.shape[0])
            self.pts = wp.array(cons0.pts, dtype=wp.int32, device=device)
            self._pin_pts_np = np.asarray(cons0.pts, dtype=np.int64)
            rest_offset_np = (rest.positions[cons0.pts] - cons0.target_pos).astype(np.float32)
            self.rest_offset = wp.array(rest_offset_np, dtype=wp.vec3, device=device)
            self.target_pos = wp.array(cons0.target_pos.copy(), dtype=wp.vec3, device=device)
        self.pin_active = None  # rebuilt by Fascia (glue-ring free) when fascia is active

        # Belly-attach constraint config (the freed-belly pin system). It carries
        # both the compliant attach stiffness / dampingratio AND tendon_pin_threshold
        # (the hard-tendon vs freed-belly split is a property of this pin mode).
        _belly = find_constraint(cfg, "belly_attach")
        if _belly is None:
            raise ValueError("constraints[] must define a 'belly_attach' entry "
                             "(the freed-belly compliant pin)")
        self.pin_attach_stiffness = float(_belly["stiffness"])
        self.pin_attach_damp = float(_belly.get("dampingratio", 0.0))

        # Partial pinning (REQUIRED). Fully hard-pinning every vert (pts == all NV,
        # targets self-overlap) forces the belly through neighbouring muscles with
        # no collision give. Instead: verts whose tendonmask exceeds
        # tendon_pin_threshold (the tendon attachment regions) stay HARD-pinned; the
        # belly is freed and driven by the compliant belly-attach below. Fascia
        # rebuilds pin_active from its own glue ring later.
        self.tendon_pin_threshold = _belly.get("tendon_pin_threshold", None)
        if self.tendon_pin_threshold is None:
            raise ValueError("belly_attach constraint must define "
                             "tendon_pin_threshold (the hard tendon vs freed-belly "
                             "split)")
        tmask = rest.point_attr("tendonmask").astype(np.float32)
        active = (tmask[self._pin_pts_np]
                  > float(self.tendon_pin_threshold)).astype(np.int32)
        self.pin_active = wp.array(active, dtype=wp.int32, device=device)
        n_pin = int(active.sum())
        n_tot = len(active)
        print(f"[partial-pin] tendonmask>{self.tendon_pin_threshold}: "
              f"{n_pin}/{n_tot} hard pins ({n_pin / n_tot * 100:.1f}% pinned tendon, "
              f"{(n_tot - n_pin) / n_tot * 100:.1f}% belly-attach)")

        # Belly attach: the freed belly (pin_active==0) is tied to its bone target
        # by a compliant XPBD point constraint solved in the iteration loop
        # (attach_target_kernel). Force-based, so collision / fascia compete with
        # it (vs a position-blend pin they could not move). The fine attach is
        # built after fascia so pin_active is final.
        self.attach_L = None
        print(f"[belly-attach] XPBD point attach for belly verts "
              f"(stiffness={self.pin_attach_stiffness:g}, "
              f"dampingratio={self.pin_attach_damp:g})")

        if self.use_multigrid:
            # The conforming cage feeds the full-pin scaffold path (Galerkin
            # restrict the dense bone target onto the cage, no coarse ARAP) — it
            # hands the belly a smooth bone-following start before the fine attach.
            if self.coarse_npz_path is None:
                raise ValueError("use_multigrid=true requires coarse_npz_path")
            self._init_coarse_cage()

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
        k = float(self.pin_attach_stiffness)
        belly = np.where(active == 0)[0].astype(np.int64)
        self._belly_cons_idx = belly                 # cons-frame index per attach
        cons = list(self.raw_constraints)
        for i in belly:
            v = int(self._pin_pts_np[i])
            cons.append(dict(
                type=ConstraintType.ATTACH, pts=[v, -1, -1, -1], stiffness=k,
                compressionstiffness=-1.0, dampingratio=self.pin_attach_damp, tetid=-1,
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

    # ------------------------------------------------------------------

    def _init_coarse_cage(self) -> None:
        """Load the conforming per-muscle cage + baked transfer maps from the npz
        produced by tools/gen_coarse_cage.py (--method inflate, the default).
        Restrict is the Galerkin least-squares projection (P^T P)^-1 P^T; prolong
        is the f->c barycentric. Projection-only V-cycle: it hands the fine solve a
        smooth bone-following start with no measurable downside. No coarse ARAP
        relax — a free coarse smooth dragged the cage off the clean projection
        toward its rest shape and blew the fine residual up ~100x (removed; see
        docs/humanV2_cage_multigrid.md)."""
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
              f"Galerkin lsq restrict (projection-only)")

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

    def _bone_frame_path(self, frame: int) -> str:
        if self.bone_dir is None:
            raise ValueError("target_mode='mesh' requires 'bone_dir' in the config")
        return str(Path(self.bone_dir) / f"bone_{frame}.geo")

    def _bone_frame_positions(self, frame: int) -> np.ndarray:
        """Raw bone-surface vertex positions at `frame` (for the slide mesh)."""
        path = self._bone_frame_path(frame)
        _check_unzipped(path)
        return np.asarray(read_geo_attr_cached(path, "P"), dtype=np.float32)

    def _build_bone_slide_binding(self, rest, device) -> None:
        """Bind each muscle vert to the closest point on the bone SURFACE at the
        start frame, storing (face, barycentric) + signed normal gap. The binding
        rides bone_<frame>.geo each frame and re-projects ("slides") tangentially
        within slide_range. target_pos carries the full goal, so rest_offset=0."""
        bone0_path = self._bone_frame_path(self.start_frame)
        _check_unzipped(bone0_path)
        g0 = GeoFile(bone0_path)
        bpos = np.asarray(g0.positions, dtype=np.float32)
        bfaces = np.asarray(g0.tet_indices, dtype=np.int32).reshape(-1)   # tris stored in tet_indices
        self._bone_mesh = wp.Mesh(
            points=wp.array(bpos, dtype=wp.vec3, device=device),
            indices=wp.array(bfaces, dtype=wp.int32, device=device))
        self._slide_max_dist = float(self.slide_range + 1.0)             # generous query radius
        self.NCONS = int(self.NV)
        _ident = np.arange(self.NV, dtype=np.int32)
        self.pts = wp.array(_ident, dtype=wp.int32, device=device)
        self._pin_pts_np = _ident.astype(np.int64)
        rest_pos_wp = wp.array(np.asarray(rest.positions, dtype=np.float32),
                               dtype=wp.vec3, device=device)
        self._anchor_face = wp.zeros(self.NV, dtype=wp.int32, device=device)
        self._anchor_uv = wp.zeros(self.NV, dtype=wp.vec2, device=device)
        self._slide_gap = wp.zeros(self.NV, dtype=wp.float32, device=device)
        wp.launch(bone_bind_kernel, dim=self.NV,
                  inputs=[self._bone_mesh.id, rest_pos_wp, self._slide_max_dist,
                          self._anchor_face, self._anchor_uv, self._slide_gap],
                  device=device)
        # Current (mutable) binding starts at the anchor.
        self._bind_face = wp.clone(self._anchor_face)
        self._bind_uv = wp.clone(self._anchor_uv)
        self.rest_offset = wp.zeros(self.NV, dtype=wp.vec3, device=device)
        self.target_pos = wp.zeros(self.NV, dtype=wp.vec3, device=device)
        # Initialise target on the start-frame bone (follow only, no slide).
        wp.launch(bone_slide_target_kernel, dim=self.NV,
                  inputs=[self._bone_mesh.id, self.pos, self._anchor_face, self._anchor_uv,
                          self._bind_face, self._bind_uv, self._slide_gap,
                          self.slide_range, 0, self._slide_max_dist, self.target_pos],
                  device=device)
        print(f"[bone-slide] bound {self.NV} muscle verts to bone surface "
              f"({bpos.shape[0]} verts, {bfaces.shape[0]//3} tris); "
              f"slide_range={self.slide_range:g}m every {self.slide_interval} frame(s)")

    def _update_slide_target(self, frame: int) -> None:
        """Advance the bone mesh to `frame`, re-project (slide) the binding every
        slide_interval frames, and write the per-vert bone target."""
        bpos = self._bone_pos_stack[frame] if hasattr(self, "_bone_pos_stack") \
            and frame in getattr(self, "_bone_pos_stack", {}) else self._bone_frame_positions(frame)
        self._bone_mesh.points.assign(bpos)
        self._bone_mesh.refit()
        reproject = 1 if (self.use_sliding and
                          (frame - self.start_frame) % max(1, self.slide_interval) == 0) else 0
        wp.launch(bone_slide_target_kernel, dim=self.NV,
                  inputs=[self._bone_mesh.id, self.pos, self._anchor_face, self._anchor_uv,
                          self._bind_face, self._bind_uv, self._slide_gap,
                          self.slide_range, reproject, self._slide_max_dist, self.target_pos],
                  device=self._device)
        self._update_belly_attach_targets()

    def preload_targets(self, frames=None) -> None:
        """Parse every frame's target once and hold it in RAM.

        Per-frame ``.geo`` is large JSON; parsing it inside the step loop is the
        dominant cost. Preloading lifts all of it out of the loop (and, via the
        ``.npy`` cache, off disk after the first ever run), so stepping is pure
        GPU work — independent of any warm/cold cache state.
        """
        if frames is None:
            frames = range(self.start_frame, self.end_frame + 1)
        if self.target_mode == "mesh":
            self._bone_pos_stack = {f: self._bone_frame_positions(f) for f in frames}
        else:
            self._target_stack = {f: read_cons_frame(self.cons_dir, f) for f in frames}

    def load_frame_targets(self, frame: int) -> None:
        if self.target_mode == "mesh":
            self._update_slide_target(frame)
            return
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
        # Hard-overwrite the pinned tendon verts (pin_active==1) to their per-frame
        # bone target. Freed belly verts (pin_active==0) are skipped here and driven
        # by the compliant belly-attach constraint in the iter loop instead.
        wp.launch(
            pintotarget_overwrite_masked_kernel,
            dim=self.NCONS,
            inputs=[self.pts, self.target_pos, self.rest_offset,
                    self.pin_active, self.pos],
            device=self._device,
        )

    # ─── Multigrid V-cycle: restrict → prolong (projection-only) ─────────────

    def _restrict(self) -> None:
        """Fine → coarse transfer, writing ``coarse_pos``.

        Coarse is a bone-driven scaffold: scatter the dense fine bone target,
        Galerkin least-squares restrict it
        onto the cage (Cdef=(P^T P)^-1 P^T T — smoothest cage deformation
        reproducing the target), copy straight in. No coarse ARAP, so it can't
        drift and feed the unpinned belly a divergent start.
        """
        s = self
        wp.launch(
            scatter_target_kernel,
            dim=s.NCONS,
            inputs=[s.pts, s.target_pos, s.rest_offset, s.fine_target_dense],
            device=s._device,
        )
        # b = P^T T (atomic scatter), then per-muscle block solve Cdef = Minv b.
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
                # V-cycle: restrict fine→coarse (Galerkin projection of the bone
                # target onto the cage), prolong coarse→fine, then re-pin
                # kinematically. Projection-only — no coarse relax.
                s._restrict()
                s._prolong()
                s.apply_pintotarget()
            else:
                # No multigrid: re-apply the hard tendon pin every substep (the
                # MG branch above does this via prolong). Freed belly verts are
                # driven by the compliant belly-attach in iter_phase.
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
            # End-of-substep velocity rebuild: v=(pos-pprev)/dt over the whole
            # substep's net displacement. Engine no-ops this when quasi_static
            # (vel held at 0 -> no inertia); when quasi_static is off the carried
            # velocity feeds next substep's integrate as momentum.
            s.update_velocities()

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
