"""Fascia shell as a standalone MuscleSim instance.

The fascia is its OWN ``MuscleSim`` sub-solver (ARAP-SVD + optional volume) —
it is NOT merged/vstacked into the muscle constraint table. It couples to the
parent muscle sim by two cross-array passes solved against the muscle ``pos``
each substep:

  * rigid **anchor** — fascia "rigid" verts snap to their nearest muscle vert
    (+ rest offset), so the shell rides the gross body motion;
  * two-way **glue** — XPBD distance springs between fascia surface verts and
    their nearest muscle surface verts. With ``fascia_rest_scale < 1`` the shell
    wants to contract (its ARAP rest is shrunk), so the glue squeezes the muscle
    belly inward — the whole point of the fascia layer.

``Fascia.substep()`` runs the shell's own phase program (anchor -> integrate ->
relax+glue). The parent calls it once per substep in its ``post_phase``.
"""
from __future__ import annotations

import logging

import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from muscle.musclesim import MuscleSim
from muscle.defs import ConstraintType
from muscle.util.util import _check_unzipped
from muscle.util.geo_io import GeoFile


# ─── Cross-array coupling kernels (fascia <-> muscle) ───────────────────────

@wp.kernel
def fascia_anchor_kernel(
    anchor_fascia: wp.array(dtype=wp.int32),
    anchor_muscle: wp.array(dtype=wp.int32),
    rest_offset: wp.array(dtype=wp.vec3),
    muscle_pos: wp.array(dtype=wp.vec3),
    fascia_pos: wp.array(dtype=wp.vec3),
):
    """``fascia_pos[af[i]] = muscle_pos[am[i]] + rest_offset[i]`` (cross-array)."""
    i = wp.tid()
    fascia_pos[anchor_fascia[i]] = muscle_pos[anchor_muscle[i]] + rest_offset[i]


@wp.kernel
def solve_glue_kernel(
    pair_fascia: wp.array(dtype=wp.int32),       # local fascia vert
    pair_muscle: wp.array(dtype=wp.int32),       # muscle vert (global)
    restlen: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),      # 1/stiffness
    fascia_invmass: wp.array(dtype=wp.float32),
    muscle_mass: wp.array(dtype=wp.float32),
    muscle_stopped: wp.array(dtype=wp.int32),    # pinned muscle vert = kinematic
    L: wp.array(dtype=wp.float32),
    dt: float,
    omega: float,
    fascia_pos: wp.array(dtype=wp.vec3),
    muscle_pos: wp.array(dtype=wp.vec3),
):
    """Two-way XPBD distance between a fascia vert and a muscle vert, each in its
    own ``pos`` array. Muscle verts that are ``stopped`` (hard-pinned) act as
    kinematic anchors (inv_mass = 0) so only the fascia side moves there."""
    c = wp.tid()
    fi = pair_fascia[c]
    mi = pair_muscle[c]
    n = fascia_pos[fi] - muscle_pos[mi]
    d = wp.length(n)
    if d < 1.0e-9:
        return
    dirn = n / d
    C = d - restlen[c]
    wi = fascia_invmass[fi]
    wj = float(0.0)
    if muscle_stopped[mi] == 0:
        mm = muscle_mass[mi]
        wj = wp.where(mm > 0.0, 1.0 / mm, 0.0)
    alpha_tilde = compliance[c] / (dt * dt)
    denom = wi + wj + alpha_tilde
    if denom < 1.0e-12:
        return
    dlambda = -(C + alpha_tilde * L[c]) / denom
    L[c] = L[c] + dlambda
    f = omega * dlambda
    wp.atomic_add(fascia_pos, fi, (f * wi) * dirn)
    wp.atomic_add(muscle_pos, mi, (-f * wj) * dirn)


class Fascia(MuscleSim):
    """Fascia shell sub-solver (ARAP-SVD + optional volume), coupled to the
    parent muscle sim by anchor + two-way glue. Built like ``_CoarseEngine``:
    manual field setup (no ``MuscleSim.__init__`` mesh/bone loading)."""

    def __init__(self, parent):
        self.parent = parent
        p = parent
        cfg = p.cfg
        self.cfg = {}
        self._device = p._device
        self.dt = p.dt
        self.omega = p.omega
        self.total_mass = float(cfg.get("fascia_total_mass", p.total_mass))
        self.glue_iters = int(p.fascia_glue_iters)
        self._rest_scale = float(p.fascia_rest_scale)
        self.use_jacobi = bool(getattr(p, "use_jacobi", True))
        self.use_colored_gs = False
        self.has_compressstiffness = False
        self.fiber_stiffness_scale = 1.0
        self.millard_stiffness_scale = 1.0
        # Volume (incompressibility) is always part of the fascia shell — the
        # shrink rest only sets the target SHAPE; volume keeps it from collapsing.
        self._volume_stiffness = float(cfg.get("fascia_volume_stiffness",
                                               cfg.get("volume_stiffness", 1.0e10)))

        _check_unzipped(p.fascia_path)
        fg = GeoFile(p.fascia_path)
        self.pos0_np = fg.positions.astype(np.float32)
        self.tet_np = np.asarray(fg.tet_indices, dtype=np.int64)
        self.n_verts = int(self.pos0_np.shape[0])
        self.NV = self.n_verts
        self.NT = int(self.tet_np.shape[0])
        self.v_fiber_np = None
        self.v_tendonmask_np = None
        # Shell ARAP stiffness from the fascia geo attribute.
        self._fascia_stiff = float(fg.point_attr("solidshapestiffness").mean())

        # SoA pipeline: ARAP-SVD (+ optional volume), then shrink the ARAP rest.
        self._allocate_fields()
        self._build_fascia_constraints()
        self._init_fields()
        self._precompute_rest()
        self._apply_rest_shrink()
        self.pprev = wp.clone(self.pos)

        # Coupling (glue / anchor) + free the glued muscle verts from the pin.
        self._build_coupling(fg)
        self.build_program()

    # -- MuscleSim seams --------------------------------------------------

    def _rest_density(self):
        _, vols, valid = self._batch_compute_tet_rest_matrices()
        return self.total_mass / float(max(vols[valid].sum(), 1e-30))

    def integrate(self):
        """Quasi-static: snapshot start positions, no gravity/inertia."""
        wp.copy(self.pprev, self.pos)

    # -- build ------------------------------------------------------------

    def _build_fascia_constraints(self):
        # ARAP-SVD (shape) + volume (incompressibility) — both always present.
        cons = self.create_tet_arap_constraints(
            {'stiffness': self._fascia_stiff}, ctype=ConstraintType.TETARAPSVD)
        cons += self.create_tet_volume_constraint(
            {'stiffness': self._volume_stiffness})
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

    def _apply_rest_shrink(self):
        """ARAP rest_matrix *= 1/scale  (== inv(scale*Dm)). At rest F = (1/scale)*I
        so sigma = 1/scale > 1 and ARAP drives the shell to the shrunk shape,
        contracting it inward. Mass keeps the UNSCALED rest volume (rest-shape
        only). Uniform scale across all fascia tets."""
        s = self._rest_scale
        if abs(s - 1.0) < 1.0e-9:
            return
        rm = self.rest_matrix.numpy() * (1.0 / s)
        self.rest_matrix = wp.from_numpy(rm.astype(np.float32), dtype=wp.mat33)

    def _build_coupling(self, fg):
        """Pair fascia<->muscle (glue + anchor), free the glued muscle verts (+ring)
        from the parent pin. Mirrors the old merged pairing but emits LOCAL fascia
        ids + GLOBAL muscle ids for the cross-array kernels."""
        p = self.parent
        cfg = p.cfg
        log = logging.getLogger(__name__)
        device = self._device

        mrest = GeoFile(p.rest_path)
        rest_muscle_pos = mrest.positions
        fascia_pos = self.pos0_np
        nvm = rest_muscle_pos.shape[0]

        glue_scale = float(cfg.get("fascia_glue_stiffness_scale", 0.001))
        glue_mode = str(cfg.get("fascia_glue_mode", "groups"))
        if glue_mode == "surface":
            glue_fascia, glue_muscle, glue_restlen, glue_stiff = self._pair_surface(
                fg, fascia_pos, mrest, rest_muscle_pos, glue_scale)
        else:
            surf = mrest.point_group("surface_points")
            surf_tree = cKDTree(rest_muscle_pos[surf])

            def pair(group_name, dist_attr, stiff_attr):
                if group_name not in fg._groups:
                    return (np.zeros(0, np.int32), np.zeros(0, np.int32),
                            np.zeros(0, np.float32), np.zeros(0, np.float32))
                fids = fg.point_group(group_name)
                d, loc = surf_tree.query(fascia_pos[fids], k=1)
                mids = surf[loc].astype(np.int32)
                want = fg.point_attr(dist_attr)[fids].astype(np.float32)
                restlen = np.minimum(want, d.astype(np.float32))
                stiff = fg.point_attr(stiff_attr)[fids].astype(np.float32)
                return fids.astype(np.int32), mids, restlen, stiff

            gi_f, gi_m, gi_r, gi_s = pair("tissue_inner_points",
                                          "tissuetomuscledistance", "tissuetomusclestiffness")
            gb_f, gb_m, gb_r, gb_s = pair("tissue_bone_points",
                                          "tissuetobonedistance", "tissuetobonestiffness")
            glue_fascia = np.concatenate([gi_f, gb_f])
            glue_muscle = np.concatenate([gi_m, gb_m])
            glue_restlen = np.concatenate([gi_r, gb_r])
            glue_stiff = np.concatenate([gi_s, gb_s]) * glue_scale

        # Rigid anchor: fascia "rigid" group follows its nearest muscle vert.
        if "rigid" in fg._groups:
            rid = fg.point_group("rigid")
            full_tree = cKDTree(rest_muscle_pos)
            _, aloc = full_tree.query(fascia_pos[rid], k=1)
            anchor_fascia = rid.astype(np.int32)
            anchor_muscle = aloc.astype(np.int32)
        else:
            anchor_fascia = np.zeros(0, np.int32)
            anchor_muscle = np.zeros(0, np.int32)

        log.info(f"fascia: {fascia_pos.shape[0]} verts, {self.NT} tets; "
                 f"glue mode={glue_mode}, glue pairs={len(glue_fascia)}; "
                 f"anchors={len(anchor_fascia)}; rest_scale={self._rest_scale}")

        # Free the muscle verts the fascia glues to (+ dilated ring) from the pin
        # so the glue can pull them inward.
        free_ring = int(cfg.get("fascia_free_ring", 1))
        muscle_tet = mrest.tet_indices
        free_mask = np.zeros(nvm, dtype=bool)
        seed = np.unique(glue_muscle.astype(np.int64))
        free_mask[seed[(seed >= 0) & (seed < nvm)]] = True
        if free_ring > 0:
            from collections import defaultdict
            v2t = defaultdict(list)
            for ti, tet in enumerate(muscle_tet):
                for v in tet:
                    v2t[int(v)].append(ti)
            for _ in range(free_ring):
                cur = np.where(free_mask)[0]
                for v in cur:
                    for ti in v2t[int(v)]:
                        for w in muscle_tet[ti]:
                            free_mask[int(w)] = True
        free_ids = np.where(free_mask)[0].astype(np.int32)

        # Rebuild the parent pin_active: hard-pin tendons (tendonmask>thr), free
        # the glue ring (union). Without this the parent re-pins the whole belly
        # every substep and the glue squeeze is overwritten.
        active = np.ones(p._pin_pts_np.shape[0], dtype=np.int32)
        if p.tendon_pin_threshold is not None and p.tendonmask_np is not None:
            tm = p.tendonmask_np
            active = (tm[p._pin_pts_np] > float(p.tendon_pin_threshold)).astype(np.int32)
            print(f"[partial-pin+fascia] tendonmask>{p.tendon_pin_threshold}: "
                  f"{int(active.sum())}/{len(active)} hard pins before glue-ring free")
        active[np.isin(p._pin_pts_np, free_ids)] = 0
        p.pin_active = wp.array(active, dtype=wp.int32, device=device)
        log.info(f"fascia: freed {len(free_ids)} muscle verts from pin "
                 f"(ring={free_ring}); {int(active.sum())}/{len(active)} pins active; "
                 f"glue_scale={glue_scale}")

        # Upload coupling arrays.
        self.n_glue = int(len(glue_fascia))
        self.glue_fascia = wp.array(glue_fascia.astype(np.int32), dtype=wp.int32, device=device)
        self.glue_muscle = wp.array(glue_muscle.astype(np.int32), dtype=wp.int32, device=device)
        self.glue_restlen = wp.array(glue_restlen.astype(np.float32), dtype=wp.float32, device=device)
        self.glue_compliance = wp.array(
            (1.0 / np.maximum(glue_stiff.astype(np.float32), 1e-30)).astype(np.float32),
            dtype=wp.float32, device=device)
        self.glue_L = wp.zeros(max(self.n_glue, 1), dtype=wp.float32, device=device)

        mass_np = self.mass.numpy()
        finv = np.where(mass_np > 0.0, 1.0 / np.maximum(mass_np, 1e-30), 0.0).astype(np.float32)
        self.fascia_invmass = wp.from_numpy(finv, dtype=wp.float32)

        self.n_anchor = int(len(anchor_fascia))
        self.anchor_fascia = wp.array(anchor_fascia, dtype=wp.int32, device=device)
        self.anchor_muscle = wp.array(anchor_muscle, dtype=wp.int32, device=device)
        offset_np = ((fascia_pos[anchor_fascia] - rest_muscle_pos[anchor_muscle]).astype(np.float32)
                     if self.n_anchor else np.zeros((0, 3), np.float32))
        self.anchor_offset = wp.array(offset_np, dtype=wp.vec3, device=device)

    def _pair_surface(self, fg, fascia_pos, mrest, rest_muscle_pos, glue_scale):
        """Full-surface glue mode: pair EVERY muscle surface vert to its nearest
        fascia vert (un-dilutes the global squeeze metric). Returns LOCAL fascia
        ids + GLOBAL muscle ids + restlen + scaled stiffness."""
        cfg = self.parent.cfg
        surf = mrest.point_group("surface_points")
        fascia_tree = cKDTree(fascia_pos)
        d, floc = fascia_tree.query(rest_muscle_pos[surf], k=1)

        want = float(cfg.get("fascia_glue_target_distance", 0.01))
        glue_muscle = surf.astype(np.int32)
        glue_fascia = floc.astype(np.int32)
        glue_restlen = np.minimum(want, d.astype(np.float32)).astype(np.float32)

        base_stiff = float(cfg.get(
            "fascia_glue_stiffness",
            float(fg.point_attr("tissuetomusclestiffness").mean())))
        glue_stiff = np.full(len(surf), base_stiff * glue_scale, dtype=np.float32)
        return glue_fascia, glue_muscle, glue_restlen, glue_stiff

    # -- run --------------------------------------------------------------

    def build_program(self):
        s = self

        def anchor_phase():
            if s.n_anchor:
                wp.launch(fascia_anchor_kernel, dim=s.n_anchor,
                          inputs=[s.anchor_fascia, s.anchor_muscle, s.anchor_offset,
                                  s.parent.pos, s.pos], device=s._device)
            s.integrate()

        def relax_phase():
            # Interleave shell ARAP/volume relax with the two-way glue so the
            # squeeze and the elastic response converge together.
            for _ in range(s.glue_iters):
                s.clear()
                s.solve_constraints(run_extras=False)
                if s.use_jacobi:
                    s.apply_dP()
                if s.n_glue:
                    s.glue_L.zero_()
                    wp.launch(solve_glue_kernel, dim=s.n_glue,
                              inputs=[s.glue_fascia, s.glue_muscle, s.glue_restlen,
                                      s.glue_compliance, s.fascia_invmass,
                                      s.parent.mass, s.parent.stopped, s.glue_L,
                                      s.dt, s.omega, s.pos, s.parent.pos],
                              device=s._device)

        self.program = [anchor_phase, relax_phase]

    def substep(self):
        if not hasattr(self, "program"):
            self.build_program()
        for phase in self.program:
            phase()
