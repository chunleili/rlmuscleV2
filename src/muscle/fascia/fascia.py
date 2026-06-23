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
def fascia_advect_bone_kernel(
    follow_cons: wp.array(dtype=wp.int32),     # fascia vert -> nearest muscle cons
    target_pos: wp.array(dtype=wp.vec3),       # parent per-frame BONE target
    rest_offset: wp.array(dtype=wp.vec3),      # parent cons rest offset
    prev_world: wp.array(dtype=wp.vec3),       # last bone-world this vert saw
    fascia_pos: wp.array(dtype=wp.vec3),
):
    """Advect each fascia vert by the BONE motion field (not the muscle it
    squeezes). ``target_pos + rest_offset`` is the lag-free per-frame world
    position the skeleton drives the nearest muscle vert to; the DELTA since the
    last frame is added to the fascia vert, so the shell rides the skeleton while
    keeping its own (shrunk + glue-squeezed) shape. Per-frame, not per-substep."""
    i = wp.tid()
    cur = target_pos[follow_cons[i]] + rest_offset[follow_cons[i]]
    fascia_pos[i] = fascia_pos[i] + (cur - prev_world[i])
    prev_world[i] = cur


@wp.kernel
def fascia_anchor_bone_kernel(
    anchor_local: wp.array(dtype=wp.int32),    # fascia vert id (subset)
    anchor_cons: wp.array(dtype=wp.int32),     # its nearest muscle cons
    anchor_offset: wp.array(dtype=wp.vec3),    # fascia_rest - bone_rest_world
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    fascia_pos: wp.array(dtype=wp.vec3),
):
    """Hard-snap a sparse set of fascia verts to their BONE-driven world target
    each substep. These are the shell's Dirichlet BCs: without them the stiff
    ARAP/volume Jacobi relax on the otherwise free-floating shell diverges. Snap
    to the SKELETON (target_pos), never the squeezable muscle pos."""
    i = wp.tid()
    c = anchor_cons[i]
    fascia_pos[anchor_local[i]] = target_pos[c] + rest_offset[c] + anchor_offset[i]


@wp.kernel
def solve_glue_kernel(
    pair_fascia: wp.array(dtype=wp.int32),       # local fascia vert
    pair_muscle: wp.array(dtype=wp.int32),       # muscle vert (global)
    restlen: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),      # 1/stiffness
    muscle_mass: wp.array(dtype=wp.float32),
    muscle_stopped: wp.array(dtype=wp.int32),    # pinned muscle vert = kinematic
    L: wp.array(dtype=wp.float32),
    dt: float,
    omega: float,
    fascia_pos: wp.array(dtype=wp.vec3),
    muscle_pos: wp.array(dtype=wp.vec3),
):
    """ONE-WAY XPBD distance from a kinematic fascia vert to a muscle vert (each
    in its own ``pos`` array). The fascia is bone-driven (kinematic here); only
    the muscle moves — toward the shrunk fascia, i.e. the squeeze. Muscle verts
    that are ``stopped`` (hard-pinned) are kinematic too, so that pair is inert."""
    c = wp.tid()
    fi = pair_fascia[c]
    mi = pair_muscle[c]
    n = fascia_pos[fi] - muscle_pos[mi]
    d = wp.length(n)
    if d < 1.0e-9:
        return
    dirn = n / d
    C = d - restlen[c]
    # The fascia side is KINEMATIC (wi = 0): driven solely by its one-to-one bone
    # follow, never by the (many) muscle springs that map onto it. In sparse-shell
    # regions one fascia vert is the nearest for dozens of muscle verts — a
    # two-way pull there flung the weakly-connected (sliver-tet) vert ~1m. With
    # wi=0 the glue only drags the muscle toward the shrunk bone-driven fascia.
    wj = float(0.0)
    if muscle_stopped[mi] == 0:
        mm = muscle_mass[mi]
        wj = wp.where(mm > 0.0, 1.0 / mm, 0.0)
    alpha_tilde = compliance[c] / (dt * dt)
    denom = wj + alpha_tilde
    if denom < 1.0e-12:
        return
    dlambda = -(C + alpha_tilde * L[c]) / denom
    L[c] = L[c] + dlambda
    f = omega * dlambda
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
        # All per-constraint params resolved by the parent from cfg["constraints"]
        # (fascia_arap / fascia_volume / fascia_glue entries, bicep.json style).
        self._volume_stiffness = float(p.fascia_volume_stiffness)
        self._volume_damp = float(p.fascia_volume_damp)
        self._arap_damp = float(p.fascia_arap_damp)

        _check_unzipped(p.fascia_path)
        fg = GeoFile(p.fascia_path)
        self.pos0_np = fg.positions.astype(np.float32)
        self.tet_np = np.asarray(fg.tet_indices, dtype=np.int64)
        self.n_verts = int(self.pos0_np.shape[0])
        self.NV = self.n_verts
        self.NT = int(self.tet_np.shape[0])
        self.v_fiber_np = None
        self.v_tendonmask_np = None
        # Shell ARAP stiffness: config fascia_arap.stiffness when >= 0, else the
        # per-vertex geo attribute (solidshapestiffness) mean.
        self._fascia_stiff = (float(p.fascia_arap_stiffness)
                              if p.fascia_arap_stiffness >= 0.0
                              else float(fg.point_attr("solidshapestiffness").mean()))

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
            {'stiffness': self._fascia_stiff, 'dampingratio': self._arap_damp},
            ctype=ConstraintType.TETARAPSVD)
        cons += self.create_tet_volume_constraint(
            {'stiffness': self._volume_stiffness, 'dampingratio': self._volume_damp})
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

        glue_scale = float(p.fascia_glue_scale)
        glue_fascia, glue_muscle, glue_restlen, glue_stiff = self._pair_surface(
            fg, fascia_pos, mrest, rest_muscle_pos, glue_scale)

        # Bone follow: every fascia vert rides the SKELETON, not the muscle it
        # squeezes. The parent is fully pinned (pts == all NV), so target_pos is a
        # complete per-frame bone-driven field over the muscle verts (lag-free,
        # even where the belly is freed from the pin). Map each fascia vert to its
        # nearest muscle vert's cons index; advect_bone() then moves the shell by
        # that vert's per-frame bone DELTA. cons index == position in pts.
        pin_pts = p._pin_pts_np                       # cons -> muscle vert
        vert2cons = np.full(nvm, -1, dtype=np.int64)
        vert2cons[pin_pts] = np.arange(pin_pts.shape[0])
        _, floc = cKDTree(rest_muscle_pos).query(fascia_pos, k=1)
        follow_cons = vert2cons[floc]
        if (follow_cons < 0).any():                   # vert not pinned -> nearest pinned
            miss = follow_cons < 0
            _, ploc = cKDTree(rest_muscle_pos[pin_pts]).query(fascia_pos[miss], k=1)
            follow_cons[miss] = ploc

        log.info(f"fascia: {fascia_pos.shape[0]} verts, {self.NT} tets; "
                 f"surface glue pairs={len(glue_fascia)}; "
                 f"bone-follow all {fascia_pos.shape[0]} verts; rest_scale={self._rest_scale}")

        # Free the muscle verts the fascia glues to (+ dilated ring) from the pin
        # so the glue can pull them inward.
        # pin_active is left as the parent's tendon mask (tendonmask >
        # tendon_pin_threshold): tendons stay hard-pinned, the belly is freed and
        # driven by belly-attach + the glue squeeze. The glue only moves verts the
        # belly-attach leaves compliant, so no extra pin rebuild is needed here.

        # Upload coupling arrays.
        self.n_glue = int(len(glue_fascia))
        self.glue_fascia = wp.array(glue_fascia.astype(np.int32), dtype=wp.int32, device=device)
        self.glue_muscle = wp.array(glue_muscle.astype(np.int32), dtype=wp.int32, device=device)
        self.glue_restlen = wp.array(glue_restlen.astype(np.float32), dtype=wp.float32, device=device)
        self.glue_compliance = wp.array(
            (1.0 / np.maximum(glue_stiff.astype(np.float32), 1e-30)).astype(np.float32),
            dtype=wp.float32, device=device)
        self.glue_L = wp.zeros(max(self.n_glue, 1), dtype=wp.float32, device=device)

        # Bone-follow arrays. prev_world seeds with the frame-start bone world
        # (target_pos + rest_offset at each vert's cons); advect_bone() adds only
        # the per-frame DELTA, so the fascia's own rest offset is preserved and
        # never snapped to the muscle surface (would fight the shrink squeeze).
        self.follow_cons = wp.array(follow_cons.astype(np.int32), dtype=wp.int32, device=device)
        tgt0 = p.target_pos.numpy()
        roff = p.rest_offset.numpy()
        bone_rest = (tgt0[follow_cons] + roff[follow_cons]).astype(np.float32)  # per fascia vert
        self.follow_prev = wp.from_numpy(bone_rest.copy(), dtype=wp.vec3, device=device)

        # Dirichlet BC set for the relax: the fascia "rigid" group (or a uniform
        # subsample as fallback) hard-snapped to bone each substep. anchor_offset
        # holds fascia_rest - bone_rest so the snap preserves the shell's offset
        # from the skeleton, not collapse onto the muscle surface.
        if "rigid" in fg._groups:
            rid = fg.point_group("rigid").astype(np.int64)
        else:
            step = max(1, fascia_pos.shape[0] // 256)
            rid = np.arange(0, fascia_pos.shape[0], step, dtype=np.int64)
        self.n_anchor = int(rid.shape[0])
        self.anchor_local = wp.array(rid.astype(np.int32), dtype=wp.int32, device=device)
        self.anchor_cons = wp.array(follow_cons[rid].astype(np.int32), dtype=wp.int32, device=device)
        anchor_off = (fascia_pos[rid] - bone_rest[rid]).astype(np.float32)
        self.anchor_offset = wp.array(anchor_off, dtype=wp.vec3, device=device)
        log.info(f"fascia: {self.n_anchor} bone-anchor BCs "
                 f"({'rigid group' if 'rigid' in fg._groups else 'subsample'})")

    def _pair_surface(self, fg, fascia_pos, mrest, rest_muscle_pos, glue_scale):
        """Full-surface glue mode: pair each muscle surface vert to its nearest
        fascia vert (un-dilutes the global squeeze metric). Returns LOCAL fascia
        ids + GLOBAL muscle ids + restlen + scaled stiffness.

        Rest length IS the rest distance — the glue is a plain attachment that
        holds each fascia/muscle pair at the gap they start with. ALL squeeze
        comes from the fascia ARAP shrink (``fascia_rest_scale``): the shell
        contracts and drags the glued muscle in. The glue itself never pulls a
        pair toward an absolute target, so it can't catapult a weakly-connected
        fascia vert where the shell is sparse (the old ``min(want, d)`` slammed
        every pair toward ~2mm regardless of gap and flung such verts ~1m)."""
        p = self.parent
        surf = mrest.point_group("surface_points")
        fascia_tree = cKDTree(fascia_pos)
        d, floc = fascia_tree.query(rest_muscle_pos[surf], k=1)

        glue_muscle = surf.astype(np.int32)
        glue_fascia = floc.astype(np.int32)
        glue_restlen = d.astype(np.float32)

        # Absolute glue stiffness from config (fascia_glue.stiffness) when set,
        # else the per-vertex geo attribute (tissuetomusclestiffness) mean.
        base_stiff = (float(p.fascia_glue_stiffness)
                      if p.fascia_glue_stiffness is not None
                      else float(fg.point_attr("tissuetomusclestiffness").mean()))
        glue_stiff = np.full(len(surf), base_stiff * glue_scale, dtype=np.float32)
        return glue_fascia, glue_muscle, glue_restlen, glue_stiff

    # -- run --------------------------------------------------------------

    def advect_bone(self):
        """Ride the skeleton: add this frame's bone DELTA to every fascia vert.
        Called ONCE per frame by the parent (target_pos is constant across a
        frame's substeps), BEFORE the substep loop — lag-free gross motion."""
        wp.launch(fascia_advect_bone_kernel, dim=self.NV,
                  inputs=[self.follow_cons, self.parent.target_pos,
                          self.parent.rest_offset, self.follow_prev, self.pos],
                  device=self._device)

    def build_program(self):
        s = self

        def anchor_phase():
            # Gross bone motion is applied per-FRAME in advect_bone(); each substep
            # re-snaps the sparse bone-anchor BCs (stabilizes the stiff relax) then
            # snapshots pprev for the quasi-static relax below.
            if s.n_anchor:
                wp.launch(fascia_anchor_bone_kernel, dim=s.n_anchor,
                          inputs=[s.anchor_local, s.anchor_cons, s.anchor_offset,
                                  s.parent.target_pos, s.parent.rest_offset, s.pos],
                          device=s._device)
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
                                      s.glue_compliance,
                                      s.parent.mass, s.parent.stopped, s.glue_L,
                                      s.dt, s.omega, s.pos, s.parent.pos],
                              device=s._device)

        self.program = [anchor_phase, relax_phase]

    def substep(self):
        if not hasattr(self, "program"):
            self.build_program()
        for phase in self.program:
            phase()
