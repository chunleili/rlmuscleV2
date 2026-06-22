"""humanV2 muscle demo — single json-driven entry point.

Baseline (``data/humanV2/config/humanV2.json``):
    fascia shell + animated-bone SDF collision, multigrid OFF (No MG).

This file MERGES the four old humanV2 examples (sdf bone, sdf+fascia, sdf+DCD
self, sdf+inter-muscle self). Every knob is read from the json config — the
ONLY command-line argument is ``--config``::

    PYTHONPATH=src python -m example.humanV2 --config data/humanV2/config/humanV2.json
    # -> output/humanV2.usd

The skeleton is driven by the same 200-frame animation that generated the
muscle's ``cons`` targets (``data/humanV2/model/bone/bone_<N>.geo``, constant
topology). Each frame the bone verts are loaded into a ``wp.Mesh`` and
``refit()``-ed before the solve, so penetration stays shallow / local.

Config schema (``demo`` object; solver keys stay at the json top level and are
consumed by ``OneWaySim``):

    "demo": {
      "out": "output/humanV2.usd",
      "bone_dir": "data/humanV2/model/bone",
      "device": "cuda:0",
      "no_export": false,
      "geo_out": null,
      "no_collide_primvars": true,
      "activate": null,                      # muscle-name substring; needs use_fiber
      "collision": {
        "enabled": true,
        "static_bone": false,
        "max_dist": 0.2,
        "contact_offset": 1e-5,
        "collide_eps": 1e-6
      },
      "self_collision": {                     # method: none | dcd | intermuscle
        "method": "none"
      }
    }
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import warp as wp

from muscle.mg.mg import MultiGridSim, load_config
from muscle.util.mesh_io import vectorized_surface_tris
from muscle.collision.sdf.kernel import solve_mesh_contact_pos_kernel
from muscle.util.geo_io import GeoFile, GeoWriter, read_geo_attr_cached


@wp.kernel
def _count_inside_kernel(pos: wp.array(dtype=wp.vec3), vmask: wp.array(dtype=wp.float32),
                         mesh: wp.uint64, max_dist: wp.float32,
                         count: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if vmask[i] == 0.0:                           # excluded (fascia) — not a meter vert
        return
    q = wp.mesh_query_point_sign_normal(mesh, pos[i], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        if wp.length(pos[i] - cp) * q.sign < 0.0:
            wp.atomic_add(count, 0, 1)


@wp.kernel
def _mark_inside_kernel(pos: wp.array(dtype=wp.vec3), vmask: wp.array(dtype=wp.float32),
                        mesh: wp.uint64, max_dist: wp.float32,
                        depth: wp.array(dtype=wp.float32)):
    """Per-vertex muscle-bone penetration depth (>0 inside the bone, 0 outside).

    ``vmask`` excludes verts that must not contribute to the meter (fascia): the
    outer fascia shell never sits inside bone, so any "inside" it reports is a
    ``mesh_query_point_sign_normal`` sign-flip false positive. Excluding it here
    keeps the muscle-bone count honest — same set the SDF resolver acts on."""
    i = wp.tid()
    depth[i] = 0.0
    if vmask[i] == 0.0:
        return
    q = wp.mesh_query_point_sign_normal(mesh, pos[i], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        d = wp.length(pos[i] - cp) * q.sign      # < 0 inside
        if d < 0.0:
            depth[i] = -d                         # positive penetration depth


def _set_is_collide_primvar(stage, prim_path, flag01, time,
                            name="is_collide_muscle_bone"):
    """Write a per-vertex int primvar (0/1) on a mesh prim.

    Lets usdview select / colour the penetrating muscle verts directly, even
    when they are hidden inside the bone mesh. ``flag01`` is one int per point
    (vertex interpolation); time-sampled to match the mesh's animation.
    ``name`` separates the kinds of penetration — ``is_collide_muscle_bone`` is
    muscle-vs-bone, ``is_collide_muscle_muscle`` is muscle-vs-muscle (self)
    penetration.
    """
    from pxr import UsdGeom, Sdf, Vt
    prim = stage.GetPrimAtPath(prim_path)
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
    if not pv:
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            name, Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.IntArray.FromNumpy(flag01.astype(np.int32)), time)


def _set_pen_depth_primvar(stage, prim_path, depth, time,
                           name="pen_depth_muscle_bone"):
    """Write a per-vertex float primvar (metres) on a mesh prim. ``name`` is
    ``pen_depth_muscle_bone`` or ``pen_depth_muscle_muscle``."""
    from pxr import UsdGeom, Sdf, Vt
    prim = stage.GetPrimAtPath(prim_path)
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
    if not pv:
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            name, Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.FloatArray.FromNumpy(depth.astype(np.float32)), time)


def _set_sum_is_collide(stage, prim_path, total, time,
                        name="sum_is_collide_muscle_bone"):
    """Write a per-frame scalar custom int attr on a prim. ``name`` is
    ``sum_is_collide_muscle_bone`` or ``sum_is_collide_muscle_muscle``."""
    from pxr import Sdf
    prim = stage.GetPrimAtPath(prim_path)
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Int)
    attr.Set(int(total), time)


class CollisionMeter:
    """Per-frame penetration metering instrument. Wraps the bone collider
    (``inside_depth``) and an optional muscle-vs-muscle ``mm_meter`` callable.
    ``measure()`` thresholds both at ``collide_eps`` and stashes the result on
    ``self`` (length-``NV`` per-vertex depth + 0/1 flag arrays); ``write_primvars``
    / ``summary`` then consume that last measurement. The arrays exist only to
    feed three sinks — USD primvars, the .geo writer, the progress print — so the
    write/format logic lives here next to the measurement.

    Threshold at ``collide_eps``, not >0: after SDF projection verts sit ON the
    surface (sub-micron inside per the signed test) — resolved, not a collision.
    eps filters that touch-noise out of the count. The self meter spans muscle
    verts only (length nvm); it is padded to ``NV`` so its slicing matches the
    bone depth array. Bone and self are SEPARATE meters — never folded together."""

    def __init__(self, collider, mm_meter, collide_eps, NV):
        self.collider = collider
        self.mm_meter = mm_meter
        self.collide_eps = collide_eps
        self.NV = NV
        self.bone_depth = self.bone_flag = None   # metres>0 inside / int 0/1
        self.self_depth = self.self_flag = None   # or None when no self meter

    def measure(self) -> "CollisionMeter":
        self.bone_depth = self.collider.inside_depth()         # >0 = inside bone
        self.bone_flag = (self.bone_depth > self.collide_eps).astype(np.int32)
        self.self_depth = self.self_flag = None
        if self.mm_meter is not None:
            dmm = self.mm_meter()
            self.self_depth = np.zeros(self.NV, dtype=np.float32)
            self.self_depth[:dmm.shape[0]] = dmm
            self.self_flag = (self.self_depth > self.collide_eps).astype(np.int32)
        return self

    def write_primvars(self, stage, prim_path, used, time, *, per_vertex):
        """Write the muscle prim's collision primvars for the last ``measure()``.
        ``per_vertex`` writes the full 0/1 + depth arrays (sliced to ``used``);
        the per-frame sums are ALWAYS written (cheap, and the headline metric)."""
        if per_vertex:
            _set_is_collide_primvar(stage, prim_path, self.bone_flag[used], time,
                                    name="is_collide_muscle_bone")
            _set_pen_depth_primvar(stage, prim_path, self.bone_depth[used], time,
                                   name="pen_depth_muscle_bone")
            if self.self_flag is not None:
                _set_is_collide_primvar(stage, prim_path, self.self_flag[used], time,
                                        name="is_collide_muscle_muscle")
                _set_pen_depth_primvar(stage, prim_path, self.self_depth[used], time,
                                       name="pen_depth_muscle_muscle")
        _set_sum_is_collide(stage, prim_path, int(self.bone_flag.sum()), time,
                            name="sum_is_collide_muscle_bone")
        if self.self_flag is not None:
            _set_sum_is_collide(stage, prim_path, int(self.self_flag.sum()), time,
                                name="sum_is_collide_muscle_muscle")

    def summary(self) -> str:
        """One-line progress string: bone count(max depth) + optional self."""
        mm = (f"mm={int(self.self_flag.sum())}({self.self_depth.max() * 1000:.1f}mm)"
              if self.self_flag is not None else "")
        return f"coll={int(self.bone_flag.sum())}({self.bone_depth.max() * 1000:.1f}mm) {mm}"


def _bake_static_primvars(stage, prim_path, sim, used):
    """Write the time-INVARIANT per-vertex muscle attributes (materialW fiber
    direction, tendonmask, optional Cd, muscle_id name) as default-timecode USD
    primvars.

    These never change across the sequence, so they are written once at the
    default timecode (not time-sampled). ``used`` is the compacted vertex index
    set actually rendered by the muscle prim (the verts referenced by its surface
    tris); every primvar is sliced to it so it matches the prim's point count."""
    from pxr import UsdGeom, Sdf, Vt, Usd
    prim = stage.GetPrimAtPath(prim_path)
    api = UsdGeom.PrimvarsAPI(prim)
    default = Usd.TimeCode.Default()
    nv = int(sim.NV)

    def _full(arr):
        """Pad a per-muscle-vertex array up to the full NV vertex set."""
        if arr.shape[0] == nv:
            return arr
        out = np.zeros((nv,) + arr.shape[1:], dtype=np.float32)
        out[:arr.shape[0]] = arr
        return out

    def _vec3(name, arr):
        pv = api.GetPrimvar(name) or api.CreatePrimvar(
            name, Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(
            np.ascontiguousarray(_full(arr)[used], np.float32)), default)

    def _scalar(name, arr):
        pv = api.GetPrimvar(name) or api.CreatePrimvar(
            name, Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.FloatArray.FromNumpy(
            np.ascontiguousarray(_full(arr)[used], np.float32)), default)

    baked = []
    if getattr(sim, "materialW_np", None) is not None:
        _vec3("materialW", sim.materialW_np)
        baked.append("materialW")
    if getattr(sim, "tendonmask_np", None) is not None:
        _scalar("tendonmask", sim.tendonmask_np)
        baked.append("tendonmask")
    if getattr(sim, "cd_np", None) is not None:
        cd = sim.cd_np
        if cd.ndim == 2 and cd.shape[1] >= 3:
            pv = api.GetPrimvar("displayColor") or api.CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec3fArray.FromNumpy(
                np.ascontiguousarray(_full(cd[:, :3])[used], np.float32)), default)
            baked.append("Cd")
    # muscle_id is a PER-TET attribute (index into muscle_id_strings); scatter it
    # to the verts of each muscle tet, then resolve to the muscle NAME. Muscles
    # are vertex-disjoint here, so every muscle vert gets a single unambiguous
    # name; fascia / unreferenced verts get "none". Written once as a static
    # per-vertex token primvar so usdview shows the human-readable muscle name
    # directly (tokens are interned -> ~199 unique strings + per-vertex indices).
    if getattr(sim, "muscle_id_np", None) is not None:
        mid = np.asarray(sim.muscle_id_np).astype(np.int32)
        tets = sim.tet_idx.numpy()
        nt_m = mid.shape[0]                       # muscle tets are the first nt_m
        vid = np.full(nv, -1, dtype=np.int32)
        for k in range(4):
            vid[tets[:nt_m, k]] = mid
        strings = list(getattr(sim, "muscle_id_strings", []) or [])
        lut = strings + ["none"]                  # index -1 -> last entry "none"
        idx = vid[used].copy()
        idx[idx < 0] = len(strings)
        names = [lut[i] for i in idx]
        pv = api.GetPrimvar("muscle_id") or api.CreatePrimvar(
            "muscle_id", Sdf.ValueTypeNames.TokenArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.TokenArray(names), default)
        baked.append("muscle_id")
    if baked:
        print(f"[usd] baked static primvars: {', '.join(baked)}")


def _load_bone_verts(bone_dir: Path, frame: int) -> np.ndarray:
    pos = read_geo_attr_cached(str(bone_dir / f"bone_{frame}.geo"), "P")
    return np.ascontiguousarray(np.asarray(pos, dtype=np.float32))


class AnimatedBoneMeshCollider:
    """Per-frame SNU-skeleton mesh as a moving rigid collider for the muscle.

    Topology (triangles) is constant across the sequence, so the ``wp.Mesh`` is
    built once and only its ``points`` are refreshed + ``refit()`` each frame.
    """

    def __init__(self, sim: MultiGridSim, bone_dir: str,
                 max_dist: float = 0.2, contact_offset: float = 0.0,
                 max_push: float = -1.0):     # <=0 = no deep-contact reject
        self.sim = sim
        self.device = sim._device
        self.bone_dir = Path(bone_dir)
        self.max_dist = float(max_dist)
        self.contact_offset = float(contact_offset)
        self.max_push = float(max_push)

        g0 = GeoFile(str(self.bone_dir / "bone_1.geo"))
        tri = np.asarray(g0.tet_indices, dtype=np.int32)
        # SNU bones are wound inward relative to Warp's signed query — without
        # flipping, mesh_query_point_sign_normal reports ~155640/157369 verts
        # "inside" at rest vs the true ~1729. Swap each tri's last two indices.
        tri = np.ascontiguousarray(tri[:, [0, 2, 1]]).reshape(-1)
        self.n_bone_verts = int(np.asarray(g0.positions).shape[0])
        self.n_bone_tris = int(tri.shape[0] // 3)

        verts0 = _load_bone_verts(self.bone_dir, 1)
        self.mesh = wp.Mesh(
            points=wp.array(verts0, dtype=wp.vec3, device=self.device),
            indices=wp.array(tri, dtype=wp.int32, device=self.device),
        )
        # humanV2 is fully pin-driven (pts == all NV); use full inv_mass so the
        # push can act, running as the substep's last op to win over the pin.
        # Fascia (verts >= nvm) is the OUTER shell wrapping the muscles, which in
        # turn wrap the bone — fascia never touches bone, so any "contact" it
        # reports is a spurious mesh_query_point_sign_normal sign-flip that
        # catapults the vert. Exclude fascia from collision entirely (inv_mass=0).
        coll_inv_mass_np = sim.inv_mass.numpy()
        nvm = int(getattr(sim, "nvm", sim.NV))
        if nvm < sim.NV:
            coll_inv_mass_np[nvm:] = 0.0
        self.coll_inv_mass = wp.array(coll_inv_mass_np, dtype=wp.float32,
                                      device=self.device)
        # Meter mask: 1 for muscle verts (< nvm), 0 for fascia. The muscle-bone
        # penetration METER must measure the same set the resolver acts on —
        # fascia's "inside" is a sign-flip false positive (see above), so it is
        # excluded from is_collide_muscle_bone / pen_depth_muscle_bone too. (This
        # is independent of inv_mass, so pinned muscle verts still get measured.)
        meter_np = np.ones(int(sim.NV), dtype=np.float32)
        if nvm < sim.NV:
            meter_np[nvm:] = 0.0
        self.meter_mask = wp.array(meter_np, dtype=wp.float32, device=self.device)
        self._count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._depth = wp.zeros(sim.NV, dtype=wp.float32, device=self.device)
        self._cur_verts = verts0  # for viz / dump
        self._bone_stack = None   # frame -> verts, filled by preload()

    def preload(self, frames):
        """Parse every frame's bone pose once and hold it in RAM (see
        OneWaySim.preload_targets — same rationale, lifts JSON parse out of the
        step loop so only the upload + BVH refit remain per frame)."""
        self._bone_stack = {f: _load_bone_verts(self.bone_dir, f) for f in frames}

    def set_frame(self, frame: int):
        """Load this frame's bone pose into the mesh and rebuild the BVH."""
        if self._bone_stack is not None and frame in self._bone_stack:
            verts = self._bone_stack[frame]
        else:
            verts = _load_bone_verts(self.bone_dir, frame)
        if verts.shape[0] != self.n_bone_verts:
            raise ValueError(
                f"bone_{frame}.geo has {verts.shape[0]} verts, "
                f"expected {self.n_bone_verts} (topology must be constant)")
        self.mesh.points.assign(wp.array(verts, dtype=wp.vec3, device=self.device))
        self.mesh.refit()
        self._cur_verts = verts

    def __call__(self):
        wp.launch(
            solve_mesh_contact_pos_kernel, dim=self.sim.NV,
            inputs=[self.sim.pos, self.coll_inv_mass, self.mesh.id,
                    wp.float32(self.max_dist), wp.float32(self.contact_offset),
                    wp.float32(self.max_push)],
            device=self.device)

    def inside_count(self) -> int:
        self._count.zero_()
        wp.launch(_count_inside_kernel, dim=self.sim.NV,
                  inputs=[self.sim.pos, self.meter_mask, self.mesh.id,
                          wp.float32(self.max_dist), self._count],
                  device=self.device)
        return int(self._count.numpy()[0])

    def inside_depth(self) -> np.ndarray:
        """Per-muscle-vertex penetration depth (0 outside, >0 inside, metres)."""
        wp.launch(_mark_inside_kernel, dim=self.sim.NV,
                  inputs=[self.sim.pos, self.meter_mask, self.mesh.id,
                          wp.float32(self.max_dist), self._depth],
                  device=self.device)
        return self._depth.numpy()

    @property
    def tris_flat(self) -> np.ndarray:
        return self.mesh.indices.numpy().astype(np.int32)


def build_humanV2_sim(cfg, demo):
    """Build the OneWaySim + preload per-frame targets, and compute the
    muscle/fascia surface triangle lists for rendering.

    All knobs (fascia, multigrid, fiber, tendon pin) come from the json ``cfg``
    — ``OneWaySim`` reads them directly. ``demo`` only carries the run-level
    settings (device, activate)."""
    sim = MultiGridSim(cfg, device=demo.get("device", "cuda:0"))
    if getattr(sim, "use_fiber", False):
        print(f"[fiber] model={sim.fiber_model} k_fib={sim.fiber_stiffness:g} "
              f"contraction_ratio={sim.contraction_ratio:g}")
    # Optional: build the activation mask for a named muscle (e.g. bicep). The
    # run loop ramps this muscle's activation 0->1 across frames so the fiber
    # contraction is visible against the passive baseline.
    sim._activate_mask = None
    act_name = demo.get("activate")
    if act_name and getattr(sim, "use_fiber", False):
        mask = sim.tet_mask_for_muscles(act_name).astype(np.float32)
        sim._activate_mask = mask
        print(f"[activate] '{act_name}' -> {int(mask.sum())}/{mask.shape[0]} "
              f"muscle tets ramp 0->1")
    print(f"NV={sim.NV}  NT={sim.NT}  frames={sim.start_frame}..{sim.end_frame}")

    # Parse all per-frame cons/bone .geo up front (RAM-resident, .npy-backed) so
    # the step loop below is pure GPU work — no disk / JSON parse interleaved.
    t_load = time.perf_counter()
    frames = range(sim.start_frame, sim.end_frame + 1)
    sim.preload_targets(frames)
    print(f"[preload] {len(frames)} frames of targets "
          f"in {time.perf_counter() - t_load:.1f}s")

    # Muscle surface tris. In the new mg solver the muscle table is muscle-ONLY
    # (nvm == NV); fascia is a SEPARATE Fascia sub-solver with its own pos/tet
    # arrays (handled below). The legacy path (nvm < NV) kept the fascia shell
    # merged into the muscle table — split by vertex range there.
    tets = sim.tet_idx.numpy()
    nvm = getattr(sim, "nvm", sim.NV)
    if nvm < sim.NV:
        muscle_tets = tets[(tets < nvm).all(axis=1)]
        fascia_tets = tets[(tets >= nvm).all(axis=1)]
        surf_flat = vectorized_surface_tris(muscle_tets).reshape(-1).astype(np.int32)
        fascia_flat = vectorized_surface_tris(fascia_tets).reshape(-1).astype(np.int32)
        print(f"fascia (merged): {muscle_tets.shape[0]} muscle tets, "
              f"{fascia_tets.shape[0]} fascia tets ({sim.NV - nvm} fascia verts)")
    else:
        surf_flat = vectorized_surface_tris(tets).reshape(-1).astype(np.int32)
        fascia_flat = None

    # Separate Fascia sub-solver: its surface tris index fascia.pos (0..fascia.NV),
    # NOT the muscle pos array. The run loop renders it from sim.fascia.pos.
    if getattr(sim, "fascia", None) is not None:
        fascia_flat = vectorized_surface_tris(
            np.asarray(sim.fascia.tet_np, dtype=np.int64)).reshape(-1).astype(np.int32)
        print(f"fascia (separate sub-solver): {sim.fascia.NT} tets, "
              f"{sim.fascia.NV} verts")
    return sim, surf_flat, fascia_flat


def build_humanV2_sim_from_args(args):
    """Back-compat shim for the diagnostic ``tools/`` scripts that drive the
    builder with an argparse Namespace instead of a json ``demo`` block.

    Returns the legacy 4-tuple ``(sim, cfg, surf_flat, fascia_flat)``. New code
    should use ``build_humanV2_sim(cfg, demo)`` directly."""
    cfg = load_config(args.config)
    if getattr(args, "end_frame", None) is not None:
        cfg["end_frame"] = args.end_frame
    if getattr(args, "no_gmg", False):
        cfg["use_multigrid"] = False
    if getattr(args, "tendon_pin_threshold", None) is not None:
        cfg["tendon_pin_threshold"] = args.tendon_pin_threshold
    if getattr(args, "fiber", "none") not in ("none", None):
        cfg["use_fiber"] = True
        cfg["fiber_model"] = args.fiber
        cfg["fiber_stiffness"] = getattr(args, "fiber_stiffness", 1.0e8)
        cfg["contraction_ratio"] = getattr(args, "contraction_ratio", 0.4)
    demo = {"device": getattr(args, "device", "cuda:0"),
            "activate": getattr(args, "activate", None)}
    sim, surf_flat, fascia_flat = build_humanV2_sim(cfg, demo)
    return sim, cfg, surf_flat, fascia_flat


def run_humanV2_loop(sim, collider, cfg, demo, surf_flat, fascia_flat,
                     *, pre_step=None, mm_meter=None):
    """Per-frame step + USD/geo export, with ``collider`` as the muscle-BONE
    penetration MEASUREMENT instrument (bone mesh + inside_depth) regardless of
    who resolves contact.  ``pre_step(frame)`` runs after the bones are
    positioned but before the solve — the self-collision modes use it to refresh
    their colliders.

    ``mm_meter()`` is the optional muscle-vs-muscle (self) penetration meter: a
    callable returning a per-muscle-vertex depth array (metres, 0 outside). It
    is reported SEPARATELY from the bone meter — bone penetration is
    ``is_collide_muscle_bone`` / ``pen_depth_muscle_bone``, self penetration is
    ``is_collide_muscle_muscle`` / ``pen_depth_muscle_muscle``. Splitting them
    keeps the muscle-bone count from being inflated by the (different,
    deformable-surface) muscle-muscle overlaps.
    """
    coll = demo.get("collision", {})
    static_bone = bool(coll.get("static_bone", False))
    collide_eps = float(coll.get("collide_eps", 1e-6))
    no_export = bool(demo.get("no_export", False))
    no_collide_primvars = bool(demo.get("no_collide_primvars", True))
    out = demo.get("out", "output/humanV2.usd")
    geo_out = demo.get("geo_out")

    renderer = None
    if not no_export:
        from muscle.engine import WarpRenderer
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        renderer = WarpRenderer(mode="usd", stage_path=out)
        print(f"[render] writing USD to {out}")
    else:
        print("[no_export] pure-sim timing, no USD/geo output")

    geo_writer = None
    if geo_out and not no_export:
        tpl = cfg["rest_path"]
        geo_writer = GeoWriter(tpl)
        Path(geo_out).mkdir(parents=True, exist_ok=True)
        print(f"[geo] in-place .geo (template {tpl}) -> {geo_out}/")

    # Compact each prim to ONLY the verts its surface tris reference, so the USD
    # mesh carries no orphan points (interior muscle verts, fascia verts in the
    # muscle prim, etc). pos_np is the full NV set every frame; index it by the
    # prim's used-vert list and remap the tris to local [0, n) indices.
    def _compact(tris_flat, n_verts):
        used = np.unique(tris_flat)
        remap = np.full(n_verts, -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0])
        local = remap[tris_flat].astype(np.int32)
        return used, local

    # Fascia tris index the separate Fascia sub-solver's pos array (0..fascia.NV)
    # when sim.fascia is present; the legacy merged path indexes the muscle pos.
    fascia_sep = getattr(sim, "fascia", None) is not None
    muscle_used, muscle_local = _compact(surf_flat, sim.NV)
    fascia_used, fascia_local = (
        _compact(fascia_flat, sim.fascia.NV if fascia_sep else sim.NV)
        if fascia_flat is not None else (None, None))
    bone_used, bone_local = _compact(collider.tris_flat, collider.n_bone_verts)

    meter = CollisionMeter(collider, mm_meter, collide_eps, sim.NV)

    t0 = time.perf_counter()
    for frame in range(sim.start_frame, sim.end_frame + 1):
        if not static_bone:
            collider.set_frame(frame)              # move bones BEFORE solving
        if pre_step is not None:
            pre_step(frame)
        if getattr(sim, "_activate_mask", None) is not None:
            span = max(1, sim.end_frame - sim.start_frame)
            a = (frame - sim.start_frame) / span      # linear ramp 0 -> 1
            sim.set_fiber_activation(a * sim._activate_mask)
        sim.step_frame(frame)
        meter.measure()

        if renderer is not None:
            pos_np = sim.positions_np()
            inner = renderer.renderer
            # One USD timecode per sim frame. (begin_frame does round(time*fps);
            # passing frame*sim.dt collapses frames since sim.dt is the SUBSTEP dt.)
            inner.begin_frame(frame / inner.fps)
            # Constant displayColor only needs writing once — warp's render_mesh
            # re-sets it (time-sampled) on every call, so pass colors ONLY on the
            # first frame and None after to keep the USD small. (Topology / face
            # indices are already written once by warp; points must be per-frame.)
            first = frame == sim.start_frame
            muscle_path = inner.render_mesh(
                name="muscle", points=pos_np[muscle_used], indices=muscle_local,
                colors=(0.8, 0.3, 0.25) if first else None)
            meter.write_primvars(inner.stage, muscle_path, muscle_used, inner.time,
                                 per_vertex=not no_collide_primvars)
            if first:
                _bake_static_primvars(inner.stage, muscle_path, sim, muscle_used)
            if fascia_flat is not None:
                fascia_pos = (sim.fascia.pos.numpy() if fascia_sep else pos_np)
                inner.render_mesh(name="fascia", points=fascia_pos[fascia_used],
                                  indices=fascia_local,
                                  colors=(0.55, 0.75, 0.9) if first else None)
            inner.render_mesh(name="bones", points=collider._cur_verts[bone_used],
                              indices=bone_local,
                              colors=(0.9, 0.88, 0.8) if first else None)
            inner.end_frame()

            if geo_writer is not None:
                geo_writer.write(str(Path(geo_out) / f"muscle_{frame}.geo"),
                                 pos_np, meter.bone_flag, meter.bone_depth)
        if frame % 5 == 0 or frame == sim.end_frame:
            elapsed = time.perf_counter() - t0
            print(f"  frame {frame:>3d}  {elapsed:6.1f}s  {meter.summary()}")
    if renderer is not None:
        renderer.save()
        print(f"USD saved -> {out}")


def _make_self_collider(sim, cfg, demo, surf_flat):
    """Build the optional self-collision resolver from ``demo.self_collision``.

    Returns ``(post_fn_or_collider, pre_step, finalize, mm_meter)`` where
    ``post_fn`` is appended to ``collision_solvers`` BEFORE the SDF bone (so the
    bone gets the final word), ``pre_step(frame)`` refreshes the collider each
    frame, ``finalize()`` prints end-of-run stats, and ``mm_meter()`` returns a
    per-muscle-vertex muscle-muscle penetration depth array for the USD export
    (None if the method has no per-vertex meter). Any may be None."""
    sc = demo.get("self_collision", {}) or {}
    method = sc.get("method", "none")
    if method in (None, "none"):
        return None, None, None, None

    if method == "dcd":
        from muscle.collision.udcf.solver import (
            ConservativeBoundCcd, UdcfDcdCollider, UdcfParams)
        device = demo.get("device", "cuda:0")
        ccd = None
        if sc.get("ccd", False):
            ccd = ConservativeBoundCcd(
                relaxation=float(sc.get("ccd_relaxation", 0.42)),
                query_radius=float(sc.get("ccd_query_radius", 0.05)),
                device=device)
        dcd = UdcfDcdCollider(
            sim.pos, sim.inv_mass,
            surface_tris=surf_flat.reshape(-1, 3),
            ccd=ccd,
            params=UdcfParams(allow_self=True,
                              thickness=float(sc.get("thickness", 1e-3)),
                              multiplier=float(sc.get("multiplier", 0.3)),
                              friction_static=float(sc.get("mu_s", 0.5)),
                              friction_dynamic=float(sc.get("mu_d", 0.4)),
                              max_pairs=int(sc.get("max_pairs", 1 << 22)),
                              max_contacts=int(sc.get("max_contacts", 1 << 18)),
                              gauss_seidel=bool(sc.get("gauss_seidel", False)),
                              jacobi_avg=not bool(sc.get("gauss_seidel", False)),
                              use_bvh=not bool(sc.get("hashgrid", False)),
                              normal_cull=float(sc.get("normal_cull", 1.0))),
            device=device)
        # Both colliders run in post_phase (substep's final ops). Order: self DCD
        # first, SDF bone LAST — self-before-bone keeps the self relief while
        # letting bone get the final word (is_collide stays ~0). Per substep self
        # does K soft pushes; solve(0) rebuilds the broadphase, solve(1) reuses.
        # Broadphase rebuilds once per frame and is reused across substeps unless
        # rebuild_substeps is set (the 2.5M-pair grid is near-constant in a frame).
        state = {"fresh": True}
        K = max(1, int(sc.get("iters", 2)))
        every = bool(sc.get("rebuild_substeps", False))

        def self_fn(_st=state, _d=dcd, _K=K, _every=every):
            rebuild = _st["fresh"] or _every
            _st["fresh"] = False
            _d.solve(0 if rebuild else 1)
            for _ in range(_K - 1):
                _d.solve(1)

        def pre_step(frame, _st=state):
            _st["fresh"] = True

        def finalize():
            print(f"DCD self final: {dcd.stats()}")

        print(f"[self] DCD muscle self-collision (mult={sc.get('multiplier', 0.3)} "
              f"x {K} iters)")
        # DCD has no cheap per-vertex inside-depth meter (its contacts are
        # tri-tri pairs, not signed point queries) — no muscle-muscle primvar.
        return self_fn, pre_step, finalize, None

    if method == "intermuscle":
        from muscle.collision.closest_point.intermuscle import InterMuscleCollider
        inter = InterMuscleCollider(sim, cfg["rest_path"],
                                    max_dist=float(sc.get("max_dist", 0.02)),
                                    contact_offset=float(sc.get("offset", 1e-4)),
                                    iters=int(sc.get("iters", 1)),
                                    substep_stride=int(sc.get("stride", 2)),
                                    push_cap=float(sc.get("push_cap", 0.0)),
                                    refit_stride=int(sc.get("refit_stride", 1)),
                                    device=demo.get("device", "cuda:0"))
        print(f"[self] inter-muscle collider: {inter.n_muscle} muscles, "
              f"max_dist={sc.get('max_dist', 0.02)} offset={sc.get('offset', 1e-4)} "
              f"push_cap={sc.get('push_cap', 0.0)} iters={inter.iters}")

        def finalize():
            print(f"inter-muscle final inside_count={inter.inside_count()}  "
                  f"pairs={inter.last_pairs}")

        # Muscle-muscle meter for the USD export: deepest inter-muscle
        # penetration per muscle vertex. Use a generous search radius so deep
        # overlap is measured, not capped at the resolver's max_dist.
        meter_md = float(sc.get("meter_max_dist", max(0.05, 2.5 * float(
            sc.get("max_dist", 0.02)))))
        mm_meter = lambda: inter.inside_depth(meter_max_dist=meter_md)
        return inter, (lambda frame: inter.refresh()), finalize, mm_meter

    raise ValueError(f"unknown self_collision.method '{method}' "
                     f"(expected none | dcd | intermuscle)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="data/humanV2/config/humanV2.json",
                    help="json scene+demo config (the ONLY argument; every other "
                         "knob lives in the json's solver keys + 'demo' object)")
    args = ap.parse_args(argv)

    wp.init()
    cfg = load_config(args.config)
    demo = cfg.get("demo", {}) or {}

    sim, surf_flat, fascia_flat = build_humanV2_sim(cfg, demo)

    coll = demo.get("collision", {}) or {}
    collision_on = bool(coll.get("enabled", True))
    static_bone = bool(coll.get("static_bone", False))
    bone_dir = demo.get("bone_dir", "data/humanV2/model/bone")

    # Always build the bone collider — it is the penetration METER (inside_depth
    # drives the USD primvars) even when collision resolution is disabled.
    collider = AnimatedBoneMeshCollider(
        sim, bone_dir,
        max_dist=float(coll.get("max_dist", 0.2)),
        contact_offset=float(coll.get("contact_offset", 1e-5)),
        # <=0 (default) = push every inside vert out, however deep (resolves the
        # non-collision-free target). >0 re-enables the sign-flip-spike reject
        # at that depth (metres).
        max_push=float(coll.get("max_push", -1.0)))
    if not static_bone:
        collider.preload(range(sim.start_frame, sim.end_frame + 1))

    # Optional self-collision resolver (post_phase, BEFORE the SDF bone).
    self_collider, self_pre, self_finalize, self_mm_meter = _make_self_collider(
        sim, cfg, demo, surf_flat)

    post = []
    if self_collider is not None:
        post.append(self_collider)             # self push first ...
    if collision_on:
        post.append(collider)                  # ... SDF bone LAST (final word)
    if post:
        sim.collision_solvers = post

    print(f"humanV2: bone={'SDF' if collision_on else 'OFF(meter only)'}  "
          f"self={demo.get('self_collision', {}).get('method', 'none')}  "
          f"({'STATIC@f1' if static_bone else 'ANIMATED'} skeleton, "
          f"{collider.n_bone_verts} bone verts)")

    run_humanV2_loop(sim, collider, cfg, demo, surf_flat, fascia_flat,
                     pre_step=self_pre, mm_meter=self_mm_meter)
    if self_finalize is not None:
        self_finalize()


if __name__ == "__main__":
    main()
