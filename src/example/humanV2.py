"""humanV2 muscle demo — single json-driven entry point.

Baseline (``data/humanV2/config/humanV2.json``):
    fascia shell + animated-bone SDF collision, multigrid OFF (No MG).

This file MERGES the four old humanV2 examples (sdf bone, sdf+fascia, sdf+DCD
self, sdf+inter-muscle self). Every knob is read from the json config — the
ONLY command-line argument is ``--config``::

    PYTHONPATH=src python -m example.humanV2 --config data/humanV2/config/humanV2.json
    # -> output/humanV2.usd

The skeleton is driven by a constant-topology, per-frame animation from one of
two backends (top-level keys, shared by the sim target_mode + the collider):

    "bone_dir": "data/humanV2/model/bone"   # dir of bone_<N>.geo (default), OR
    "bone_usd": "data/Emil/model/bone.usd"  # single USD, points timeSamples
    "bone_root": "/character/bone"          # optional: scope which USD prims

Each frame the bone verts are loaded into a ``wp.Mesh`` and ``refit()``-ed
before the solve, so penetration stays shallow / local.

Config schema (``demo`` object; solver keys stay at the json top level and are
consumed by ``OneWaySim``):

    "demo": {
      "out": "output/humanV2.usd",
      "graft_src": null,                    # set -> graft sim meshes onto a copy of
      "sim_scope": "/char/sim",             # this UsdSkel asset under sim_scope; bones
                                            #   ride its skel anim (no baked bone points)
      "device": "cuda:0",                   # NB: bone_dir lives at the json top level
                                            #     (shared by sim target_mode + collider)
      "no_export": false,
      "geo_out": null,
      "output_collide_primvars": false,    # per-vertex is_collide/pen_depth primvars
      "activate": null,                      # muscle-name substring; needs use_fiber
      "collision": {
        "enabled": true,
        "static_bone": false,
        "max_dist": 0.2,
        "contact_offset": 1e-5,
        "collide_eps": 1e-6,
        "flip_bone_winding": true            # .geo bones wind inward; USD may not
      },
      "self_collision": {                     # method: none | dcd | intermuscle
        "method": "none"
      }
    }
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import warp as wp

from muscle.mg.mg import MultiGridSim, load_config
from muscle.util.mesh_io import vectorized_surface_tris
from muscle.collision.sdf.kernel import solve_mesh_contact_pos_kernel
from muscle.util.geo_io import GeoFile, GeoWriter, read_geo_attr_cached
from muscle.util.bone_source import make_bone_source


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


def _set_primvar(stage, prim_path, name, vtype, value, time, *, interp=None):
    """Get-or-create a USD primvar on a mesh prim and set ``value`` at ``time`` —
    the single get-or-create-set idiom shared by every primvar writer here.
    ``value`` is the already-wrapped Vt value (caller owns the numpy->Vt conversion).
    ``interp`` defaults to vertex; pass ``UsdGeom.Tokens.constant`` for whole-prim
    scalars. Returns the primvar."""
    from pxr import UsdGeom
    api = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(prim_path))
    pv = api.GetPrimvar(name) or api.CreatePrimvar(
        name, vtype, interp or UsdGeom.Tokens.vertex)
    pv.Set(value, time)
    return pv


def _set_is_collide_primvar(stage, prim_path, flag01, time,
                            name="is_collide_muscle_bone"):
    """Per-vertex int primvar (0/1) marking penetrating muscle verts so usdview can
    select / colour them even when hidden inside the bone mesh. ``name`` separates
    muscle-vs-bone (``is_collide_muscle_bone``) from muscle-vs-muscle/self
    (``is_collide_muscle_muscle``)."""
    from pxr import Sdf, Vt
    _set_primvar(stage, prim_path, name, Sdf.ValueTypeNames.IntArray,
                 Vt.IntArray.FromNumpy(flag01.astype(np.int32)), time)


def _set_pen_depth_primvar(stage, prim_path, depth, time,
                           name="pen_depth_muscle_bone"):
    """Per-vertex float primvar (metres) of penetration depth. ``name`` is
    ``pen_depth_muscle_bone`` or ``pen_depth_muscle_muscle``."""
    from pxr import Sdf, Vt
    _set_primvar(stage, prim_path, name, Sdf.ValueTypeNames.FloatArray,
                 Vt.FloatArray.FromNumpy(depth.astype(np.float32)), time)


def _set_sum_is_collide(stage, prim_path, total, time,
                        name="sum_is_collide_muscle_bone"):
    """Per-frame scalar custom int ATTR (not a primvar). ``name`` is
    ``sum_is_collide_muscle_bone`` or ``sum_is_collide_muscle_muscle``."""
    from pxr import Sdf
    prim = stage.GetPrimAtPath(prim_path)
    attr = prim.GetAttribute(name) or prim.CreateAttribute(name, Sdf.ValueTypeNames.Int)
    attr.Set(int(total), time)


def _set_global_primvar(stage, prim_path, name, value, time, vtype):
    """Whole-prim (constant-interpolation) scalar primvar, time-sampled — one value
    per frame for the entire mesh (global diagnostics like inverted-tet count /
    percentage that usdview can plot)."""
    from pxr import UsdGeom
    _set_primvar(stage, prim_path, name, vtype, value, time,
                 interp=UsdGeom.Tokens.constant)


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
    from pxr import Sdf, Vt, Usd
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
        _set_primvar(stage, prim_path, name, Sdf.ValueTypeNames.Float3Array,
                     Vt.Vec3fArray.FromNumpy(
                         np.ascontiguousarray(_full(arr)[used], np.float32)), default)

    def _scalar(name, arr):
        _set_primvar(stage, prim_path, name, Sdf.ValueTypeNames.FloatArray,
                     Vt.FloatArray.FromNumpy(
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
            _set_primvar(stage, prim_path, "displayColor",
                         Sdf.ValueTypeNames.Color3fArray, Vt.Vec3fArray.FromNumpy(
                             np.ascontiguousarray(_full(cd[:, :3])[used], np.float32)),
                         default)
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
        _set_primvar(stage, prim_path, "muscle_id", Sdf.ValueTypeNames.TokenArray,
                     Vt.TokenArray(names), default)
        baked.append("muscle_id")
    if baked:
        print(f"[usd] baked static primvars: {', '.join(baked)}")


def _graft_sim_into_asset(out, src_usd, sim_stage_path, scope_path, mesh_names,
                          start, end):
    """Build the final output by GRAFTING the freshly-simulated meshes onto a
    self-contained copy of the source asset.

    The source ``src_usd`` (e.g. Emil.usd) keeps its compact UsdSkel bones +
    skinned surface muscle — the skeleton plays back via LBS, so NO per-frame bone
    points are baked (the real space saving vs a full mesh bake). Only the
    deforming sim meshes cost per-frame points; warp has already authored them
    (the proven-visible Mesh attrs) into ``sim_stage_path`` under ``/root/<name>``
    in WORLD metres.

    Those meshes are copied (``Sdf.CopySpec``, timesamples and all) under a new
    ``scope_path`` (e.g. ``/char/sim``, leaving room for ``/char/sim/fascia`` etc.)
    so the simulation lives beside the asset's own ``/char/muscle``. The asset's
    parent (``/char``) carries a cm->m world xform; the scope gets that xform's
    INVERSE so the world-metre sim points render in place (otherwise the 0.01
    scale shrinks them 100x). Exported as ONE dependency-free ``.usd`` (the source
    has no sublayers/references)."""
    from pxr import Usd, UsdGeom, Sdf
    stage = Usd.Stage.Open(src_usd)
    parent = scope_path.rsplit("/", 1)[0] or "/"
    tinv = (UsdGeom.Xformable(stage.GetPrimAtPath(parent))
            .ComputeLocalToWorldTransform(float(start)).GetInverse())
    scope = UsdGeom.Xform(stage.DefinePrim(scope_path, "Xform"))
    scope.AddTransformOp().Set(tinv)
    stage.SetStartTimeCode(float(start))
    stage.SetEndTimeCode(float(end))
    dst = stage.GetRootLayer()
    sim = Sdf.Layer.FindOrOpen(os.path.abspath(sim_stage_path))
    grafted = []
    for nm in mesh_names:
        s, d = f"/root/{nm}", f"{scope_path}/{nm}"
        if sim.GetPrimAtPath(s):
            Sdf.CreatePrimInLayer(dst, d)
            Sdf.CopySpec(sim, s, dst, d)
            grafted.append(nm)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    stage.Export(out)
    print(f"[graft] {os.path.basename(src_usd)} (skel bones) + {scope_path}/"
          f"{{{','.join(grafted)}}} -> {out} (self-contained)")


class AnimatedBoneMeshCollider:
    """Per-frame skeleton mesh as a moving rigid collider for the muscle.

    Topology (triangles) is constant across the sequence, so the ``wp.Mesh`` is
    built once and only its ``points`` are refreshed + ``refit()`` each frame.
    Vertices come from a :class:`BoneSource` (``.geo`` dir or animated USD).
    """

    def __init__(self, sim: MultiGridSim, bone_source,
                 max_dist: float = 0.2, contact_offset: float = 0.0,
                 max_push: float = -1.0,      # <=0 = no deep-contact reject
                 flip_winding: bool = True):
        self.sim = sim
        self.device = sim._device
        self.source = bone_source
        self.start_frame = int(sim.start_frame)
        self.max_dist = float(max_dist)
        self.contact_offset = float(contact_offset)
        self.max_push = float(max_push)

        tri = self.source.tris.reshape(-1, 3)
        # SNU bones are wound inward relative to Warp's signed query — without
        # flipping, mesh_query_point_sign_normal reports ~155640/157369 verts
        # "inside" at rest vs the true ~1729. Swap each tri's last two indices.
        # (USD bones may already wind correctly — toggle via flip_bone_winding.)
        if flip_winding:
            tri = tri[:, [0, 2, 1]]
        tri = np.ascontiguousarray(tri).reshape(-1)
        self.n_bone_verts = int(self.source.n_verts)
        self.n_bone_tris = int(tri.shape[0] // 3)

        verts0 = self.source.positions(self.start_frame)
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
        self._depth = wp.zeros(sim.NV, dtype=wp.float32, device=self.device)
        self._cur_verts = verts0  # for viz / dump
        self._bone_stack = None   # frame -> verts, filled by preload()

    def preload(self, frames):
        """Parse every frame's bone pose once and hold it in RAM (see
        OneWaySim.preload_targets — same rationale, lifts JSON parse out of the
        step loop so only the upload + BVH refit remain per frame)."""
        self._bone_stack = {f: self.source.positions(f) for f in frames}

    def set_frame(self, frame: int):
        """Load this frame's bone pose into the mesh and rebuild the BVH."""
        if self._bone_stack is not None and frame in self._bone_stack:
            verts = self._bone_stack[frame]
        else:
            verts = self.source.positions(frame)
        if verts.shape[0] != self.n_bone_verts:
            raise ValueError(
                f"bone frame {frame} has {verts.shape[0]} verts, "
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


def build_humanV2_sim(cfg):
    """Build the OneWaySim + preload per-frame targets, and compute the
    muscle/fascia surface triangle lists for rendering.

    All knobs (fascia, multigrid, fiber, tendon pin) come from the json ``cfg`` —
    ``OneWaySim`` reads them directly. The ``cfg['demo']`` sub-object only carries the
    run-level settings (device, activate)."""
    demo = cfg.get("demo", {}) or {}
    sim = MultiGridSim(cfg, device=demo.get("device", "cuda:0"))
    if getattr(sim, "use_fiber", False):
        print(f"[fiber] model={sim.fiber_model} k_fib={sim.fiber_stiffness:g} "
              f"contraction_ratio={sim.contraction_ratio:g}")
    # Fiber activation source — a polymorphic object with per_tet(frame) -> (NT,);
    # the run loop just asks it each frame (USD curve / ramp / RL policy / ... all
    # interchangeable). Precedence: a USD per-muscle curve asset, else the single
    # named-muscle 0->1 ramp, else None (passive). Only when the solver runs fiber.
    sim.activation_source = None
    if getattr(sim, "use_fiber", False):
        act_usd = cfg.get("activation_usd") or demo.get("activation_usd")
        act_name = demo.get("activate")
        if act_usd:
            # USD curve asset (e.g. walk.usd): per-muscle activation curves mapped
            # onto per-tet via muscle_mapping.json.
            from muscle.util.activation import UsdActivationSource
            map_path = cfg.get("muscle_mapping") or demo.get("muscle_mapping")
            if not map_path:
                raise ValueError("activation_usd set but 'muscle_mapping' missing "
                                 "(curve -> muscle_id table)")
            sim.activation_source = UsdActivationSource(
                usd_path=act_usd, mapping_path=map_path,
                muscle_id_np=sim.muscle_id_np, muscle_id_strings=sim.muscle_id_strings,
                start_frame=sim.start_frame, end_frame=sim.end_frame,
                muscles_root=cfg.get("activation_muscles_root", "/Human/Muscles"),
                aggregate=cfg.get("activation_aggregate", "mean"))
        elif act_name:
            # Single named muscle (e.g. bicep) ramped 0->1 so the fiber contraction
            # is visible against the passive baseline.
            from muscle.util.activation import RampActivationSource
            mask = sim.tet_mask_for_muscles(act_name).astype(np.float32)
            sim.activation_source = RampActivationSource(
                mask, sim.start_frame, sim.end_frame)
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
    elif getattr(sim, "surface_tris_np", None) is not None:
        surf_flat = np.asarray(sim.surface_tris_np, dtype=np.int32).reshape(-1)
        fascia_flat = None
        print(f"muscle surface: authored {surf_flat.shape[0] // 3} tris")
    else:
        surf_flat = vectorized_surface_tris(tets).reshape(-1).astype(np.int32)
        fascia_flat = None

    # Separate Fascia sub-solver: its surface tris index fascia.pos (0..fascia.NV),
    # NOT the muscle pos array. The run loop renders it from sim.fascia.pos.
    if getattr(sim, "fascia", None) is not None:
        if getattr(sim.fascia, "surface_tris_np", None) is not None:
            fascia_flat = np.asarray(sim.fascia.surface_tris_np,
                                     dtype=np.int32).reshape(-1)
        else:
            fascia_flat = vectorized_surface_tris(
                np.asarray(sim.fascia.tet_np, dtype=np.int64)).reshape(-1).astype(np.int32)
        print(f"fascia (separate sub-solver): {sim.fascia.NT} tets, "
              f"{sim.fascia.NV} verts")
    # Stash the resolved render/collision surface ON the sim it describes, so callers
    # read sim.surf_flat / sim.fascia_flat instead of threading loose params. Both are
    # flat int32 (groups of 3 = a tri); surf_flat indexes sim.pos, fascia_flat indexes
    # sim.fascia.pos (or muscle pos on the legacy merged path); fascia_flat is None
    # when there is no fascia.
    sim.surf_flat = surf_flat
    sim.fascia_flat = fascia_flat
    return sim


def build_humanV2_sim_from_args(args):
    """Back-compat shim for the diagnostic ``tools/`` scripts that drive the
    builder with an argparse Namespace instead of a json ``demo`` block.

    Returns the legacy 4-tuple ``(sim, cfg, surf_flat, fascia_flat)``. New code
    should use ``build_humanV2_sim(cfg)`` directly."""
    cfg = load_config(args.config)
    if getattr(args, "end_frame", None) is not None:
        cfg["end_frame"] = args.end_frame
    if getattr(args, "no_gmg", False):
        cfg["use_multigrid"] = False
    # CLI overrides patch the constraints[] array (bicep.json style) in place.
    cons = cfg.setdefault("constraints", [])
    _by_name = {c.get("name"): c for c in cons}
    if getattr(args, "tendon_pin_threshold", None) is not None:
        _by_name["belly_attach"]["tendon_pin_threshold"] = args.tendon_pin_threshold
    if getattr(args, "fiber", "none") not in ("none", None):
        cons.append(dict(type="fiber", name="fiber", fiber_model=args.fiber,
                         stiffness=getattr(args, "fiber_stiffness", 1.0e8),
                         dampingratio=0.0,
                         contraction_ratio=getattr(args, "contraction_ratio", 0.4)))
    # Inject the CLI overrides into cfg['demo'] (the builder reads device/activate
    # from there) so the sole input is the cfg dict.
    demo = cfg.setdefault("demo", {})
    demo["device"] = getattr(args, "device", "cuda:0")
    demo["activate"] = getattr(args, "activate", None)
    sim = build_humanV2_sim(cfg)
    return sim, cfg, sim.surf_flat, sim.fascia_flat


class HumanV2:
    """humanV2 orchestration layer: drives the ``MultiGridSim`` one output frame at
    a time (activation input + collider pose + sim dispatch) and exports each frame.
    ``step_frame`` is the orchestration twin of ``MultiGridSim.step_frame`` (solver):
    this one sets the per-frame inputs and moves the external bone collider, then
    hands off to the sim's stepper.

    ``__init__`` builds the bone collider (``self.collider`` — the muscle-BONE
    penetration MEASUREMENT instrument, bone mesh + inside_depth, regardless of who
    resolves contact) and the optional self-collision resolver, and wires the substep
    collision pass (self push first, SDF bone last). The self resolver supplies
    ``self.pre_step(frame)`` (runs after the bones are positioned but before the solve
    to refresh its collider) and an optional muscle-vs-muscle meter.

    Bone and self penetration are metered SEPARATELY — bone is
    ``is_collide_muscle_bone`` / ``pen_depth_muscle_bone``, self is
    ``is_collide_muscle_muscle`` / ``pen_depth_muscle_muscle``. Splitting them keeps
    the muscle-bone count from being inflated by the (different, deformable-surface)
    muscle-muscle overlaps.

    Structure: ``__init__`` is build-once setup; ``run`` loops ``step_frame`` + export.

      * ``_compact`` trims every prim (muscle / fascia / bones) to ONLY the verts its
        surface tris touch, remapping the tris to local indices; packed into
        ``self.compaction`` and shared by all the export helpers.
      * Per frame, ``_render_frame`` authors that frame's full meshes plus the
        optional in-place ``.geo`` dump.

    Output modes (default = full WarpRenderer bake to ``demo.out``):
      * ``demo.graft_src``: warp renders ONLY the sim meshes (muscle/fascia, no
        bones) to a sidecar cache, then ``_graft_sim_into_asset`` copies them onto a
        self-contained copy of that source asset under ``demo.sim_scope`` (default
        ``/char/sim``). The asset's UsdSkel bones ride a compact skel animation
        instead of baked per-frame points — the real space saving.
    """

    def __init__(self, sim, cfg):
        self.sim = sim
        self.fascia_flat = sim.fascia_flat

        demo = cfg.get("demo", {}) or {}
        coll = demo.get("collision", {}) or {}
        self.collision_on = bool(coll.get("enabled", True))
        self.static_bone = bool(coll.get("static_bone", False))
        collide_eps = float(coll.get("collide_eps", 1e-6))
        no_export = bool(demo.get("no_export", False))

        # Bone collider — the muscle-bone penetration METER (inside_depth -> USD
        # primvars) even when resolution is off. Single source of truth: the top-level
        # bone source (also read by the sim for target_mode='mesh') — a 'bone_usd'
        # file or a 'bone_dir' of bone_<frame>.geo.
        bone_source = make_bone_source(cfg, sim.start_frame)
        self.collider = AnimatedBoneMeshCollider(
            sim, bone_source,
            max_dist=float(coll.get("max_dist", 0.2)),
            contact_offset=float(coll.get("contact_offset", 1e-5)),
            # <=0 (default) = push every inside vert out, however deep; >0 re-enables
            # the sign-flip-spike reject at that depth (metres).
            max_push=float(coll.get("max_push", -1.0)),
            # SNU .geo bones wind inward -> flip; USD bones may not -> override.
            flip_winding=bool(coll.get("flip_bone_winding", True)))
        if not self.static_bone:
            self.collider.preload(range(sim.start_frame, sim.end_frame + 1))

        # Optional self-collision resolver (post_phase, BEFORE the SDF bone). It owns
        # pre_step (per-frame collider refresh), the end-of-run finalize, and the
        # muscle-muscle meter; any may be None.
        self_collider, self.pre_step, self._self_finalize, mm_meter = \
            _make_self_collider(sim, cfg)
        # Substep collision pass: self push first, SDF bone LAST (final word).
        post = []
        if self_collider is not None:
            post.append(self_collider)
        if self.collision_on:
            post.append(self.collider)
        if post:
            sim.collision_solvers = post
        print(f"humanV2: bone={'SDF' if self.collision_on else 'OFF(meter only)'}  "
              f"self={demo.get('self_collision', {}).get('method', 'none')}  "
              f"({'STATIC@f1' if self.static_bone else 'ANIMATED'} skeleton, "
              f"{self.collider.n_bone_verts} bone verts)")

        self.output_collide_primvars = bool(demo.get("output_collide_primvars", False))
        self.out = demo.get("out", "output/humanV2.usd")
        self.geo_out = demo.get("geo_out")

        # Output mode: full WarpRenderer bake by default; 'graft_src' renders only the
        # sim meshes to a sidecar cache and grafts them onto a copy of the asset (bones
        # ride the asset's compact skel anim). Warp always writes to ``sim_out`` (== out
        # for the full bake, a sidecar cache for graft).
        self.graft_src = demo.get("graft_src")
        self.sim_scope = demo.get("sim_scope", "/char/sim")
        self.renderer = None
        self.sim_out = self.out
        if no_export:
            print("[no_export] pure-sim timing, no USD/geo output")
        else:
            from muscle.engine import WarpRenderer
            if self.graft_src:
                self.sim_out = str(Path(self.out).with_name(Path(self.out).stem + ".simcache.usd"))
            Path(self.sim_out).parent.mkdir(parents=True, exist_ok=True)
            self.renderer = WarpRenderer(mode="usd", stage_path=self.sim_out)
            print(f"[render] writing USD to {self.sim_out}"
                  + (f"  (graft {os.path.basename(self.graft_src)} -> {self.out})"
                     if self.graft_src else ""))

        self.geo_writer = None
        if self.geo_out and not no_export:
            tpl = cfg["rest_path"]
            self.geo_writer = GeoWriter(tpl)
            Path(self.geo_out).mkdir(parents=True, exist_ok=True)
            print(f"[geo] in-place .geo (template {tpl}) -> {self.geo_out}/")

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
        muscle_used, muscle_local = _compact(sim.surf_flat, sim.NV)
        fascia_used, fascia_local = (
            _compact(self.fascia_flat, sim.fascia.NV if fascia_sep else sim.NV)
            if self.fascia_flat is not None else (None, None))
        bone_used, bone_local = _compact(self.collider.tris_flat,
                                          self.collider.n_bone_verts)
        self.compaction = (muscle_used, muscle_local, fascia_used, fascia_local,
                           bone_used, bone_local, fascia_sep)

        self.meter = CollisionMeter(self.collider, mm_meter, collide_eps, sim.NV)

        # Sub-frame bone: sample the bone pose at `nsubsteps` intermediate poses easing
        # frame-1 -> frame so the muscle tracks the bone's PATH (no per-frame teleport ->
        # less hysteretic detach). Needs a per-frame-interpolatable source (UsdSkel /
        # baked points); fascia advect is per-frame-only so it is disallowed here.
        self.nsub = int(getattr(sim, "nsubsteps", 1))
        self.subframe = bool(cfg.get("subframe_bone", False)) and self.nsub > 1
        if self.subframe and getattr(sim, "fascia", None) is not None:
            raise ValueError("subframe_bone is not supported with fascia "
                             "(set use_fascia=false or subframe_bone=false)")
        if self.subframe:
            print(f"[subframe] bone sampled at {self.nsub} sub-poses/frame (frame-1 -> frame)")

    def _set_activation(self, frame: int) -> None:
        """Per-frame fiber activation input from the sim's activation source (any
        object with ``per_tet(frame)`` — USD curve, ramp, ...; None -> passive)."""
        src = getattr(self.sim, "activation_source", None)
        if src is not None:
            self.sim.set_fiber_activation(src.per_tet(frame))

    def step_frame(self, frame: int) -> None:
        """Drive the sim one output frame (no export): refresh self-colliders, set
        activation, move the bone collider, then hand off to the sim's solver. Held
        bone -> ``sim.step_frame``; sub-frame -> the bone eases frame-1 -> frame over
        ``nsub`` ``sim.substep_at_frac_frame`` calls (collider moved each substep)."""
        sim = self.sim
        if self.pre_step is not None:
            self.pre_step(frame)
        self._set_activation(frame)
        if self.subframe and frame > sim.start_frame:
            for s in range(self.nsub):                 # bone eases frame-1 -> frame
                t = (frame - 1) + (s + 1) / self.nsub  # last substep lands on `frame`
                if not self.static_bone:
                    self.collider.set_frame(t)         # interpolated bone for SDF
                sim.substep_at_frac_frame(t)           # target@t + pin + one substep
        else:
            if not self.static_bone:
                self.collider.set_frame(frame)         # move bones BEFORE solving
            sim.step_frame(frame)

    def run(self) -> None:
        """Loop every output frame: ``step_frame`` + bone/self meters + USD/geo export
        + progress print, then ``_finalize`` (save / graft)."""
        sim = self.sim
        t0 = time.perf_counter()
        for frame in range(sim.start_frame, sim.end_frame + 1):
            self.step_frame(frame)
            self.meter.measure()
            n_inv = sim.count_inverted()   # every frame: feeds the USD primvar + print
            if self.renderer is not None:
                self._render_frame(frame, n_inv)
            if frame % 5 == 0 or frame == sim.end_frame:
                elapsed = time.perf_counter() - t0
                print(f"  frame {frame:>3d}  {elapsed:6.1f}s  {self.meter.summary()}  "
                      f"inverted {n_inv}/{sim.NT} ({n_inv / sim.NT * 100:.2f}%)")
        self._finalize()
        if self._self_finalize is not None:
            self._self_finalize()              # self-collision end-of-run stats

    def _render_frame(self, frame: int, n_inv=None) -> None:
        """Author one frame of output: muscle/fascia (and, unless grafting, bone)
        meshes into the USD plus the optional in-place .geo dump. First frame gates
        the write-once data (colors, static primvars) so the per-frame USD stays
        small. ``n_inv`` is baked as global per-frame inverted-tet count/pct primvars.
        Bones are skipped under graft output (the source asset's UsdSkel bones ride a
        compact skel animation instead of baked per-frame points)."""
        sim = self.sim
        (muscle_used, muscle_local, fascia_used, fascia_local,
         bone_used, bone_local, fascia_sep) = self.compaction
        pos_np = sim.positions_np()
        inner = self.renderer.renderer
        # One USD timecode per sim frame. (begin_frame does round(time*fps); passing
        # frame*sim.dt collapses frames since sim.dt is the SUBSTEP dt.)
        inner.begin_frame(frame / inner.fps)
        first = frame == sim.start_frame
        # Constant displayColor only needs writing once — warp's render_mesh re-sets it
        # (time-sampled) every call, so pass colors ONLY on the first frame and None
        # after to keep the USD small. (Topology already written once by warp; points
        # must be per-frame.)
        muscle_path = inner.render_mesh(
            name="muscle", points=pos_np[muscle_used], indices=muscle_local,
            colors=(0.8, 0.3, 0.25) if first else None)
        self.meter.write_primvars(inner.stage, muscle_path, muscle_used, inner.time, per_vertex=self.output_collide_primvars)
        if first:
            _bake_static_primvars(inner.stage, muscle_path, sim, muscle_used)
        if self.fascia_flat is not None:
            fascia_pos = (sim.fascia.pos.numpy() if fascia_sep else pos_np)
            inner.render_mesh(name="fascia", points=fascia_pos[fascia_used],
                              indices=fascia_local,
                              colors=(0.55, 0.75, 0.9) if first else None)
        if not self.graft_src:
            inner.render_mesh(name="bones", points=self.collider._cur_verts[bone_used],
                              indices=bone_local,
                              colors=(0.9, 0.88, 0.8) if first else None)
        # Global per-frame inversion diagnostics on the muscle prim (constant interp).
        if n_inv is not None:
            from pxr import Sdf
            pct = (n_inv / sim.NT * 100.0) if sim.NT else 0.0
            _set_global_primvar(inner.stage, muscle_path, "inverted_count",
                                int(n_inv), inner.time, Sdf.ValueTypeNames.Int)
            _set_global_primvar(inner.stage, muscle_path, "inverted_pct",
                                float(pct), inner.time, Sdf.ValueTypeNames.Float)
        inner.end_frame()
        if self.geo_writer is not None:
            self.geo_writer.write(str(Path(self.geo_out) / f"muscle_{frame}.geo"),
                                  pos_np, self.meter.bone_flag, self.meter.bone_depth)

    def _finalize(self) -> None:
        if self.renderer is None:
            return
        sim = self.sim
        self.renderer.save()
        if self.graft_src:
            names = ["muscle"] + (["fascia"] if self.fascia_flat is not None else [])
            _graft_sim_into_asset(self.out, self.graft_src, self.sim_out, self.sim_scope,
                                  names, sim.start_frame, sim.end_frame)
            if os.path.exists(self.sim_out):
                os.remove(self.sim_out)
        print(f"USD saved -> {self.out}")


def _make_self_collider(sim, cfg):
    """Build the optional self-collision resolver from ``cfg['demo'].self_collision``.

    Returns ``(post_fn_or_collider, pre_step, finalize, mm_meter)`` where
    ``post_fn`` is appended to ``collision_solvers`` BEFORE the SDF bone (so the
    bone gets the final word), ``pre_step(frame)`` refreshes the collider each
    frame, ``finalize()`` prints end-of-run stats, and ``mm_meter()`` returns a
    per-muscle-vertex muscle-muscle penetration depth array for the USD export
    (None if the method has no per-vertex meter). Any may be None."""
    demo = cfg.get("demo", {}) or {}
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
            surface_tris=sim.surf_flat.reshape(-1, 3),
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
    ap.add_argument("--config", default="data/humanV2/config/otto.json",
                    help="json scene+demo config (the ONLY argument; every other "
                         "knob lives in the json's solver keys + 'demo' object)")
    args = ap.parse_args(argv)

    wp.init()
    cfg = load_config(args.config)

    sim = build_humanV2_sim(cfg)
    # HumanV2 builds the bone collider + optional self-collider, wires the substep
    # collision pass, and runs the per-frame loop + export.
    HumanV2(sim, cfg).run()


if __name__ == "__main__":
    main()
