"""Animated bone collider for humanV2: muscle vs the MOVING skeleton.

The static-bone version (``example_humanV2_bone``) holds the skeleton at its
rest pose while the muscle is pulled away by the per-frame pin targets, so the
muscle plunges ever deeper into the frozen bones (verts_inside grows ~250 ->
~4000 over 20 frames) and the single-push contact cannot keep up once a vertex
sinks past ``max_dist``.

Here the bones are driven by the same 200-frame animation that generated the
muscle's ``cons`` targets — a per-frame mesh sequence
``data/humanV2/model/bone/bone_<N>.geo`` (36811 verts, 73218 tris, constant
topology, same world frame as the muscle). Each frame we load the bone vertex
positions into the collider's ``wp.Mesh`` and ``refit()`` it before stepping
the solver, so the bones track the muscle and penetration stays shallow / local
— well within ``max_dist`` of the surface.

Run::

    PYTHONPATH=src .venv/bin/python -m muscle.collision.sdf.example.humanV2 \\
        --config data/humanV2/config/humanV2_gmg.json \\
        --bone_dir data/humanV2/model/bone --end_frame 20 \\
        --usd output/humanV2_sdf_coll.usd
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import warp as wp

from muscle.coupleOneWayGmg.solver import OneWaySim, load_config
from muscle.coupleOneWayGmg.example import _vectorized_surface_tris
from muscle.collision.sdf.kernel import solve_mesh_contact_pos_kernel
from muscle.util.geo_io import GeoFile, GeoWriter, read_geo_attr_cached


@wp.kernel
def _count_inside_kernel(pos: wp.array(dtype=wp.vec3),
                         mesh: wp.uint64, max_dist: wp.float32,
                         count: wp.array(dtype=wp.int32)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(mesh, pos[i], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        if wp.length(pos[i] - cp) * q.sign < 0.0:
            wp.atomic_add(count, 0, 1)


@wp.kernel
def _mark_inside_kernel(pos: wp.array(dtype=wp.vec3),
                        mesh: wp.uint64, max_dist: wp.float32,
                        depth: wp.array(dtype=wp.float32)):
    """Per-vertex penetration depth (>0 inside the bone, 0 outside)."""
    i = wp.tid()
    depth[i] = 0.0
    q = wp.mesh_query_point_sign_normal(mesh, pos[i], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        d = wp.length(pos[i] - cp) * q.sign      # < 0 inside
        if d < 0.0:
            depth[i] = -d                         # positive penetration depth


def _set_is_collide_primvar(stage, prim_path, flag01, time):
    """Write a per-vertex int primvar ``is_collide`` (0/1) on a mesh prim.

    Lets usdview select / colour the penetrating muscle verts directly, even
    when they are hidden inside the bone mesh. ``flag01`` is one int per point
    (vertex interpolation); time-sampled to match the mesh's animation.
    """
    from pxr import UsdGeom, Sdf, Vt
    prim = stage.GetPrimAtPath(prim_path)
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("is_collide")
    if not pv:
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "is_collide", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.IntArray.FromNumpy(flag01.astype(np.int32)), time)


def _set_pen_depth_primvar(stage, prim_path, depth, time):
    """Write a per-vertex float primvar ``pen_depth`` (metres) on a mesh prim."""
    from pxr import UsdGeom, Sdf, Vt
    prim = stage.GetPrimAtPath(prim_path)
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("pen_depth")
    if not pv:
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "pen_depth", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.FloatArray.FromNumpy(depth.astype(np.float32)), time)


def _set_sum_is_collide(stage, prim_path, total, time):
    """Write a per-frame scalar custom attr ``sum_is_collide`` on a prim."""
    from pxr import Sdf
    prim = stage.GetPrimAtPath(prim_path)
    attr = prim.GetAttribute("sum_is_collide")
    if not attr:
        attr = prim.CreateAttribute("sum_is_collide", Sdf.ValueTypeNames.Int)
    attr.Set(int(total), time)


def _bake_static_primvars(stage, prim_path, sim):
    """Write the time-INVARIANT per-vertex muscle attributes (materialW fiber
    direction, tendonmask, optional Cd) as default-timecode USD primvars.

    These never change across the sequence, so they are written once at the
    default timecode (not time-sampled) — a few MB total. ``materialW`` lands as
    a Float3Array, ``tendonmask`` as a FloatArray, ``Cd`` as a Color3fArray
    named ``displayColor`` so usdview shades by it. All vertex-interpolated to
    match the muscle mesh points (the full NV vertex set)."""
    from pxr import UsdGeom, Sdf, Vt, Usd
    prim = stage.GetPrimAtPath(prim_path)
    api = UsdGeom.PrimvarsAPI(prim)
    default = Usd.TimeCode.Default()
    # The muscle prim renders ALL particles (sim.NV) — under a fascia merge that
    # is muscle + fascia verts, while these attrs cover only the muscle verts.
    # Vertex primvars must match the point count, so pad the tail (fascia verts,
    # unreferenced by the muscle surface tris) with zeros.
    nv = int(sim.NV)

    def _pad(arr):
        if arr.shape[0] == nv:
            return arr
        out = np.zeros((nv,) + arr.shape[1:], dtype=np.float32)
        out[:arr.shape[0]] = arr
        return out

    def _vec3(name, arr):
        pv = api.GetPrimvar(name) or api.CreatePrimvar(
            name, Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(_pad(arr), np.float32)),
               default)

    def _scalar(name, arr):
        pv = api.GetPrimvar(name) or api.CreatePrimvar(
            name, Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(_pad(arr), np.float32)),
               default)

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
                np.ascontiguousarray(_pad(cd[:, :3]), np.float32)), default)
            baked.append("Cd")
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

    def __init__(self, sim: OneWaySim, bone_dir: str,
                 max_dist: float = 0.2, contact_offset: float = 0.0,
                 max_push: float = 0.005):
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
                  inputs=[self.sim.pos, self.mesh.id,
                          wp.float32(self.max_dist), self._count],
                  device=self.device)
        return int(self._count.numpy()[0])

    def inside_depth(self) -> np.ndarray:
        """Per-muscle-vertex penetration depth (0 outside, >0 inside, metres)."""
        wp.launch(_mark_inside_kernel, dim=self.sim.NV,
                  inputs=[self.sim.pos, self.mesh.id,
                          wp.float32(self.max_dist), self._depth],
                  device=self.device)
        return self._depth.numpy()

    @property
    def tris_flat(self) -> np.ndarray:
        return self.mesh.indices.numpy().astype(np.int32)


def build_humanV2_sim(args):
    """Build the OneWaySim + preload per-frame targets, and compute the
    muscle/fascia surface triangle lists for rendering.

    Shared by the SDF demo (this module) and the UDCF DCD demo
    (``muscle.collision.udcf.example.humanV2``) — both drive the SAME humanV2
    GMG solver and only differ in which collider resolves contact, so the
    config load, preload, and fascia split live here once.
    """
    cfg = load_config(args.config)
    cfg["end_frame"] = args.end_frame
    # Optional overrides for the partial-pin experiment (no-op if args absent).
    if getattr(args, "no_gmg", False):
        cfg["use_multigrid"] = False
    if getattr(args, "tendon_pin_threshold", None) is not None:
        cfg["tendon_pin_threshold"] = args.tendon_pin_threshold
    if getattr(args, "pin_soft_stiffness", None) is not None:
        cfg["pin_soft_stiffness"] = args.pin_soft_stiffness
    if getattr(args, "fiber", "none") != "none":
        cfg["use_fiber"] = True
        cfg["fiber_model"] = args.fiber
        cfg["fiber_stiffness"] = args.fiber_stiffness
        cfg["contraction_ratio"] = args.contraction_ratio
    sim = OneWaySim(cfg, device=args.device)
    if getattr(sim, "use_fiber", False):
        print(f"[fiber] model={sim.fiber_model} k_fib={sim.fiber_stiffness:g} "
              f"contraction_ratio={sim.contraction_ratio:g}")
    # Optional: build the activation mask for a named muscle (e.g. bicep). The
    # run loop ramps this muscle's activation 0->1 across frames so the fiber
    # contraction is visible against the passive baseline.
    sim._activate_mask = None
    act_name = getattr(args, "activate", None)
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

    # When fascia is active (use_fascia) the tet set is muscle + an outer fascia
    # shell. Split by vertex range so each renders as its own USD prim; nvm is
    # the muscle vertex count (fascia verts are appended above it).
    tets = sim.tet_idx.numpy()
    nvm = getattr(sim, "nvm", sim.NV)
    if nvm < sim.NV:
        muscle_tets = tets[(tets < nvm).all(axis=1)]
        fascia_tets = tets[(tets >= nvm).all(axis=1)]
        surf_flat = _vectorized_surface_tris(muscle_tets).reshape(-1).astype(np.int32)
        fascia_flat = _vectorized_surface_tris(fascia_tets).reshape(-1).astype(np.int32)
        print(f"fascia: {muscle_tets.shape[0]} muscle tets, "
              f"{fascia_tets.shape[0]} fascia tets ({sim.NV - nvm} fascia verts)")
    else:
        surf_flat = _vectorized_surface_tris(tets).reshape(-1).astype(np.int32)
        fascia_flat = None
    return sim, cfg, surf_flat, fascia_flat


def run_humanV2_loop(sim, collider, args, cfg, surf_flat, fascia_flat,
                     *, pre_step=None):
    """Per-frame step + USD/geo export, with ``collider`` as the penetration
    MEASUREMENT instrument (bone mesh + inside_depth) regardless of who resolves
    contact.  ``pre_step(frame)`` runs after the bones are positioned but before
    the solve — the UDCF demo uses it to push the new bone pose into its DCD
    collider's static-mesh region.
    """
    renderer = None
    if not args.no_export:
        from muscle.engine import WarpRenderer
        Path(args.usd).parent.mkdir(parents=True, exist_ok=True)
        renderer = WarpRenderer(mode="usd", stage_path=args.usd)
        print(f"[render] writing USD to {args.usd}")
    else:
        print("[no_export] pure-sim timing, no USD/geo output")

    geo_writer = None
    if args.geo_out and not args.no_export:
        tpl = cfg["rest_path"]
        geo_writer = GeoWriter(tpl)
        Path(args.geo_out).mkdir(parents=True, exist_ok=True)
        print(f"[geo] in-place .geo (template {tpl}) -> {args.geo_out}/")

    t0 = time.perf_counter()
    for frame in range(sim.start_frame, sim.end_frame + 1):
        if not args.static_bone:
            collider.set_frame(frame)              # move bones BEFORE solving
        if pre_step is not None:
            pre_step(frame)
        if getattr(sim, "_activate_mask", None) is not None:
            span = max(1, sim.end_frame - sim.start_frame)
            a = (frame - sim.start_frame) / span      # linear ramp 0 -> 1
            sim.set_fiber_activation(a * sim._activate_mask)
        sim.step_frame(frame)
        depth = collider.inside_depth()             # >0 = inside bone
        # Threshold at collide_eps, not >0: after SDF projection verts sit ON
        # the surface (sub-micron inside per the signed test) — that is resolved,
        # not a collision. eps filters that touch-noise out of the count.
        is_collide = (depth > args.collide_eps).astype(np.int32)  # per-vertex 0/1

        if renderer is not None:
            pos_np = sim.positions_np()
            inner = renderer.renderer
            # One USD timecode per sim frame. (begin_frame does round(time*fps);
            # passing frame*sim.dt collapses frames since sim.dt is the SUBSTEP dt.)
            inner.begin_frame(frame / inner.fps)
            muscle_path = inner.render_mesh(name="muscle", points=pos_np,
                                            indices=surf_flat, colors=(0.8, 0.3, 0.25))
            if not getattr(args, "no_collide_primvars", False):
                _set_is_collide_primvar(inner.stage, muscle_path, is_collide, inner.time)
                _set_pen_depth_primvar(inner.stage, muscle_path, depth, inner.time)
            _set_sum_is_collide(inner.stage, muscle_path, int(is_collide.sum()), inner.time)
            if frame == sim.start_frame:
                _bake_static_primvars(inner.stage, muscle_path, sim)
            if fascia_flat is not None:
                inner.render_mesh(name="fascia", points=pos_np,
                                  indices=fascia_flat, colors=(0.55, 0.75, 0.9))
            inner.render_mesh(name="bones", points=collider._cur_verts,
                              indices=collider.tris_flat, colors=(0.9, 0.88, 0.8))
            inner.end_frame()

            if geo_writer is not None:
                geo_writer.write(str(Path(args.geo_out) / f"muscle_{frame}.geo"),
                                 pos_np, is_collide, depth)
        if frame % 5 == 0 or frame == sim.end_frame:
            elapsed = time.perf_counter() - t0
            print(f"  frame {frame:>3d}  {elapsed:6.1f}s  "
                  f"is_collide={int(is_collide.sum())}  "
                  f"max_depth={depth.max()*1000:.1f}mm")
    if renderer is not None:
        renderer.save()
        print(f"USD saved -> {args.usd}")


def add_humanV2_args(ap):
    """Args shared by the SDF and UDCF DCD humanV2 demos."""
    ap.add_argument("--config", default="data/humanV2/config/humanV2_gmg.json")
    ap.add_argument("--bone_dir", default="data/humanV2/model/bone")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--end_frame", type=int, default=20)
    ap.add_argument("--max_dist", type=float, default=0.2)
    ap.add_argument("--no_collision", action="store_true")
    ap.add_argument("--static_bone", action="store_true",
                    help="freeze bones at frame 1 (for A/B vs animated)")
    ap.add_argument("--no_export", action="store_true",
                    help="skip all USD/geo output (measure pure sim time)")
    ap.add_argument("--geo_out", default=None,
                    help="dir for per-frame .geo (in-place P + is_collide). "
                         "Template = cfg rest_path. ~37MB/frame.")
    ap.add_argument("--contact_offset", type=float, default=1e-5,
                    help="SDF push target = surface + this gap (m); holds muscle "
                         "verts a hair proud of the bone instead of exactly on it "
                         "(default 10um). 0 = project onto the surface.")
    ap.add_argument("--collide_eps", type=float, default=1e-6,
                    help="is_collide depth threshold (m); verts shallower than "
                         "this count as resolved (default 1um — filters the "
                         "sub-micron surface-touch noise after SDF projection)")
    ap.add_argument("--fiber", choices=("none", "fibernorm", "aniso_arap"),
                    default="none",
                    help="add an along-fiber constraint using the materialW "
                         "fiber field: 'fibernorm' (linear Houdini TETFIBERNORM) "
                         "or 'aniso_arap' (quadratic HOBAK fiber). Passive unless "
                         "--activate is set. Default none.")
    ap.add_argument("--fiber_stiffness", type=float, default=1.0e8,
                    help="fiber constraint stiffness k_fib (XPBD compliance "
                         "1/(k_fib·V0)); default 1e8.")
    ap.add_argument("--activate", default=None,
                    help="muscle-name substring (e.g. 'bicepsBrachialis') whose "
                         "fiber activation ramps 0->1 across frames so the active "
                         "contraction is visible. Needs --fiber. Matches muscle_id "
                         "strings (both L_/R_).")
    ap.add_argument("--contraction_ratio", type=float, default=0.4,
                    help="max along-fiber shortening at activation=1: target "
                         "stretch s* = 1 - belly·a·ratio (default 0.4 = 40%%).")
    ap.add_argument("--no_collide_primvars", action="store_true",
                    help="skip the per-vertex per-frame is_collide / pen_depth "
                         "USD primvars (~+30%% file size at 200f). The cheap "
                         "per-frame scalar sum_is_collide is still written.")
    return ap


def main(argv=None):
    ap = add_humanV2_args(argparse.ArgumentParser())
    ap.add_argument("--usd", default="output/humanV2_sdf_coll.usd")
    args = ap.parse_args(argv)

    wp.init()
    sim, cfg, surf_flat, fascia_flat = build_humanV2_sim(args)

    # Always build the collider (we need its mesh to MEASURE penetration even
    # when --no_collision disables the push) — only registering it as a solver
    # turns the actual contact resolution on.
    collider = AnimatedBoneMeshCollider(sim, args.bone_dir, max_dist=args.max_dist,
                                        contact_offset=args.contact_offset)
    if not args.static_bone:
        collider.preload(range(sim.start_frame, sim.end_frame + 1))
    if not args.no_collision:
        sim.collision_solvers = [collider]
    print(f"bone collider (SDF): {collider.n_bone_verts} verts, "
          f"{collider.n_bone_tris} tris  "
          f"({'STATIC@f1' if args.static_bone else 'ANIMATED'}, "
          f"push={'OFF' if args.no_collision else 'ON'})")

    run_humanV2_loop(sim, collider, args, cfg, surf_flat, fascia_flat)


if __name__ == "__main__":
    main()
