"""U-shape soft-body falling onto a static plane under gravity.

Smoke test for the warp collision pipeline (``muscle.collision.udcf_warp``):
the demo owns every buffer and the HashGrid, and drives the broadphase →
narrow → XPBD apply sequence by directly ``wp.launch``-ing the kernels each
substep — no wrapper classes between this file and the kernels.

* Loads a U-shaped tet mesh from a Houdini ``.geo`` file via ``muscle.geo.Geo``.
* Builds a ``MuscleSim`` from procedural data (no bones, no activation —
  just volume + ARAP to keep tets non-degenerate).
* Drops the mesh under gravity onto a static horizontal plane.
* Each substep, runs broadphase + narrow + XPBD apply over the concatenated
  ``[muscle_tris, plane_tris]`` triangle set.

Run::

    .venv/bin/python -m muscle.collision.example_u_collision \\
        --config data/collisionTest/u_collision.json
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import warp as wp

from muscle.collision.udcf_warp import (
    centroid_radius_kernel,
    max_radius_reduce_kernel,
    emit_pairs_kernel,
    collide_and_solve_kernel,
    collide_and_solve_gs_kernel,
)
from muscle.config import load_config
from muscle.geo import Geo
from muscle.log import setup_logging
from muscle.mesh_io import build_surface_tris
from muscle.muscle_warp import MuscleSim
from muscle.vis_warp import WarpRenderer

log = logging.getLogger(__name__)

DEFAULT_CONFIG = "data/collisionTest/u_collision.json"


# ----------------------------------------------------------------------
# Demo-local glue kernels
# ----------------------------------------------------------------------

@wp.kernel
def _mirror_pos_kernel(src: wp.array(dtype=wp.vec3),
                       dst: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def _apply_collision_dP_kernel(pos: wp.array(dtype=wp.vec3),
                               pprev: wp.array(dtype=wp.vec3),
                               dP: wp.array(dtype=wp.vec3),
                               dPw: wp.array(dtype=wp.float32)):
    """Apply collision dP/dPw to BOTH pos and pprev. Syncing pprev makes
    ``update_velocities`` ``v = (pos - pprev)/dt`` invariant to the
    collision push — otherwise every substep would inject (push)/dt of
    upward velocity, which propagates via ARAP/volume coupling and
    eventually launches the whole body upward."""
    i = wp.tid()
    w = dPw[i]
    if w > 1e-9:
        d = dP[i] / w
        pos[i] = pos[i] + d
        pprev[i] = pprev[i] + d


# ----------------------------------------------------------------------
# Static plane mesh builder (in X-Z plane, fixed Y; gravity is along -Y)
# ----------------------------------------------------------------------

def _build_plane_mesh_xz(y: float, half_extent: float, subdiv: int,
                         center_xz: tuple = (0.0, 0.0)):
    """Subdivided horizontal plane lying in the X-Z plane at ``y``.

    Returns ``(verts (N,3), tris (M,3))``. Winding is chosen so every
    triangle's outward normal (per ``cross(e1,e2)``) points +Y — UDCF
    takes this as the opposing-surface normal when pushing muscle verts
    off the floor, so the sign is critical.
    """
    n = int(max(1, subdiv))
    cx, cz = float(center_xz[0]), float(center_xz[1])
    xs_1d = cx + np.linspace(-half_extent, +half_extent, n + 1)
    zs_1d = cz + np.linspace(-half_extent, +half_extent, n + 1)
    xs, zs = np.meshgrid(xs_1d, zs_1d, indexing="xy")
    verts = np.stack([xs.flatten(),
                      np.full(xs.size, y),
                      zs.flatten()], axis=1).astype(np.float32)
    tris = []
    for j in range(n):
        for i in range(n):
            v00 = j * (n + 1) + i
            v10 = v00 + 1
            v01 = (j + 1) * (n + 1) + i
            v11 = v01 + 1
            tris.append([v00, v11, v10])  # CCW from +Y
            tris.append([v00, v01, v11])
    return verts, np.asarray(tris, dtype=np.int32)


# ----------------------------------------------------------------------
# Mesh loading
# ----------------------------------------------------------------------

def _pick_tip(pos_np: np.ndarray, surface_tris: np.ndarray) -> int:
    """Surface vert with max initial x — that's the open-end tip of one
    U-arm in this dataset (≈ idx 113)."""
    surf = np.unique(surface_tris.reshape(-1))
    return int(surf[np.argmax(pos_np[surf, 0])])


def _scan_max_bary(pos_np: np.ndarray, tets: np.ndarray,
                   vert_side: np.ndarray, tet_side: np.ndarray
                   ) -> tuple[float, int, int]:
    """Exhaustive (vert × opposite-arm tet) bary scan. For each pair,
    compute bary4(vert in tet); ``min(bary4) ≥ 0`` ⇒ vert is inside tet.
    Return the global ``(max min_bary, vert, tet)`` so the worst
    self-intersection (if any) is pinpointed."""
    V0 = pos_np[tets[:, 0]]
    V1 = pos_np[tets[:, 1]]
    V2 = pos_np[tets[:, 2]]
    V3 = pos_np[tets[:, 3]]
    M = np.stack([V1 - V0, V2 - V0, V3 - V0], axis=-1)  # (Nt, 3, 3)
    det = np.linalg.det(M)
    ok_tet = np.abs(det) > 1e-12
    M_inv = np.zeros_like(M)
    M_inv[ok_tet] = np.linalg.inv(M[ok_tet])
    # b123[v, t, i] = M_inv[t, i, k] · (pos[v, k] - V0[t, k])
    term1 = np.einsum('tik,vk->vti', M_inv, pos_np)
    term2 = np.einsum('tik,tk->ti', M_inv, V0)
    b123 = term1 - term2[None, :, :]
    b0 = 1.0 - b123.sum(axis=2)
    min_bary = np.minimum.reduce([b0, b123[..., 0], b123[..., 1], b123[..., 2]])
    pair_mask = (vert_side[:, None] * tet_side[None, :]) < 0
    full_mask = pair_mask & ok_tet[None, :]
    min_bary_m = np.where(full_mask, min_bary, -np.inf)
    best_per_vert = min_bary_m.max(axis=1)
    worst_vert = int(np.argmax(best_per_vert))
    worst_tet = int(np.argmax(min_bary_m[worst_vert]))
    return float(best_per_vert[worst_vert]), worst_vert, worst_tet


def _load_u_mesh(geo_path: str, drop_height: float):
    """Load tet mesh and shift so its lowest vertex sits at ``y = drop_height``
    above the plane at ``y = 0``."""
    g = Geo(geo_path)
    verts = np.asarray(g.positions, dtype=np.float32)
    tets = np.asarray(g.vert, dtype=np.int32)
    if tets.shape[1] != 4:
        raise ValueError(
            f"{geo_path}: expected 4-vert primitives (tets), got "
            f"NVERT_ONE_CONS={g.NVERT_ONE_CONS}")
    cx = 0.5 * (verts[:, 0].min() + verts[:, 0].max())
    cz = 0.5 * (verts[:, 2].min() + verts[:, 2].max())
    verts[:, 0] -= cx
    verts[:, 2] -= cz
    verts[:, 1] += (drop_height - float(verts[:, 1].min()))
    surface_tris = build_surface_tris(tets, verts)
    log.info(f"[u] loaded {len(verts)} verts, {len(tets)} tets, "
             f"{len(surface_tris)} surface tris from {geo_path}")
    return verts, tets, surface_tris


# ----------------------------------------------------------------------
# Collision host
# ----------------------------------------------------------------------

class UCollisionDemo:
    """Owns every collision buffer + the HashGrid, and drives the broadphase
    → narrow → XPBD apply sequence each substep by ``wp.launch``-ing the
    kernels in ``muscle.collision.udcf_warp`` directly. There is no wrapper
    around the kernels — this class IS the orchestrator.

    Registered on the sim:
      * ``collision_solver`` → ``sim.constraint_solvers`` (per substep)
    """

    def __init__(self, sim: MuscleSim, plane_v: np.ndarray, plane_t: np.ndarray,
                 *, max_pairs: int, thickness: float,
                 cell_size: float | None, log_every: int,
                 restitution_n: float = 0.0, restitution_t: float = 1.0,
                 use_self_collision: bool = True, grid_dim: int = 32,
                 coll_iters: int = 4, coll_mode: str = "jacobi"):
        self.sim = sim
        self.log_every = max(1, int(log_every))
        self._frame_counter = 0
        self.restitution_n = float(restitution_n)
        self.restitution_t = float(restitution_t)
        self.thickness = float(thickness)
        self.max_pairs = int(max_pairs)
        self._cell_size_fixed = float(cell_size) if cell_size is not None else None
        # K inner sweeps per substep. Broadphase runs once; narrow+solve+apply
        # re-runs K times against freshly mirrored sim.pos so residual
        # penetration drives the next push. dPw averaging dilutes a single
        # Jacobi sweep when one vert lives in many contact pairs — K sweeps
        # amortize. For GS mode there's no dPw averaging so 1-2 sweeps suffice.
        self.coll_iters = max(1, int(coll_iters))
        if coll_mode not in ("jacobi", "gs"):
            raise ValueError(f"coll_mode must be 'jacobi' or 'gs', got {coll_mode!r}")
        self.coll_mode = coll_mode

        device = str(sim.pos.device)
        self.device = device
        n_mu = int(sim.pos.shape[0])
        n_pl = int(plane_v.shape[0])
        self.n_mu = n_mu
        self.n_pl = n_pl
        n_total = n_mu + n_pl

        muscle_tris = sim.surface_tris.astype(np.int32)
        plane_tris_off = plane_t + n_mu
        all_tris = np.vstack([muscle_tris, plane_tris_off])
        self.n_tris = int(all_tris.shape[0])

        # tri_group: muscle=0, plane=1. Self-collision toggle filters only
        # group-0-vs-group-0 pairs, leaving muscle↔plane untouched.
        tri_group_np = np.zeros(self.n_tris, dtype=np.int32)
        tri_group_np[muscle_tris.shape[0]:] = 1
        self.tri_group = wp.array(tri_group_np, dtype=wp.int32, device=device)
        self.allow_self_group0 = int(bool(use_self_collision))

        self.tris = wp.array2d(all_tris, dtype=wp.int32, device=device)

        global_pos = np.zeros((n_total, 3), dtype=np.float32)
        global_pos[n_mu:] = plane_v
        self.global_pos = wp.array(global_pos, dtype=wp.vec3, device=device)

        muscle_mass = sim.mass.numpy()
        muscle_im = np.where(muscle_mass > 0,
                             1.0 / np.where(muscle_mass > 0, muscle_mass, 1.0),
                             0.0).astype(np.float32)
        if hasattr(sim, "stopped"):
            muscle_im = np.where(sim.stopped.numpy() == 0, muscle_im, 0.0)
        all_im = np.concatenate([muscle_im, np.zeros(n_pl, dtype=np.float32)])
        self.inv_mass = wp.array(all_im, dtype=wp.float32, device=device)

        # Broadphase state
        self.centroids = wp.zeros(self.n_tris, dtype=wp.vec3, device=device)
        self.radii = wp.zeros(self.n_tris, dtype=wp.float32, device=device)
        self._max_radius = wp.zeros(1, dtype=wp.float32, device=device)
        self.pair_a = wp.zeros(self.max_pairs, dtype=wp.int32, device=device)
        self.pair_b = wp.zeros(self.max_pairs, dtype=wp.int32, device=device)
        self.n_pairs = wp.zeros(1, dtype=wp.int32, device=device)
        self.grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device=device)

        # XPBD apply output (per muscle vert; plane verts have inv_mass=0
        # and the apply kernel guards every write with ``if w > 0``).
        self.contact_normal = wp.zeros(n_mu, dtype=wp.vec3, device=device)
        self.is_collide = wp.zeros(n_mu, dtype=wp.int32, device=device)
        self.dP_coll = wp.zeros(n_mu, dtype=wp.vec3, device=device)
        self.dPw_coll = wp.zeros(n_mu, dtype=wp.float32, device=device)

        # Diagnostic 1-elem atomics. UDCF's per-pair (C, gradC, bary, dom)
        # never crosses a buffer boundary in the fused kernel — these
        # 2 scalars are all the host needs for logging.
        self.n_collide = wp.zeros(1, dtype=wp.int32, device=device)
        self.max_C = wp.zeros(1, dtype=wp.float32, device=device)

        log.info(f"[collision] muscle tris={muscle_tris.shape[0]} "
                 f"plane tris={plane_t.shape[0]} total={self.n_tris}")
        log.info(f"[collision] verts: muscle={n_mu} plane={n_pl} total={n_total}")
        log.info(f"[collision] coll_iters={self.coll_iters} coll_mode={self.coll_mode}")
        log.info(f"[collision] use_self_collision={bool(use_self_collision)} "
                 f"(restitution_n={self.restitution_n}, restitution_t={self.restitution_t})")

    def collision_solver(self):
        """Per-substep hook: ``coll_iters`` Jacobi sweeps of (mirror pos →
        broadphase → narrow+solve → apply). Each sweep rebuilds the pair
        list against the post-previous-push positions, so new contacts
        introduced by the push are caught and resolved contacts drop out.
        Push goes into BOTH ``pos`` and ``pprev`` so the position fix is
        invisible to ``update_velocities``.
        """
        sim = self.sim
        device = self.device

        # Outer state — kept across sweeps. ``max_C`` uses atomic_max so it
        # latches the worst depth across all sweeps (initial worst dominates,
        # so it's a per-substep diagnostic of pre-correction penetration);
        # ``contact_normal`` accumulates over sweeps so the velocity-step gets
        # total contact direction, not just the last sweep's residual.
        self.contact_normal.zero_()
        self.is_collide.zero_()
        self.n_collide.zero_()
        self.max_C.zero_()

        for _ in range(self.coll_iters):
            # Refresh global_pos from updated sim.pos and rebuild broadphase
            # against the new geometry (prior sweep moved verts; pair list
            # would otherwise be stale).
            wp.launch(_mirror_pos_kernel, dim=self.n_mu,
                      inputs=[sim.pos, self.global_pos], device=device)
            wp.launch(centroid_radius_kernel, dim=self.n_tris,
                      inputs=[self.tris, self.global_pos, self.centroids, self.radii],
                      device=device)
            if self._cell_size_fixed is None:
                wp.launch(max_radius_reduce_kernel, dim=1,
                          inputs=[self.radii, self._max_radius], device=device)
                cell_size = max(2.0 * float(self._max_radius.numpy()[0]) + self.thickness,
                                1.0e-4)
            else:
                cell_size = self._cell_size_fixed
            self.grid.build(self.centroids, cell_size)
            self.n_pairs.zero_()
            wp.launch(emit_pairs_kernel, dim=self.n_tris,
                      inputs=[self.grid.id, self.centroids, self.radii, self.tris,
                              self.tri_group, wp.int32(self.allow_self_group0),
                              wp.float32(cell_size), wp.float32(self.thickness),
                              wp.int32(self.max_pairs),
                              self.pair_a, self.pair_b, self.n_pairs],
                      device=device)

            if self.coll_mode == "gs":
                # Single-threaded GS: each pair pushes verts directly into
                # global_pos AND sim.pos/pprev, in sequence. No dP buffer.
                wp.launch(collide_and_solve_gs_kernel, dim=1,
                          inputs=[self.pair_a, self.pair_b, self.n_pairs,
                                  self.tris, self.global_pos,
                                  sim.pos, sim.pprev,
                                  self.inv_mass, wp.int32(self.n_mu),
                                  self.contact_normal,
                                  self.is_collide, self.n_collide, self.max_C],
                          device=device)
            else:
                self.dP_coll.zero_()
                self.dPw_coll.zero_()
                wp.launch(collide_and_solve_kernel, dim=self.max_pairs,
                          inputs=[self.pair_a, self.pair_b, self.n_pairs,
                                  self.tris, self.global_pos, self.inv_mass,
                                  self.dP_coll, self.dPw_coll, self.contact_normal,
                                  self.is_collide, self.n_collide, self.max_C],
                          device=device)
                wp.launch(_apply_collision_dP_kernel, dim=self.n_mu,
                          inputs=[sim.pos, sim.pprev, self.dP_coll, self.dPw_coll],
                          device=device)

    def log_diag(self, step: int):
        self._frame_counter += 1
        if self._frame_counter % self.log_every != 0:
            return
        n = int(self.n_pairs.numpy()[0])
        if n > self.max_pairs:
            raise RuntimeError(
                f"emitted {n} pairs > max_pairs={self.max_pairs}; "
                f"increase max_pairs (atomic counter saturated)")
        n_collide = int(self.n_collide.numpy()[0])
        max_C = float(self.max_C.numpy()[0])
        log.info(f"[collision] step={step:4d} pairs={n:5d} "
                 f"collide={n_collide:5d} max_C={max_C:.5f}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="U-shape gravity collision demo.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.nsteps = args.steps
    if args.output is not None:
        cfg.output_path = args.output

    wp.init()
    wp.set_device(cfg.arch)

    coll_cfg = getattr(cfg, "collision", {}) or {}
    plane_cfg = coll_cfg["plane"]

    verts, tets, _ = _load_u_mesh(cfg.geo_path,
                                  drop_height=float(plane_cfg["drop_height"]))

    fiber_zero = np.zeros((tets.shape[0], 3), dtype=np.float32)
    fiber_zero[:, 0] = 1.0  # unused (no fiber/active)

    gravity_y = -abs(float(getattr(cfg, "gravity_z",
                                    getattr(cfg, "gravity", 9.8))))
    sim = MuscleSim.from_procedural(
        verts, tets, fiber_zero,
        constraint_configs=list(cfg.constraints),
        dts=float(cfg.dt),
        num_substeps=int(cfg.num_substeps),
        device=cfg.arch,
        density=float(cfg.density),
        veldamping=float(cfg.veldamping),
        gravity=gravity_y,
    )
    log.info(f"sim: n_verts={sim.n_verts} n_tets={sim.tet_np.shape[0]} "
             f"n_surface_tris={sim.surface_tris.shape[0]}")

    plane_v, plane_t = _build_plane_mesh_xz(
        y=0.0,
        half_extent=float(plane_cfg["half_extent"]),
        subdiv=int(plane_cfg["subdiv"]),
    )

    use_collision = bool(coll_cfg.get("use_collision", True))
    use_self_collision = bool(coll_cfg.get("use_self_collision", True))

    demo = UCollisionDemo(
        sim, plane_v, plane_t,
        max_pairs=int(coll_cfg.get("max_pairs", 65536)),
        thickness=float(coll_cfg.get("thickness", 0.001)),
        cell_size=coll_cfg.get("cell_size"),
        log_every=int(coll_cfg.get("log_every", 10)),
        restitution_n=float(coll_cfg.get("restitution_n", 0.0)),
        restitution_t=float(coll_cfg.get("restitution_t", 1.0)),
        use_self_collision=use_self_collision,
        coll_iters=int(coll_cfg.get("coll_iters", 4)),
        coll_mode=str(coll_cfg.get("coll_mode", "jacobi")),
    )
    # When use_collision=False we skip registration entirely: no broadphase,
    # no narrow phase, no XPBD apply. The body free-falls past the plane.
    if use_collision:
        sim.constraint_solvers = [demo.collision_solver]
        # Wire collision velocity handling directly into sim.update_velocities:
        # is_collide[i]==1 gates the per-vert reflection; contact_normal gives
        # the direction, restitution_{n,t} the energy loss.
        sim.is_collide = demo.is_collide
        sim.contact_normal = demo.contact_normal
        sim.restitution_n = demo.restitution_n
        sim.restitution_t = demo.restitution_t
    log.info(f"[collision] use_collision={use_collision}")

    renderer = None
    if cfg.render_mode == "usd":
        os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
        renderer = WarpRenderer(mode="usd", stage_path=cfg.output_path)
        log.info(f"[render] writing USD to {cfg.output_path}")

    # Muscle (red) and plane (gray) are two separate render meshes for clarity.
    muscle_t_flat = sim.surface_tris_flat
    plane_t_flat = plane_t.reshape(-1).astype(np.int32)

    render_stride = max(1, int(round((1.0 / cfg.render_fps) / cfg.dt)))
    log.info(f"render_stride={render_stride} (dt={cfg.dt}, fps={cfg.render_fps})")

    rest_pos = sim.pos.numpy().copy()
    # Per-vert / per-tet arm label from rest x: +1 right arm, -1 left arm,
    # 0 = bend (|x|<0.1, excluded from both sides). Search restricted to
    # opposite-arm tets so "self-skin contact" doesn't show as penetration.
    vert_side = np.sign(rest_pos[:, 0]).astype(np.int32)
    tet_cx = rest_pos[sim.tet_np].mean(axis=1)[:, 0]
    tet_side = np.sign(tet_cx).astype(np.int32)
    log.info(f"[diag] verts L/R={int((vert_side<0).sum())}/{int((vert_side>0).sum())} "
             f"tets L/R={int((tet_side<0).sum())}/{int((tet_side>0).sum())} "
             f"(sign(rest_x), no bend threshold)")

    for step in range(1, cfg.nsteps + 1):
        sim.step_cnt = step
        sim.step()
        demo.log_diag(step)

        if step % render_stride == 0 or step == 1:
            mu_pos = sim.pos.numpy()
            mb, vw, tw = _scan_max_bary(mu_pos, sim.tet_np, vert_side, tet_side)
            tag = "PEN!" if mb > 0.0 else "ok  "
            log.info(f"step={step:4d} t={step*cfg.dt:6.3f}s "
                     f"min_y={mu_pos[:,1].min():.4f} "
                     f"[{tag} max_bary={mb:+.4f} @vert{vw} tet{tw}]")
            # Per-vert push diag on the current worst (vw, tw): is collision
            # actually pushing it toward "out of tet tw"? Read last-substep's
            # accumulated contact_normal (sum of gradC over K sweeps) and
            # last-sweep's dP_coll / dPw_coll.
            if mb > 0.0 and step % (render_stride * 5) == 0:
                cn = demo.contact_normal.numpy()[vw]
                dpw = float(demo.dPw_coll.numpy()[vw])
                dp = demo.dP_coll.numpy()[vw]
                tet_verts = sim.tet_np[tw]
                tet_centroid = mu_pos[tet_verts].mean(axis=0)
                out_vec = mu_pos[vw] - tet_centroid
                out_n = np.linalg.norm(out_vec)
                cn_n = np.linalg.norm(cn)
                if out_n > 1e-9 and cn_n > 1e-9:
                    dot_out = float(np.dot(cn / cn_n, out_vec / out_n))
                else:
                    dot_out = 0.0
                applied = dp / dpw if dpw > 1e-9 else np.zeros(3)
                log.info(f"  [push@v{vw}] |cn|={cn_n:.4f} dot(cn,out)={dot_out:+.3f} "
                         f"dPw_last={dpw:.1f} |dp_last/dPw|={np.linalg.norm(applied)*1000:.3f}mm")
            if renderer is not None:
                inner = renderer.renderer
                inner.begin_frame(step * cfg.dt)
                inner.render_mesh(name="muscle", points=mu_pos,
                                  indices=muscle_t_flat, colors=(0.8, 0.3, 0.2))
                inner.render_mesh(name="plane", points=plane_v,
                                  indices=plane_t_flat, colors=(0.55, 0.55, 0.55))
                inner.end_frame()

    if renderer is not None:
        renderer.save()
        log.info(f"Done — USD saved to {cfg.output_path}")


if __name__ == "__main__":
    main()
