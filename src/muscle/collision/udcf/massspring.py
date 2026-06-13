"""Faithful Warp port of Gaia_UDCF's MassSpring tet material + PBD substep loop.

Reference: ~/Dev/Gaia_UDCF/Simulator/Modules/PBD/
  PBDTetMeshMassSpring::solveSpringConstraintCPU()   :115-188
  PBDTetMeshMassSpring::solveVolumeConstraintCPU()   :192-285
  PBDTetMeshMassSpring::InversionSolveCPU()          :482-548
  PBDTetMeshFEM::applyInitialGuess()                 (xprev=x; v+=g*dt; x+=v*dt)
  PBDPhysics::updateVelocities()                     :3559-3595
  TetData mass / DmInv / restVolume init             :413-435

This is the missing material layer behind the S01 U-drop demo.  The earlier
demo drove the U with a volume+ARAP material (soft, lets the arms fold into
deep self-crossings that the faithful DCD then reports as huge plane
distances -> velocity injection -> explosion).  Gaia's actual S01 material is
MassSpring: stiff edge springs + a hard det(F)=1 volume constraint, which
keeps the mesh from crossing deeply so the same DCD stays bounded.

Substep order (Gaia runStepCPU, iterations=1, collisionDetectionSubSteps=1):
  integrate -> spring -> volume -> boundary -> inversion -> narrowPhase -> updateVel
Boundary + narrowPhase are NOT here (udcf_dcd owns both); this
module owns integrate, spring, volume, inversion, updateVelocities.

Gauss-Seidel parity: Gaia's CPU path sweeps edges/tets serially.  We sweep
per graph-coloring group (greedy vertex coloring): within a group no two
edges/tets share a vertex, so colored-parallel Jacobi == serial GS for that
group; groups run sequentially.  Result is physically equivalent (the spring
SET and rest lengths are identical; only intra-group order differs, which is
irrelevant when groups are conflict-free).

Gaia S01 params (Models.json / Parameters.json, both JSON-default where the
model file omits them): springCompliance=1e-4, springDamping=0.01,
volume hard (no compliance), density=3, gravity=(0,-9.8,0),
dt_sub=(1/60)/20, inversionSolveConstraintMultiplier=0.02,
exponentialVelDamping=0.95**dt_sub, constantVelDamping=0.02*dt_sub.

All arithmetic fp32.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
# Topology helpers (host)
# ---------------------------------------------------------------------------
def build_unique_edges(tets: np.ndarray) -> np.ndarray:
    """All unique tet edges (the set MeshFrame's getEdges() yields).  Ordering
    differs from Gaia's MeshFrame iteration but the SET is identical, and the
    spring physics is order-independent up to GS sweep order."""
    es = set()
    for t in tets:
        for a in range(4):
            for b in range(a + 1, 4):
                i, j = int(t[a]), int(t[b])
                es.add((i, j) if i < j else (j, i))
    return np.asarray(sorted(es), dtype=np.int32)


def greedy_color(items: np.ndarray, n_verts: int) -> list[np.ndarray]:
    """Greedy graph coloring: partition item rows (edges=2 cols, tets=4 cols)
    so no two items in a group share a vertex.  Returns list of int32 index
    arrays, one per color."""
    n = items.shape[0]
    color = np.full(n, -1, dtype=np.int32)
    # vert -> set of colors already used by an incident item
    used = [set() for _ in range(n_verts)]
    for idx in range(n):
        vs = items[idx]
        c = 0
        while any(c in used[v] for v in vs):
            c += 1
        color[idx] = c
        for v in vs:
            used[v].add(c)
    groups = []
    for c in range(color.max() + 1):
        groups.append(np.where(color == c)[0].astype(np.int32))
    return groups


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@wp.kernel
def integrate_kernel(pos: wp.array(dtype=wp.vec3),
                     pprev: wp.array(dtype=wp.vec3),
                     vel: wp.array(dtype=wp.vec3),
                     inv_mass: wp.array(dtype=wp.float32),
                     gravity: wp.vec3, dt: wp.float32):
    """Gaia applyInitialGuess: xprev=x; v+=g*dt; x+=v*dt.  Fixed verts
    (inv_mass==0) keep position (still record xprev so velocity reads 0)."""
    i = wp.tid()
    pprev[i] = pos[i]
    if inv_mass[i] == 0.0:
        return
    vel[i] = vel[i] + gravity * dt
    pos[i] = pos[i] + vel[i] * dt


@wp.kernel
def spring_kernel(pos: wp.array(dtype=wp.vec3),
                  pprev: wp.array(dtype=wp.vec3),
                  inv_mass: wp.array(dtype=wp.float32),
                  edges: wp.array(dtype=wp.vec2i),
                  rest_len: wp.array(dtype=wp.float32),
                  lam: wp.array(dtype=wp.float32),
                  group: wp.array(dtype=wp.int32),
                  alpha: wp.float32, spring_damping: wp.float32,
                  dt: wp.float32):
    """XPBD distance constraint, one group (conflict-free) per launch.
    Gaia solveSpringConstraintCPU:117-188."""
    g = wp.tid()
    e = group[g]
    v1 = edges[e][0]
    v2 = edges[e][1]
    diff = pos[v1] - pos[v2]
    l = wp.length(diff)
    if l == 0.0:
        return
    x1grad = diff * (1.0 / l)
    x2grad = -x1grad
    C = l - rest_len[e]
    w = inv_mass[v1] + inv_mass[v2]
    if spring_damping != 0.0:
        vel1 = pos[v1] - pprev[v1]
        vel2 = pos[v2] - pprev[v2]
        damp = wp.dot(x1grad, vel1) + wp.dot(x2grad, vel2)
        C = C + damp * (spring_damping / dt)
        w = w * (1.0 + spring_damping / dt)
    denom = w + alpha
    if denom == 0.0:
        return
    dlam = (-C - alpha * lam[e]) / denom
    lam[e] = lam[e] + dlam
    pos[v1] = pos[v1] + x1grad * (dlam * inv_mass[v1])
    pos[v2] = pos[v2] + x2grad * (dlam * inv_mass[v2])


@wp.kernel
def volume_kernel(pos: wp.array(dtype=wp.vec3),
                  inv_mass: wp.array(dtype=wp.float32),
                  tets: wp.array(dtype=wp.vec4i),
                  dm_inv: wp.array(dtype=wp.mat33),
                  group: wp.array(dtype=wp.int32)):
    """Hard PBD volume preservation, C=det(F)-1, dlambda=-C/w (no compliance).
    Gaia solveVolumeConstraintCPU:192-285."""
    g = wp.tid()
    t = group[g]
    i0 = tets[t][0]
    i1 = tets[t][1]
    i2 = tets[t][2]
    i3 = tets[t][3]
    v0 = pos[i0]
    P = wp.matrix_from_cols(pos[i1] - v0, pos[i2] - v0, pos[i3] - v0)
    iDS = dm_inv[t]
    F = P * iDS
    # column accessors
    f0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    dF0 = wp.cross(f1, f2)
    dF1 = wp.cross(f2, f0)
    dF2 = wp.cross(f0, f1)
    g1 = dF0 * iDS[0, 0] + dF1 * iDS[0, 1] + dF2 * iDS[0, 2]
    g2 = dF0 * iDS[1, 0] + dF1 * iDS[1, 1] + dF2 * iDS[1, 2]
    g3 = dF0 * iDS[2, 0] + dF1 * iDS[2, 1] + dF2 * iDS[2, 2]
    g0 = -g1 - g2 - g3
    vol = wp.determinant(F)
    C = vol - 1.0
    if C == 0.0:
        return
    w = (wp.dot(g0, g0) * inv_mass[i0] + wp.dot(g1, g1) * inv_mass[i1]
         + wp.dot(g2, g2) * inv_mass[i2] + wp.dot(g3, g3) * inv_mass[i3])
    if w == 0.0:
        return
    dlam = -C / w
    pos[i0] = pos[i0] + g0 * (dlam * inv_mass[i0])
    pos[i1] = pos[i1] + g1 * (dlam * inv_mass[i1])
    pos[i2] = pos[i2] + g2 * (dlam * inv_mass[i2])
    pos[i3] = pos[i3] + g3 * (dlam * inv_mass[i3])


@wp.kernel
def inversion_kernel(pos: wp.array(dtype=wp.vec3),
                     inv_mass: wp.array(dtype=wp.float32),
                     tets: wp.array(dtype=wp.vec4i),
                     rest_vol: wp.array(dtype=wp.float32),
                     group: wp.array(dtype=wp.int32),
                     multiplier: wp.float32):
    """Untangle inverted tets (signed vol<=0).  Gaia InversionSolveCPU:482-548.
    (Gaia's write uses local index `i`; we use the global vId — the local form
    is a reference bug that corrupts verts 0-3.)"""
    g = wp.tid()
    t = group[g]
    i0 = tets[t][0]
    i1 = tets[t][1]
    i2 = tets[t][2]
    i3 = tets[t][3]
    e1 = pos[i1] - pos[i0]
    e2 = pos[i2] - pos[i0]
    e3 = pos[i3] - pos[i0]
    vol = wp.dot(e1, wp.cross(e2, e3)) / 6.0
    if vol > 0.0:
        return
    gr0 = wp.cross(e3 - e1, e2 - e1)
    gr1 = wp.cross(e2, e3)
    gr2 = wp.cross(e3, e1)
    gr3 = wp.cross(e1, e2)
    C = 6.0 * (vol - rest_vol[t] * multiplier)
    if C == 0.0:
        return
    a0 = gr0 * inv_mass[i0]
    a1 = gr1 * inv_mass[i1]
    a2 = gr2 * inv_mass[i2]
    a3 = gr3 * inv_mass[i3]
    w = wp.dot(a0, a0) + wp.dot(a1, a1) + wp.dot(a2, a2) + wp.dot(a3, a3)
    if w == 0.0:
        return
    dlam = -C / w
    pos[i0] = pos[i0] + gr0 * (dlam * inv_mass[i0])
    pos[i1] = pos[i1] + gr1 * (dlam * inv_mass[i1])
    pos[i2] = pos[i2] + gr2 * (dlam * inv_mass[i2])
    pos[i3] = pos[i3] + gr3 * (dlam * inv_mass[i3])


@wp.kernel
def update_vel_kernel(pos: wp.array(dtype=wp.vec3),
                      pprev: wp.array(dtype=wp.vec3),
                      vel: wp.array(dtype=wp.vec3),
                      dt: wp.float32, exp_damp: wp.float32,
                      const_damp: wp.float32):
    """Gaia updateVelocities:3559-3595.  v=(x-xprev)/dt then
    vMagNew = vMag*exp_damp - const_damp."""
    i = wp.tid()
    v = (pos[i] - pprev[i]) * (1.0 / dt)
    vmag = wp.length(v)
    if vmag > 1.0e-6:
        vnew = vmag * exp_damp - const_damp
        if vnew < 1.0e-6:
            vnew = 0.0
        vel[i] = v * (vnew / vmag)
    else:
        vel[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _zero_kernel(a: wp.array(dtype=wp.float32)):
    a[wp.tid()] = 0.0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class MassSpringParams:
    spring_compliance: float = 1e-4
    spring_damping: float = 0.01
    inversion_multiplier: float = 0.02
    density: float = 3.0
    gravity: float = -9.8
    # velocity damping (per-substep, pre-raised by dt at init)
    exp_vel_damping: float = 0.95       # raised to dt_sub internally
    const_vel_damping: float = 0.02     # multiplied by dt_sub internally


class MassSpringTet:
    """Gaia MassSpring tet material over a Warp position buffer.  Owns
    integrate / spring / volume / inversion / updateVelocities; the caller
    interleaves boundary + DCD between solve_material() and update_velocities()
    to reproduce Gaia's substep order."""

    def __init__(self, verts: np.ndarray, tets: np.ndarray,
                 dt_sub: float, params: MassSpringParams = MassSpringParams(),
                 device: str = "cuda:0"):
        self.device = device
        self.dt = float(dt_sub)
        self.p = params
        verts = np.asarray(verts, np.float32)
        tets = np.asarray(tets, np.int32).copy()
        self.n_verts = verts.shape[0]
        self.n_tets = tets.shape[0]

        # Gaia's convention is positive signed rest volume (det(Dm) > 0); the
        # U.geo tets wind the other way (all det < 0 -> negative mass ->
        # frozen verts, and the inversion check would fire on every rest tet).
        # Flip orientation by swapping two vertices so det(Dm) > 0.
        v = verts
        Dm0 = np.stack([v[tets[:, 1]] - v[tets[:, 0]],
                        v[tets[:, 2]] - v[tets[:, 0]],
                        v[tets[:, 3]] - v[tets[:, 0]]], axis=2)
        if np.median(np.linalg.det(Dm0)) < 0:
            tets[:, [2, 3]] = tets[:, [3, 2]]

        # --- rest geometry: Dm, DmInv, restVol, mass (Gaia TetData:413-435)
        Dm = np.stack([v[tets[:, 1]] - v[tets[:, 0]],
                       v[tets[:, 2]] - v[tets[:, 0]],
                       v[tets[:, 3]] - v[tets[:, 0]]], axis=2)  # (T,3,3) cols
        DmInv = np.linalg.inv(Dm)
        rest_vol = np.linalg.det(Dm) / 6.0
        mass = np.zeros(self.n_verts, np.float64)
        for k in range(4):
            np.add.at(mass, tets[:, k], 0.25 * rest_vol * params.density)
        inv_mass = np.where(mass > 0, 1.0 / np.where(mass > 0, mass, 1.0), 0.0)

        # --- edges + rest lengths
        edges = build_unique_edges(tets)
        el = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        self.n_edges = edges.shape[0]

        # --- colorings (conflict-free groups)
        edge_groups = greedy_color(edges, self.n_verts)
        tet_groups = greedy_color(tets, self.n_verts)

        dev = device
        self.pos = wp.array(v, dtype=wp.vec3, device=dev)
        self.pprev = wp.array(v, dtype=wp.vec3, device=dev)
        self.vel = wp.zeros(self.n_verts, dtype=wp.vec3, device=dev)
        self.inv_mass = wp.array(inv_mass.astype(np.float32),
                                 dtype=wp.float32, device=dev)
        self.edges = wp.array(edges, dtype=wp.vec2i, device=dev)
        self.rest_len = wp.array(el.astype(np.float32), dtype=wp.float32,
                                 device=dev)
        self.lam = wp.zeros(self.n_edges, dtype=wp.float32, device=dev)
        self.tets = wp.array(tets, dtype=wp.vec4i, device=dev)
        self.dm_inv = wp.array(DmInv.astype(np.float32), dtype=wp.mat33,
                               device=dev)
        self.rest_vol = wp.array(rest_vol.astype(np.float32),
                                 dtype=wp.float32, device=dev)
        self.edge_groups = [wp.array(gr, dtype=wp.int32, device=dev)
                            for gr in edge_groups]
        self.tet_groups = [wp.array(gr, dtype=wp.int32, device=dev)
                           for gr in tet_groups]

        self.gravity = wp.vec3(0.0, params.gravity, 0.0)
        self.alpha = params.spring_compliance / (self.dt * self.dt)
        self.exp_damp = float(params.exp_vel_damping ** self.dt)
        self.const_damp = float(params.const_vel_damping * self.dt)

    # -- substep pieces ----------------------------------------------------
    def integrate(self):
        wp.launch(integrate_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pprev, self.vel, self.inv_mass,
                          self.gravity, wp.float32(self.dt)],
                  device=self.device)
        wp.launch(_zero_kernel, dim=self.n_edges, inputs=[self.lam],
                  device=self.device)

    def solve_spring(self):
        for gr in self.edge_groups:
            wp.launch(spring_kernel, dim=gr.shape[0],
                      inputs=[self.pos, self.pprev, self.inv_mass, self.edges,
                              self.rest_len, self.lam, gr,
                              wp.float32(self.alpha),
                              wp.float32(self.p.spring_damping),
                              wp.float32(self.dt)],
                      device=self.device)

    def solve_volume(self):
        for gr in self.tet_groups:
            wp.launch(volume_kernel, dim=gr.shape[0],
                      inputs=[self.pos, self.inv_mass, self.tets,
                              self.dm_inv, gr],
                      device=self.device)

    def solve_inversion(self):
        for gr in self.tet_groups:
            wp.launch(inversion_kernel, dim=gr.shape[0],
                      inputs=[self.pos, self.inv_mass, self.tets,
                              self.rest_vol, gr,
                              wp.float32(self.p.inversion_multiplier)],
                      device=self.device)

    def solve_material(self):
        """spring -> volume (Gaia projectConstraintsCPU)."""
        self.solve_spring()
        self.solve_volume()

    def update_velocities(self):
        wp.launch(update_vel_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pprev, self.vel, wp.float32(self.dt),
                          wp.float32(self.exp_damp),
                          wp.float32(self.const_damp)],
                  device=self.device)
