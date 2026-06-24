"""ARAP-SVD inversion-recovery example.

Isolation test: can pure ARAP-SVD recover a tet mesh from inversion? Loads a
tetgen mesh (bunny), applies ONLY the engine `tetarapsvd` constraint (same
signed-SVD kernel humanV2 uses), deforms it by scaling along y about the y=0
plane, then runs quasi-static Jacobi ARAP projection and reports whether the
inverted-tet count returns to zero. One animated USD is written per y-scale.

y-scale `s` (deform = pos.y *= s):
    s>0  compressed, non-inverted (det F>0)   -> should recover
    s=0  degenerate flat plane (det F=0)       -> ill-posed (sign-ambiguous)
    s<0  inverted (det F<0); s=-1 = pure reflection (every tet det F=-1)

All options live in the JSON config; the only CLI arg is --config.

Usage:
    PYTHONPATH=src python -m example.inversion_test \
        --config data/inversion_test/config/inversion_test.json
"""

import argparse
import json
import os

import numpy as np
import warp as wp

from muscle.constraint.solve_kernels import solve_tetarap_svd_kernel
from muscle.engine import (
    precompute_rest_kernel,
    count_inverted_tets_kernel,
    clear_dP_kernel,
    apply_dP_kernel,
)

DEFAULT_CONFIG = "data/inversion_test/config/inversion_test.json"


def load_tetgen(model_dir, stem):
    """Read tetgen .node (verts) + .ele (tets). 0-based indices."""
    with open(os.path.join(model_dir, stem + ".node")) as f:
        n = int(f.readline().split()[0])
        pos = np.zeros((n, 3), np.float32)
        for _ in range(n):
            t = f.readline().split()
            pos[int(t[0])] = [float(t[1]), float(t[2]), float(t[3])]
    with open(os.path.join(model_dir, stem + ".ele")) as f:
        m = int(f.readline().split()[0])
        tets = np.zeros((m, 4), np.int32)
        for _ in range(m):
            t = f.readline().split()
            tets[int(t[0])] = [int(t[1]), int(t[2]), int(t[3]), int(t[4])]
    return pos, tets


def load_faces(model_dir, stem):
    """Read tetgen .face surface triangles for USD export."""
    with open(os.path.join(model_dir, stem + ".face")) as f:
        k = int(f.readline().split()[0])
        faces = np.zeros((k, 3), np.int32)
        for j in range(k):
            t = f.readline().split()
            faces[j] = [int(t[1]), int(t[2]), int(t[3])]
    return faces


def signed_detF(pos, tets, B):
    """det(F) per tet (F = Ds * B, V2 convention pt3 = reference); sign = inversion."""
    c0 = pos[tets[:, 0]] - pos[tets[:, 3]]
    c1 = pos[tets[:, 1]] - pos[tets[:, 3]]
    c2 = pos[tets[:, 2]] - pos[tets[:, 3]]
    Ds = np.stack([c0, c1, c2], axis=2)  # (T,3,3) columns
    return np.linalg.det(np.matmul(Ds, B))


def write_usd(path, faces, frames, fps=24):
    """Static topology, animated points. frames = list of (N,3) positions."""
    from pxr import Usd, UsdGeom, Vt
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetFramesPerSecond(fps)
    stage.SetTimeCodesPerSecond(fps)
    mesh = UsdGeom.Mesh.Define(stage, "/bunny")
    mesh.CreateFaceVertexCountsAttr([3] * faces.shape[0])
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).astype(np.int32).tolist())
    pts = mesh.CreatePointsAttr()
    for fi, pos in enumerate(frames):
        pts.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pos, dtype=np.float32)),
                Usd.TimeCode(fi))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)
    stage.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser(description="ARAP-SVD inversion-recovery example.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to JSON config (all options live there).")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    dev = cfg["device"]
    stiff = float(cfg["stiffness"])
    dt = float(cfg["dt"])
    cap_every = int(cfg["cap_every"])
    log_every = int(cfg["log_every"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    wp.init()
    pos0_np, tets_np = load_tetgen(cfg["model_dir"], cfg["mesh_stem"])
    faces_np = load_faces(cfg["model_dir"], cfg["mesh_stem"])
    N, T = pos0_np.shape[0], tets_np.shape[0]
    print(f"{cfg['mesh_stem']}: NV={N} NT={T}  "
          f"y-range [{pos0_np[:,1].min():.3f},{pos0_np[:,1].max():.3f}]")

    pos0 = wp.array(pos0_np, dtype=wp.vec3, device=dev)
    tet = wp.array(tets_np, dtype=wp.vec4i, device=dev)
    rest_matrix = wp.zeros(T, dtype=wp.mat33, device=dev)
    rest_volume = wp.zeros(T, dtype=wp.float32, device=dev)
    mass = wp.zeros(N, dtype=wp.float32, device=dev)
    tot = wp.zeros(1, dtype=wp.float32, device=dev)
    wp.launch(precompute_rest_kernel, dim=T, device=dev,
              inputs=[pos0, tet, rest_volume, rest_matrix, mass, 1000.0, tot])
    B_np = rest_matrix.numpy()

    # Uniform mass = isolate ARAP geometry (no tiny-tet inv-mass blowup).
    mass = wp.array(np.ones(N, np.float32), dtype=wp.float32, device=dev)
    stopped = wp.zeros(N, dtype=wp.int32, device=dev)  # no pins
    c_stiff = wp.array(np.full(T, stiff, np.float32), dtype=wp.float32, device=dev)
    c_tetid = wp.array(np.arange(T, dtype=np.int32), dtype=wp.int32, device=dev)
    c_L = wp.zeros(T, dtype=wp.vec3, device=dev)
    dP = wp.zeros(N, dtype=wp.vec3, device=dev)
    dPw = wp.zeros(N, dtype=wp.float32, device=dev)

    def count_inv(pos_wp):
        cnt = wp.zeros(1, dtype=wp.int32, device=dev)
        wp.launch(count_inverted_tets_kernel, dim=T, device=dev,
                  inputs=[tet, pos_wp, rest_matrix, cnt])
        return int(cnt.numpy()[0])

    def run(s, nframes):
        iters = (nframes - 1) * cap_every
        pos_np = pos0_np.copy()
        pos_np[:, 1] = pos_np[:, 1] * s  # flatten/invert about y=0
        pos = wp.array(pos_np, dtype=wp.vec3, device=dev)
        pprev = wp.clone(pos)
        c_L.zero_()
        inv0 = count_inv(pos)
        d0 = signed_detF(pos_np, tets_np, B_np)
        traj = [(0, inv0)]
        frames = [pos_np.copy()]  # frame 0 = deformed start
        for it in range(1, iters + 1):
            wp.launch(clear_dP_kernel, dim=N, device=dev, inputs=[dP, dPw])
            wp.launch(solve_tetarap_svd_kernel, dim=T, device=dev,
                      inputs=[c_stiff, tet, c_tetid, c_L, pos, pprev,
                              dP, dPw, mass, stopped, rest_matrix, dt, 1, 0])
            wp.launch(apply_dP_kernel, dim=N, device=dev, inputs=[pos, dP, dPw])
            if it % cap_every == 0:
                frames.append(pos.numpy())
            if it % log_every == 0:
                traj.append((it, count_inv(pos)))
        dF = signed_detF(pos.numpy(), tets_np, B_np)
        usd_path = os.path.join(out_dir, f"bunny_s{s:+.2f}.usd")
        write_usd(usd_path, faces_np, frames)
        print(f"  s={s:+.2f}: inv {inv0:5d} -> {count_inv(pos):5d} | "
              f"detF init[min {d0.min():+.3f} mean {d0.mean():+.3f}] "
              f"final[min {dF.min():+.3f} mean {dF.mean():+.3f}] "
              f"| {len(frames)}f -> {os.path.basename(usd_path)}")

    write_usd(os.path.join(out_dir, "bunny_rest.usd"), faces_np, [pos0_np])
    print(f"\n=== pure ARAP-SVD recovery, Jacobi, stiff={stiff:.0e}, "
          f"{cap_every} iters/frame ===")
    for entry in cfg["sweep"]:
        run(float(entry["y_scale"]), int(entry["nframes"]))
    print(f"\nUSD per y-scale (frame0=deformed, then every {cap_every} iters) -> {out_dir}")


if __name__ == "__main__":
    main()
