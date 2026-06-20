"""Bake a trained native-MuJoCo bimanual policy rollout to an animated USD.

For the MyoBimanualArm case (BimanualEnv, tendon-backed 1D muscles) there is
no UsdSkel asset like the humanoid snu_ysy.usd, so this does NOT reuse
tools/bake_motion.py. Instead it follows musclemimic's `_export_usd_custom`
approach: each body mesh geom becomes /root/bone/<name> with static points and
a per-frame world transform, baked from a deterministic policy rollout.

Muscle activations ride along as a time-sampled scalar primvar on /root/muscles:
    primvars:muscle:activation = float[nu]  (per frame, [0, 1])
    muscle:names               = string[nu] (static, actuator order)

Usage:
    uv run python tools/bake_bimanual_usd.py \
        -d data/rl_mass_bimanual/config.json \
        -m output/bimanual_long/nn/max \
        -o output/bimanual_long/bake/bimanual.usdc
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

from rl_mass.finger.bimanual_env import BimanualEnv
from rl_mass.model import MuscleNN, SimulationNN

try:
    import torch
    from torch import Tensor
    _USE_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch always present in this repo
    raise


def rollout(env, model, muscle_model, max_steps, deterministic):
    """Run one episode, recording per-control-step geom transforms + activations.

    Returns:
        geom_xpos   (T+1, ngeom, 3)
        geom_xmat   (T+1, ngeom, 3, 3)
        activations (T+1, nu)   -- activation applied during the step ending at
                                   this frame (frame 0 is the reset pose: zeros)
        qpos        (T+1, nq)
        rewards     list[float]  (length T)
    """
    import mujoco

    state = env.reset(rsi=False)
    mujoco.mj_forward(env.model, env.data)
    num_sim_per_control = env.simulation_hz // env.control_hz

    geom_xpos = [env.data.geom_xpos.copy()]
    geom_xmat = [env.data.geom_xmat.copy().reshape(-1, 3, 3)]
    activations = [np.zeros(env.num_muscles, dtype=np.float32)]
    qpos = [env.data.qpos.copy()]
    rewards = []

    for _ in range(max_steps):
        s = Tensor(state.reshape(1, -1).astype(np.float32))
        if _USE_CUDA:
            s = s.cuda()
        a_dist, _ = model(s)
        action = (a_dist.loc if deterministic else a_dist.sample())
        action = action.cpu().detach().numpy().squeeze()
        env.set_action(action)

        if env.use_muscle and muscle_model is not None:
            mt = Tensor(env.get_muscle_torques().reshape(1, -1).astype(np.float32))
            if _USE_CUDA:
                mt = mt.cuda()
            for _ in range(num_sim_per_control // 2):
                dt = Tensor(
                    env.get_desired_torques().reshape(1, -1).astype(np.float32)
                )
                if _USE_CUDA:
                    dt = dt.cuda()
                act = muscle_model(mt, dt).cpu().detach().numpy().squeeze()
                env.set_activation_levels(act)
                env.step()
                env.step()
        else:
            env.steps_at_once()

        geom_xpos.append(env.data.geom_xpos.copy())
        geom_xmat.append(env.data.geom_xmat.copy().reshape(-1, 3, 3))
        activations.append(env.get_activation_levels().astype(np.float32))
        qpos.append(env.data.qpos.copy())

        if env.is_end_of_episode():
            break
        rewards.append(env.get_reward())
        state = env.get_state()

    return (
        np.stack(geom_xpos).astype(np.float32),
        np.stack(geom_xmat).astype(np.float32),
        np.stack(activations).astype(np.float32),
        np.stack(qpos).astype(np.float32),
        rewards,
    )


def _sanitize(name: str, used: set) -> str:
    san = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not san or san[0].isdigit():
        san = "_" + san
    base, i = san, 1
    while san in used:
        san = f"{base}_{i}"
        i += 1
    used.add(san)
    return san


def export_usd(env, geom_xpos, geom_xmat, activations, out_path, fps, to_yup=True):
    """Author the animated USD (bones + activation primvar). Returns (n_frames, n_geoms)."""
    import mujoco
    from pxr import Gf, Sdf, Usd, UsdGeom

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model = env.model

    MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
    mesh_geoms = [g for g in range(model.ngeom) if int(model.geom_type[g]) == MESH]

    stage = Usd.Stage.CreateNew(out_path)
    n_frames = geom_xpos.shape[0]
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(0, n_frames - 1))
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if to_yup else UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    if to_yup:
        # MuJoCo Z-up -> USD Y-up: rotate the whole rig -90 deg about X.
        root.AddRotateXOp().Set(-90.0)
    UsdGeom.Scope.Define(stage, "/root/bone")

    used: set = set()
    geom_prims = []
    for g in mesh_geoms:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        san = _sanitize(name, used)

        mid = int(model.geom_dataid[g])
        vadr = int(model.mesh_vertadr[mid])
        vnum = int(model.mesh_vertnum[mid])
        fadr = int(model.mesh_faceadr[mid])
        fnum = int(model.mesh_facenum[mid])
        verts = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(model.mesh_face[fadr:fadr + fnum], dtype=np.int32).reshape(-1, 3)

        mesh = UsdGeom.Mesh.Define(stage, f"/root/bone/{san}")
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in verts])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        geom_prims.append((g, mesh.AddTransformOp()))

    for f in range(n_frames):
        gx = geom_xpos[f]
        gm = geom_xmat[f]
        tc = Usd.TimeCode(f)
        for g, xop in geom_prims:
            R, t = gm[g], gx[g]
            # USD row-vector convention (p' = p * M); MuJoCo world = R @ local + t,
            # so the upper-left 3x3 holds R^T and the last row holds t.
            m = Gf.Matrix4d(
                float(R[0, 0]), float(R[1, 0]), float(R[2, 0]), 0.0,
                float(R[0, 1]), float(R[1, 1]), float(R[2, 1]), 0.0,
                float(R[0, 2]), float(R[1, 2]), float(R[2, 2]), 0.0,
                float(t[0]), float(t[1]), float(t[2]), 1.0,
            )
            xop.Set(m, tc)

    # Muscle activations: time-sampled scalar primvar on /root/muscles.
    nu = int(model.nu)
    muscles = UsdGeom.Xform.Define(stage, "/root/muscles")
    pv_api = UsdGeom.PrimvarsAPI(muscles.GetPrim())
    act_pv = pv_api.CreatePrimvar(
        "muscle:activation", Sdf.ValueTypeNames.FloatArray,
        UsdGeom.Tokens.constant,
    )
    for f in range(n_frames):
        act_pv.Set([float(x) for x in activations[f]], Usd.TimeCode(f))

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"muscle{i}"
        for i in range(nu)
    ]
    names_attr = muscles.GetPrim().CreateAttribute(
        "muscle:names", Sdf.ValueTypeNames.StringArray, custom=True
    )
    names_attr.Set(names)

    stage.GetRootLayer().Save()
    return n_frames, len(geom_prims)


def main():
    parser = argparse.ArgumentParser(
        description="Bake native bimanual policy rollout to animated USD (+ activations)"
    )
    parser.add_argument("-d", "--meta", required=True, help="Bimanual config JSON")
    parser.add_argument(
        "-m", "--model", required=True,
        help="Model path prefix (e.g. output/bimanual_long/nn/max)",
    )
    parser.add_argument(
        "-o", "--output", default="output/bimanual_long/bake/bimanual.usdc",
        help="Output USD path",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--no-yup", action="store_true", help="Keep MuJoCo Z-up (default converts to Y-up)"
    )
    args = parser.parse_args()

    env = BimanualEnv(args.meta)
    print(
        f"num states: {env.num_state}, num actions: {env.num_action}, "
        f"num muscles: {env.num_muscles}"
    )

    model = SimulationNN(env.num_state, env.num_action)
    model.load(args.model + ".pt")
    if _USE_CUDA:
        model.cuda()
    model.eval()

    muscle_model = None
    if env.use_muscle:
        muscle_model = MuscleNN(
            env.num_total_related_dofs, env.num_action, env.num_muscles
        )
        muscle_model.load(args.model + "_muscle.pt")
        if _USE_CUDA:
            muscle_model.cuda()
        muscle_model.eval()

    geom_xpos, geom_xmat, activations, qpos, rewards = rollout(
        env, model, muscle_model,
        max_steps=args.max_steps, deterministic=not args.stochastic,
    )
    print(
        f"rollout: {len(rewards)} steps, return={sum(rewards):.3f}, "
        f"frames={geom_xpos.shape[0]}, act_mean={activations.mean():.4f} "
        f"act_max={activations.max():.4f}"
    )

    n_frames, n_geoms = export_usd(
        env, geom_xpos, geom_xmat, activations, args.output,
        fps=env.control_hz, to_yup=not args.no_yup,
    )
    print(f"USD written: {args.output} ({n_frames} frames, {n_geoms} bone meshes, "
          f"{env.num_muscles} muscle activations)")


if __name__ == "__main__":
    main()
