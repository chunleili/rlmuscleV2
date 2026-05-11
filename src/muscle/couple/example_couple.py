"""Couple Warp MuscleSim with Newton rigid-body skeleton.

Default config (bicep_coupled.json) uses TETFIBERNORM constraint.
Switch to bicep_fibermillard_coupled.json via --config to use TETFIBERMILLARD.

Usage:
    uv run python -m muscle.example_couple --auto --steps 300
    uv run python -m muscle.example_couple --auto --render
    uv run python -m muscle.example_couple --eval
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("WARP_CACHE_PATH", str((PROJECT_ROOT / ".cache" / "warp").resolve()))

import warp as wp

from muscle.couple.bicep_helpers import ELBOW_AXIS, ELBOW_PIVOT, build_elbow_model
from muscle.config import load_config
from muscle.couple.controllability import (
    build_coupling_config,
    config_to_dict,
    parse_levels,
    run_eval_sweep,
)
from muscle.couple.mesh_utils import MeshDistortionError, check_mesh_quality
from muscle.muscle_warp import MuscleSim, activation_ramp
from muscle.couple.solver_muscle_bone_coupled_warp import SolverMuscleBoneCoupled
from muscle.couple.usd_io import UsdIO, build_bone_prim_map

DEFAULT_CONFIG = "data/muscle/config/bicep_coupled.json"

log = logging.getLogger("couple")


def setup_logging(to_file: bool = False):
    """Configure 'couple' logger. Optionally write to log.md for debugging."""
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    couple_log = logging.getLogger("couple")
    couple_log.setLevel(logging.DEBUG)
    couple_log.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if to_file:
        couple_log.addHandler(logging.FileHandler("log.md", mode="w", encoding="utf-8"))
    couple_log.addHandler(logging.StreamHandler())
    for handler in couple_log.handlers:
        handler.setFormatter(fmt)


# ---------------------------------------------------------------------------
# Reusable setup -- scripts import this to avoid duplicating init logic
# ---------------------------------------------------------------------------

def setup_couple(
    config_path: str | None = None,
    device: str | None = None,
    render: bool = False,
    config_overrides: dict | None = None,
):
    """Initialize coupling simulation components.

    Args:
        config_path: Path to muscle config JSON. Defaults to DEFAULT_CONFIG.
        device: Warp device string override. If None, reads from config "arch".
        render: Enable OpenGL interactive rendering.
        config_overrides: Optional dict of cfg attribute overrides (for sweep scripts).

    Returns:
        (solver, sim, state, cfg, dt) tuple.
    """
    config_path = config_path or DEFAULT_CONFIG

    wp.init()

    cfg = load_config(config_path)
    # Device: CLI override > config arch > default "cuda:0"
    device = device or getattr(cfg, 'arch', 'cuda:0')
    cfg.arch = device
    wp.set_device(device)
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)
    cfg.gui = bool(render)
    cfg.render_mode = "human" if render else None

    sim = MuscleSim(cfg)

    dt = 1.0 / 60.0
    joint_friction = float(getattr(cfg, "joint_friction", 0.05))
    model, state, radius_link, joint, selected_indices = build_elbow_model(
        sim, joint_friction=joint_friction)

    # Build coupling config from JSON "coupling" section or defaults
    coupling = getattr(cfg, "coupling", None)
    if coupling:
        preset = getattr(coupling, "preset", "smooth_nonlinear")
        overrides = {k: v for k, v in vars(coupling).items() if k != "preset"}
        control_config = build_coupling_config(preset, **overrides)
    else:
        control_config = build_coupling_config("smooth_nonlinear")

    solver = SolverMuscleBoneCoupled(model, sim, control_config=control_config)

    if radius_link is not None and selected_indices.size > 0:
        solver.configure_coupling(
            bone_body_id=radius_link,
            bone_rest_verts=sim.bone_pos[selected_indices].astype(np.float32),
            bone_vertex_indices=selected_indices,
            joint_index=joint,
            joint_pivot=ELBOW_PIVOT,
            joint_axis=ELBOW_AXIS,
        )

    log.info(
        "dt=%.6f muscle_substeps=%d bone_substeps=%d control=%s",
        dt,
        int(cfg.num_substeps),
        int(solver.bone_substeps),
        config_to_dict(control_config),
    )

    return solver, sim, state, cfg, dt


# ---------------------------------------------------------------------------
# Activation schedule
# ---------------------------------------------------------------------------

def _activation_schedule(step: int, total: int) -> float:
    return activation_ramp(step / max(total, 1))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_muscle_surface_mesh(usd: UsdIO, sim: MuscleSim) -> str | None:
    """Create a renderable surface Mesh prim from the TetMesh boundary triangles."""
    from pxr import Sdf, UsdGeom, Vt

    tet_prim_path = usd.muscle_mesh.mesh_path
    stage = usd._stage
    tet_prim = stage.GetPrimAtPath(tet_prim_path)
    if not tet_prim.IsValid():
        log.warning("TetMesh prim not found: %s", tet_prim_path)
        return None

    sfvi_attr = tet_prim.GetAttribute("surfaceFaceVertexIndices")
    sfvi = sfvi_attr.Get() if sfvi_attr else None
    if sfvi is None or len(sfvi) == 0:
        log.warning("No surfaceFaceVertexIndices on %s", tet_prim_path)
        return None

    tri_indices = np.array(sfvi, dtype=np.int32).reshape(-1)
    n_tris = len(sfvi)

    surf_path = tet_prim_path + "_surface"
    surf_prim = stage.DefinePrim(surf_path, "Mesh")
    mesh = UsdGeom.Mesh(surf_prim)
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * n_tris))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(tri_indices))

    pts_np = np.ascontiguousarray(sim.pos.numpy(), dtype=np.float32)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts_np))

    src_pv = UsdGeom.PrimvarsAPI(tet_prim).GetPrimvar("displayColor")
    if src_pv and src_pv.Get() is not None:
        dst_pv = UsdGeom.PrimvarsAPI(surf_prim).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        dst_pv.Set(src_pv.Get())

    log.info("Created muscle surface mesh: %s (%d triangles)", surf_path, n_tris)
    return surf_path


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_loop(solver, state, cfg, dt: float, n_steps: int, auto: bool,
             usd: UsdIO | None = None, bone_prim_map: dict | None = None,
             sim: MuscleSim | None = None,
             muscle_surface_path: str | None = None,
             use_cuda_graph: bool = False):
    """Run loop with optional rendering, scheduled activation and USD export."""
    tet_idx = sim.tet_np[:, [3, 0, 1, 2]] if sim else None
    tet_poses = sim.rest_matrix.numpy() if sim else None

    for step in range(1, n_steps + 1):
        # Capture CUDA graph after first step (JIT warmup complete)
        if use_cuda_graph and step == 2:
            solver.capture_muscle_graph()
        if sim and sim.renderer and not sim.renderer.is_running():
            break
        if auto:
            cfg.activation = _activation_schedule(step, n_steps)

        solver.step(state, state, dt=dt)

        if tet_idx is not None and tet_poses is not None:
            pos_np = solver.core.pos.numpy()
            try:
                check_mesh_quality(pos_np, tet_idx, tet_poses, step=step)
            except MeshDistortionError:
                log.warning("Mesh distortion at step %d, continuing...", step)

        if sim:
            sim.render()

        if usd is not None:
            usd.set_points(usd.muscle_mesh.mesh_path, solver.core.pos, frame=step)
            if muscle_surface_path:
                usd.set_points(muscle_surface_path, solver.core.pos, frame=step)
            bone_np = solver.core.bone_pos_field.numpy()
            for prim_path, indices in (bone_prim_map or {}).items():
                usd.set_points(prim_path, bone_np[indices], frame=step)
            usd.set_runtime("activation", float(cfg.activation), frame=step)
            joint_q = state.joint_q.numpy()
            usd.set_runtime("joint_angle", float(joint_q[0]) if len(joint_q) > 0 else 0.0, frame=step)

        if step % 25 == 0 or step == 1:
            body_q = state.body_q.numpy()[0]
            joint_q = state.joint_q.numpy()
            joint_angle = float(joint_q[0]) if len(joint_q) > 0 else 0.0
            tau = solver._muscle_torque
            log.info(
                "step=%4d exc=%.2f act=%.2f pos=(%.4f, %.4f, %.4f) q=%.4f axis_tau=%.4f |tau|=%.4f",
                step,
                float(cfg.activation),
                float(solver._effective_activation),
                float(body_q[0]),
                float(body_q[1]),
                float(body_q[2]),
                joint_angle,
                float(solver._axis_torque),
                float(np.linalg.norm(tau)),
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XPBD-Millard muscle-bone coupling (bicep flexion)")
    parser.add_argument("--auto", action="store_true", help="Use built-in activation schedule")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to muscle config JSON")
    parser.add_argument("--device", type=str, default=None, help="Warp device (overrides config arch)")
    parser.add_argument("--render", action="store_true", help="Enable OpenGL interactive rendering")
    parser.add_argument("--no-usd", action="store_true", help="Disable default USD export")
    parser.add_argument("--eval", action="store_true",
                        help="Run activation sweep evaluation instead of time schedule")
    parser.add_argument("--eval-levels", type=str, default="0,0.1,0.3,0.5,0.7,1.0",
                        help="Comma-separated activation levels for sweep")
    parser.add_argument("--eval-hold-steps", type=int, default=90,
                        help="Steps to hold each activation level")
    parser.add_argument("--eval-release-steps", type=int, default=90,
                        help="Steps after returning activation to zero")
    parser.add_argument("--eval-warmup-steps", type=int, default=20,
                        help="Zero-activation warmup before each sweep episode")
    parser.add_argument("--no-cuda-graph", action="store_true",
                        help="Disable CUDA graph capture (auto-enabled on GPU)")
    return parser


def main():
    args = _create_parser().parse_args()

    setup_logging(to_file=True)

    solver, sim, state, cfg, dt = setup_couple(
        config_path=args.config,
        device=args.device,
        render=args.render,
    )
    use_cuda_graph = sim.pos.device.is_cuda and not args.no_cuda_graph

    if args.eval:
        run_eval_sweep(
            solver, sim, state, cfg, dt,
            label="example_couple",
            levels=args.eval_levels,
            hold_steps=args.eval_hold_steps,
            release_steps=args.eval_release_steps,
            warmup_steps=args.eval_warmup_steps,
            output_dir=str(PROJECT_ROOT / "output"),
        )
        return

    usd = None
    bone_prim_map = None
    muscle_surface_path = None
    usd_source = str(cfg.geo_path)
    if not args.no_usd:
        usd = UsdIO(usd_source, y_up_to_z_up=False)
        usd.muscle_mesh = usd.find_mesh("muscle")
        usd.bone_meshes = usd.find_meshes("bone")
        usd.start("output/example_couple.anim.usd", copy_usd=True)
        usd.set_runtime("fps", 60)
        bone_prim_map = build_bone_prim_map(usd, sim)
        muscle_surface_path = _create_muscle_surface_mesh(usd, sim)
        log.info("USD export: %s  bone_groups=%s",
                 usd.output_path, list(bone_prim_map.keys()))

    try:
        run_loop(solver, state, cfg, dt=dt, n_steps=int(max(1, args.steps)),
                 auto=bool(args.auto), usd=usd, bone_prim_map=bone_prim_map,
                 sim=sim, muscle_surface_path=muscle_surface_path,
                 use_cuda_graph=use_cuda_graph)
    finally:
        if usd is not None:
            usd.close()
            log.info("USD saved: %s", usd.output_path)


if __name__ == "__main__":
    main()
