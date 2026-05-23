"""Batched `mujoco_warp` engine wrapper for the Phase 1 GPU backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _patch_mujoco_warp_tendon_compat() -> None:
    """Patch `mujoco_warp.put_data` for MuJoCo builds with dense `ten_J`.

    MASS currently loads the muscle geometry from a separate XML file, so this
    path is usually dormant. Keeping the patch here makes the wrapper robust to
    future tendonized models and matches the proven fix from the sibling
    `RLMuscle` codebase.
    """

    import mujoco_warp
    import mujoco_warp._src.io as _mw_io

    if getattr(mujoco_warp.put_data, "__name__", "") == "_patched_put_data":
        return

    original_put_data = _mw_io.put_data

    class _MjDataTendonProxy:
        def __init__(self, mjd, mjm):
            object.__setattr__(self, "_mjd", mjd)
            nv = mjm.nv
            nt = mjm.ntendon
            object.__setattr__(self, "_ten_J_rownnz", np.full(nt, nv, dtype=np.int32))
            object.__setattr__(self, "_ten_J_rowadr", np.arange(nt, dtype=np.int32) * nv)
            object.__setattr__(
                self,
                "_ten_J_colind",
                np.tile(np.arange(nv, dtype=np.int32), nt),
            )

        def __getattr__(self, name):
            if name == "ten_J_rownnz":
                return object.__getattribute__(self, "_ten_J_rownnz")
            if name == "ten_J_rowadr":
                return object.__getattribute__(self, "_ten_J_rowadr")
            if name == "ten_J_colind":
                return object.__getattribute__(self, "_ten_J_colind")
            return getattr(object.__getattribute__(self, "_mjd"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_mjd"), name, value)

    def _patched_put_data(mjm, mjd, nworld=1, nconmax=None, njmax=None):
        if mjm.ntendon > 0 and not hasattr(mjd, "ten_J_rownnz"):
            mjd = _MjDataTendonProxy(mjd, mjm)
        return original_put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)

    _mw_io.put_data = _patched_put_data
    mujoco_warp.put_data = _patched_put_data


@dataclass(frozen=True)
class BodyMetadata:
    parent_id: int
    joint_addr: int
    joint_count: int
    mass: float


class WarpEngine:
    """Batched MuJoCo physics wrapper backed by `mujoco_warp`.

    This is the first real Phase 1 building block: it owns a tiled MuJoCo
    model/data pair on the Warp device and exposes batched state access plus
    stepping/forward/reset operations. Higher-level MASS environment logic
    still needs to be ported separately.
    """

    def __init__(
        self,
        model_path: str,
        timestep: float,
        num_worlds: int = 1,
        device: Optional[str] = None,
    ):
        if num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")

        import mujoco
        import mujoco_warp
        import warp as wp

        _patch_mujoco_warp_tendon_compat()

        self._mujoco = mujoco
        self._mujoco_warp = mujoco_warp
        self._wp = wp
        self.device = device or wp.get_preferred_device().alias
        self.num_worlds = int(num_worlds)

        with wp.ScopedDevice(self.device):
            self.cpu_model = mujoco.MjModel.from_xml_path(model_path)
            self.cpu_model.opt.timestep = timestep
            self.cpu_data = mujoco.MjData(self.cpu_model)
            self.model = mujoco_warp.put_model(self.cpu_model)
            # Force ample njmax/nconmax to avoid "nefc overflow" silently
            # dropping constraints at large N (default sizes were tuned for
            # single-world; at 1024+ worlds, occasional worlds exceed them
            # and physics diverges -> all envs immediately hit eoe).
            self.data = mujoco_warp.put_data(
                self.cpu_model,
                self.cpu_data,
                nworld=self.num_worlds,
                njmax=1024,
                nconmax=128,
            )

        self._initial_qpos = self.get_qpos().copy()
        self._initial_qvel = self.get_qvel().copy()
        self._step_requests = 0
        self._jacp_cpu_buf = np.zeros((3, self.nv), dtype=np.float64)
        self._jacp_com_cpu_buf = np.zeros((3, self.nv), dtype=np.float64)
        self._jacr_com_cpu_buf = np.zeros((3, self.nv), dtype=np.float64)
        self._body_meta = [
            BodyMetadata(
                parent_id=int(self.cpu_model.body_parentid[bid]),
                joint_addr=int(self.cpu_model.body_jntadr[bid]),
                joint_count=int(self.cpu_model.body_jntnum[bid]),
                mass=float(self.cpu_model.body_mass[bid]),
            )
            for bid in range(self.cpu_model.nbody)
        ]
        qM_rows, qM_cols = self._build_qm_scatter_indices()
        self._qM_rows = torch.as_tensor(qM_rows, device=self.device, dtype=torch.long)
        self._qM_cols = torch.as_tensor(qM_cols, device=self.device, dtype=torch.long)
        self._torch_diag_idx = torch.arange(self.nv, device=self.device, dtype=torch.long)

    @property
    def nq(self) -> int:
        return int(self.cpu_model.nq)

    @property
    def nv(self) -> int:
        return int(self.cpu_model.nv)

    @property
    def nbody(self) -> int:
        return int(self.cpu_model.nbody)

    @property
    def timestep(self) -> float:
        return float(self.cpu_model.opt.timestep)

    def body_id(self, name: str) -> int:
        return self._mujoco.mj_name2id(
            self.cpu_model, self._mujoco.mjtObj.mjOBJ_BODY, name
        )

    def body_mass(self, bid: int) -> float:
        return self._body_meta[bid].mass

    def body_parent_id(self, bid: int) -> int:
        return self._body_meta[bid].parent_id

    def body_jnt_addr(self, bid: int) -> int:
        return self._body_meta[bid].joint_addr

    def body_jnt_num(self, bid: int) -> int:
        return self._body_meta[bid].joint_count

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos.numpy().copy()

    def get_qvel(self) -> np.ndarray:
        return self.data.qvel.numpy().copy()

    def set_qpos(self, q: np.ndarray) -> None:
        q_arr = np.asarray(q, dtype=np.float32)
        if q_arr.shape == (self.nq,):
            q_arr = np.broadcast_to(q_arr, (self.num_worlds, self.nq)).copy()
        if q_arr.shape != (self.num_worlds, self.nq):
            raise ValueError(
                f"qpos shape {q_arr.shape} != ({self.num_worlds}, {self.nq})"
            )
        self.data.qpos.assign(q_arr)

    def set_qvel(self, v: np.ndarray) -> None:
        v_arr = np.asarray(v, dtype=np.float32)
        if v_arr.shape == (self.nv,):
            v_arr = np.broadcast_to(v_arr, (self.num_worlds, self.nv)).copy()
        if v_arr.shape != (self.num_worlds, self.nv):
            raise ValueError(
                f"qvel shape {v_arr.shape} != ({self.num_worlds}, {self.nv})"
            )
        self.data.qvel.assign(v_arr)

    def get_time(self) -> np.ndarray:
        return self.data.time.numpy().copy()

    def set_time(self, t) -> None:
        t_arr = np.asarray(t, dtype=np.float32)
        if t_arr.shape == ():
            t_arr = np.full((self.num_worlds,), float(t_arr), dtype=np.float32)
        if t_arr.shape != (self.num_worlds,):
            raise ValueError(f"time shape {t_arr.shape} != ({self.num_worlds},)")
        self.data.time.assign(t_arr)

    def forward(self) -> None:
        self._mujoco_warp.forward(self.model, self.data)

    def forward_dynamics(self) -> None:
        """Refresh device-side dynamics buffers for the current state."""
        self.forward()

    def kinematics(self) -> None:
        self._mujoco_warp.kinematics(self.model, self.data)

    def com_pos(self) -> None:
        self._mujoco_warp.com_pos(self.model, self.data)

    def step_physics_only(self) -> None:
        """Pure-GPU substep: snapshot qpos/qvel, then mujoco_warp step.

        Safe to record into a CUDA graph. Caller must invoke
        `recover_nan_if_any` afterwards to roll back any world whose state
        became non-finite (the host-side check forces a D->H sync, so it
        cannot live inside the captured graph).
        """
        import warp as wp
        if not hasattr(self, "_qpos_snap"):
            self._qpos_snap = wp.zeros_like(self.data.qpos)
            self._qvel_snap = wp.zeros_like(self.data.qvel)
        wp.copy(self._qpos_snap, self.data.qpos)
        wp.copy(self._qvel_snap, self.data.qvel)
        self._mujoco_warp.step(self.model, self.data)

    def recover_nan_if_any(self) -> None:
        """Roll back worlds whose qpos/qvel went non-finite to their snapshot."""
        q_host = self.data.qpos.numpy()
        v_host = self.data.qvel.numpy()
        bad_q = ~np.all(np.isfinite(q_host), axis=1)
        bad_v = ~np.all(np.isfinite(v_host), axis=1)
        bad = bad_q | bad_v
        if bad.any():
            snap_q = self._qpos_snap.numpy()
            snap_v = self._qvel_snap.numpy()
            for w in np.where(bad)[0]:
                q_host[w] = snap_q[w]
                v_host[w] = 0.0
            self.set_qpos(q_host)
            self.set_qvel(v_host)
            self.forward()

    def step(self) -> None:
        self.step_physics_only()
        self.recover_nan_if_any()

    def reset(self) -> None:
        self.set_qpos(self._initial_qpos)
        self.set_qvel(self._initial_qvel)
        self.set_time(0.0)
        self.clear_applied_forces()
        self.set_qfrc_applied(np.zeros((self.num_worlds, self.nv), dtype=np.float32))
        self.forward()

    def get_coriolis_gravity(self) -> np.ndarray:
        return self.data.qfrc_bias.numpy().copy()

    def get_constraint_forces(self) -> np.ndarray:
        return self.data.qfrc_constraint.numpy().copy()

    def qpos_tensor(self) -> torch.Tensor:
        return self._warp_to_torch(self.data.qpos)

    def qvel_tensor(self) -> torch.Tensor:
        return self._warp_to_torch(self.data.qvel)

    def coriolis_gravity_tensor(self) -> torch.Tensor:
        return self._warp_to_torch(self.data.qfrc_bias)

    def constraint_forces_tensor(self) -> torch.Tensor:
        return self._warp_to_torch(self.data.qfrc_constraint)

    def mass_matrix_tensor(self) -> torch.Tensor:
        qM = self._warp_to_torch(self.data.qM).squeeze(1)
        M = torch.zeros(
            (self.num_worlds, self.nv, self.nv),
            device=qM.device,
            dtype=qM.dtype,
        )
        row_idx = self._qM_rows.unsqueeze(0).expand(self.num_worlds, -1)
        col_idx = self._qM_cols.unsqueeze(0).expand(self.num_worlds, -1)
        world_idx = torch.arange(
            self.num_worlds, device=qM.device, dtype=torch.long
        ).unsqueeze(1).expand_as(row_idx)
        M[world_idx, row_idx, col_idx] = qM
        M[world_idx, col_idx, row_idx] = qM
        return M

    def get_mass_matrix_world(self, world_id: int) -> np.ndarray:
        self._sync_world_to_cpu(world_id)
        M = np.zeros((self.nv, self.nv), dtype=np.float64)
        self._mujoco.mj_fullM(self.cpu_model, M, self.cpu_data.qM)
        return M

    def get_body_xpos(self, bid: int) -> np.ndarray:
        return self.data.xpos.numpy()[:, bid, :].copy()

    def get_body_xmat(self, bid: int) -> np.ndarray:
        return self.data.xmat.numpy()[:, bid, :, :].copy()

    def get_body_com(self, bid: int) -> np.ndarray:
        return self.data.xipos.numpy()[:, bid, :].copy()

    def get_body_com_velocity(self, bid: int) -> np.ndarray:
        cvel = self.data.cvel.numpy()[:, bid, :]
        return cvel[:, 3:].copy()

    def get_body_xpos_world(self, world_id: int, bid: int) -> np.ndarray:
        return self.data.xpos.numpy()[world_id, bid, :].copy()

    def get_body_xmat_world(self, world_id: int, bid: int) -> np.ndarray:
        return self.data.xmat.numpy()[world_id, bid, :, :].copy()

    def get_body_com_world(self, world_id: int, bid: int) -> np.ndarray:
        return self.data.xipos.numpy()[world_id, bid, :].copy()

    def get_body_com_velocity_world(self, world_id: int, bid: int) -> np.ndarray:
        return self.data.cvel.numpy()[world_id, bid, 3:].copy()

    def get_joint_anchor_world(self, world_id: int, jnt_id: int) -> np.ndarray:
        self._sync_world_to_cpu(world_id)
        return self.cpu_data.xanchor[jnt_id].copy()

    def clear_applied_forces(self) -> None:
        self.data.xfrc_applied.zero_()

    def clear_applied_forces_world(self, world_id: int) -> None:
        xfrc = self.data.xfrc_applied.numpy()
        xfrc[world_id, :, :] = 0.0
        self.data.xfrc_applied.assign(xfrc)

    def apply_force_and_torque(
        self,
        bid: int,
        force: np.ndarray,
        torque: np.ndarray,
        world_id: Optional[int] = None,
    ) -> None:
        xfrc = self.data.xfrc_applied.numpy()
        force_arr = np.asarray(force, dtype=np.float32)
        torque_arr = np.asarray(torque, dtype=np.float32)
        if world_id is None:
            if force_arr.shape == (3,):
                force_arr = np.broadcast_to(force_arr, (self.num_worlds, 3)).copy()
            if torque_arr.shape == (3,):
                torque_arr = np.broadcast_to(torque_arr, (self.num_worlds, 3)).copy()
            if force_arr.shape != (self.num_worlds, 3):
                raise ValueError(
                    f"force shape {force_arr.shape} != ({self.num_worlds}, 3)"
                )
            if torque_arr.shape != (self.num_worlds, 3):
                raise ValueError(
                    f"torque shape {torque_arr.shape} != ({self.num_worlds}, 3)"
                )
            xfrc[:, bid, :3] += force_arr
            xfrc[:, bid, 3:] += torque_arr
        else:
            xfrc[world_id, bid, :3] += force_arr
            xfrc[world_id, bid, 3:] += torque_arr
        self.data.xfrc_applied.assign(xfrc)

    def set_qfrc_applied(self, forces: np.ndarray) -> None:
        frc_arr = np.asarray(forces, dtype=np.float32)
        if frc_arr.shape == (self.nv,):
            frc_arr = np.broadcast_to(frc_arr, (self.num_worlds, self.nv)).copy()
        if frc_arr.shape != (self.num_worlds, self.nv):
            raise ValueError(
                f"qfrc_applied shape {frc_arr.shape} != ({self.num_worlds}, {self.nv})"
            )
        self.data.qfrc_applied.assign(frc_arr)

    def set_qfrc_applied_world(self, world_id: int, forces: np.ndarray) -> None:
        frc = self.data.qfrc_applied.numpy()
        frc[world_id, :] = np.asarray(forces, dtype=np.float32)
        self.data.qfrc_applied.assign(frc)

    def set_qpos_world(self, world_id: int, q: np.ndarray) -> None:
        qpos = self.data.qpos.numpy()
        qpos[world_id, :] = np.asarray(q, dtype=np.float32)
        self.data.qpos.assign(qpos)

    def set_qvel_world(self, world_id: int, v: np.ndarray) -> None:
        qvel = self.data.qvel.numpy()
        qvel[world_id, :] = np.asarray(v, dtype=np.float32)
        self.data.qvel.assign(qvel)

    def set_time_world(self, world_id: int, t: float) -> None:
        times = self.data.time.numpy()
        times[world_id] = np.float32(t)
        self.data.time.assign(times)

    def reset_world(self, world_id: int) -> None:
        self.set_qpos_world(world_id, self._initial_qpos[world_id])
        self.set_qvel_world(world_id, self._initial_qvel[world_id])
        self.set_time_world(world_id, 0.0)
        self.clear_applied_forces_world(world_id)
        self.set_qfrc_applied_world(world_id, np.zeros((self.nv,), dtype=np.float32))
        self.forward()

    def request_batched_step(self) -> None:
        self._step_requests += 1
        if self._step_requests > self.num_worlds:
            self._step_requests = 0
            raise RuntimeError("WarpEngine step barrier overflow")
        if self._step_requests == self.num_worlds:
            self.step()
            self._step_requests = 0

    def get_linear_jacobian_world(self, world_id: int, point: np.ndarray, bid: int) -> np.ndarray:
        self._sync_world_to_cpu(world_id)
        self._mujoco.mj_jac(
            self.cpu_model,
            self.cpu_data,
            self._jacp_cpu_buf,
            None,
            np.asarray(point, dtype=np.float64),
            bid,
        )
        return self._jacp_cpu_buf.copy()

    def get_com_jacobian_world(self, world_id: int, bid: int) -> np.ndarray:
        self._sync_world_to_cpu(world_id)
        self._mujoco.mj_jacBodyCom(
            self.cpu_model,
            self.cpu_data,
            self._jacp_com_cpu_buf,
            None,
            bid,
        )
        return self._jacp_com_cpu_buf.copy()

    def get_body_com_jacobians_world(self, world_id: int, bid: int) -> tuple[np.ndarray, np.ndarray]:
        self._sync_world_to_cpu(world_id)
        self._mujoco.mj_jacBodyCom(
            self.cpu_model,
            self.cpu_data,
            self._jacp_com_cpu_buf,
            self._jacr_com_cpu_buf,
            bid,
        )
        return self._jacp_com_cpu_buf.copy(), self._jacr_com_cpu_buf.copy()

    def _sync_world_to_cpu(self, world_id: int) -> None:
        self.cpu_data.qpos[:] = self.data.qpos.numpy()[world_id]
        self.cpu_data.qvel[:] = self.data.qvel.numpy()[world_id]
        self.cpu_data.time = float(self.data.time.numpy()[world_id])
        self.cpu_data.qfrc_applied[:] = self.data.qfrc_applied.numpy()[world_id]
        self.cpu_data.xfrc_applied[:] = self.data.xfrc_applied.numpy()[world_id]
        self._mujoco.mj_forward(self.cpu_model, self.cpu_data)

    def close(self) -> None:
        pass

    def _warp_to_torch(self, arr):
        return self._wp.to_torch(arr)

    def _build_qm_scatter_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Build compact-qM -> dense-(row,col) mapping for MuJoCo sparse M."""
        rows = np.empty((self.cpu_model.nM,), dtype=np.int64)
        cols = np.empty_like(rows)
        for row in range(self.nv):
            adr = int(self.cpu_model.M_rowadr[row])
            for k in range(int(self.cpu_model.M_rownnz[row])):
                cols[adr + k] = int(self.cpu_model.M_colind[adr + k])
                rows[adr + k] = row
        compact = np.asarray(self.cpu_model.mapM2M, dtype=np.int64)
        return rows[compact], cols[compact]


class WarpWorldView:
    """Single-world `PhysicsEngine`-like view over a shared `WarpEngine`."""

    def __init__(self, engine: WarpEngine, world_id: int):
        self._engine = engine
        self.world_id = int(world_id)
        self.model = engine.cpu_model

    @property
    def data(self):
        raise AttributeError("WarpWorldView has no direct `.data`; use scalar APIs.")

    @property
    def nv(self) -> int:
        return self._engine.nv

    @property
    def nq(self) -> int:
        return self._engine.nq

    @property
    def nbody(self) -> int:
        return self._engine.nbody

    @property
    def timestep(self) -> float:
        return self._engine.timestep

    def body_id(self, name: str) -> int:
        return self._engine.body_id(name)

    def body_mass(self, bid: int) -> float:
        return self._engine.body_mass(bid)

    def body_parent_id(self, bid: int) -> int:
        return self._engine.body_parent_id(bid)

    def body_jnt_addr(self, bid: int) -> int:
        return self._engine.body_jnt_addr(bid)

    def body_jnt_num(self, bid: int) -> int:
        return self._engine.body_jnt_num(bid)

    def get_qpos(self) -> np.ndarray:
        return self._engine.get_qpos()[self.world_id].copy()

    def get_qvel(self) -> np.ndarray:
        return self._engine.get_qvel()[self.world_id].copy()

    def set_qpos(self, q: np.ndarray) -> None:
        self._engine.set_qpos_world(self.world_id, q)

    def set_qvel(self, v: np.ndarray) -> None:
        self._engine.set_qvel_world(self.world_id, v)

    def get_time(self) -> float:
        return float(self._engine.get_time()[self.world_id])

    def set_time(self, t: float) -> None:
        self._engine.set_time_world(self.world_id, t)

    def forward(self) -> None:
        self._engine.forward()

    def kinematics(self) -> None:
        self._engine.kinematics()

    def com_pos(self) -> None:
        self._engine.com_pos()

    def step(self) -> None:
        self._engine.request_batched_step()

    def reset(self) -> None:
        self._engine.reset_world(self.world_id)

    def get_mass_matrix(self) -> np.ndarray:
        return self._engine.get_mass_matrix_world(self.world_id)

    def get_coriolis_gravity(self) -> np.ndarray:
        return self._engine.get_coriolis_gravity()[self.world_id].copy()

    def get_constraint_forces(self) -> np.ndarray:
        return self._engine.get_constraint_forces()[self.world_id].copy()

    def get_body_xpos(self, bid: int) -> np.ndarray:
        return self._engine.get_body_xpos_world(self.world_id, bid)

    def get_body_xmat(self, bid: int) -> np.ndarray:
        return self._engine.get_body_xmat_world(self.world_id, bid)

    def get_body_com(self, bid: int) -> np.ndarray:
        return self._engine.get_body_com_world(self.world_id, bid)

    def get_body_com_velocity(self, bid: int) -> np.ndarray:
        return self._engine.get_body_com_velocity_world(self.world_id, bid)

    def get_joint_anchor(self, jnt_id: int) -> np.ndarray:
        return self._engine.get_joint_anchor_world(self.world_id, jnt_id)

    def get_linear_jacobian(self, point: np.ndarray, bid: int) -> np.ndarray:
        return self._engine.get_linear_jacobian_world(self.world_id, point, bid)

    def get_com_jacobian(self, bid: int) -> np.ndarray:
        return self._engine.get_com_jacobian_world(self.world_id, bid)

    def get_body_com_jacobians(self, bid: int) -> tuple[np.ndarray, np.ndarray]:
        return self._engine.get_body_com_jacobians_world(self.world_id, bid)

    def clear_applied_forces(self) -> None:
        self._engine.clear_applied_forces_world(self.world_id)

    def apply_force_and_torque(self, bid: int, force: np.ndarray, torque: np.ndarray) -> None:
        self._engine.apply_force_and_torque(
            bid,
            force,
            torque,
            world_id=self.world_id,
        )

    def set_qfrc_applied(self, forces: np.ndarray) -> None:
        self._engine.set_qfrc_applied_world(self.world_id, forces)
