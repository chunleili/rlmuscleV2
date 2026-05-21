"""Physics engine abstraction for MASS.

Provides a `PhysicsEngine` abstract interface and `MuJoCoEngine` concrete
implementation. All physics-engine-specific calls (MuJoCo C API) are
encapsulated here so that env.py, character.py, and muscle.py only depend
on the abstract interface.

Usage:
    from rl_mass.engine import MuJoCoEngine
    engine = MuJoCoEngine("data/rl_mass/humanoid.xml", timestep=1/600)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PhysicsEngine(ABC):
    """Abstract physics engine interface for MASS."""

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def nv(self) -> int:
        """Number of degrees of freedom (velocity space)."""
        ...

    @property
    @abstractmethod
    def nq(self) -> int:
        """Number of generalized coordinates (position space)."""
        ...

    @property
    @abstractmethod
    def nbody(self) -> int:
        """Number of bodies."""
        ...

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Simulation timestep (seconds)."""
        ...

    # ------------------------------------------------------------------
    # Body queries
    # ------------------------------------------------------------------

    @abstractmethod
    def body_id(self, name: str) -> int:
        """Return body index for the given body name."""
        ...

    @abstractmethod
    def body_mass(self, bid: int) -> float:
        """Return mass of body *bid*."""
        ...

    @abstractmethod
    def body_parent_id(self, bid: int) -> int:
        """Return parent body index for body *bid*."""
        ...

    @abstractmethod
    def body_jnt_addr(self, bid: int) -> int:
        """Return address of the first joint belonging to body *bid*."""
        ...

    @abstractmethod
    def body_jnt_num(self, bid: int) -> int:
        """Return number of joints belonging to body *bid*."""
        ...

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @abstractmethod
    def get_qpos(self) -> np.ndarray:
        """Return a *copy* of generalized positions (nq,)."""
        ...

    @abstractmethod
    def get_qvel(self) -> np.ndarray:
        """Return a *copy* of generalized velocities (nv,)."""
        ...

    @abstractmethod
    def set_qpos(self, q: np.ndarray) -> None:
        """Set generalized positions."""
        ...

    @abstractmethod
    def set_qvel(self, v: np.ndarray) -> None:
        """Set generalized velocities."""
        ...

    @abstractmethod
    def get_time(self) -> float:
        """Return current simulation time."""
        ...

    @abstractmethod
    def set_time(self, t: float) -> None:
        """Set current simulation time."""
        ...

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self) -> None:
        """Compute forward kinematics/dynamics (no integration)."""
        ...

    @abstractmethod
    def kinematics(self) -> None:
        """Compute forward kinematics only."""
        ...

    @abstractmethod
    def com_pos(self) -> None:
        """Recompute center-of-mass related quantities."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance simulation by one timestep."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation data to defaults."""
        ...

    @abstractmethod
    def get_mass_matrix(self) -> np.ndarray:
        """Return dense mass matrix (nv, nv)."""
        ...

    @abstractmethod
    def get_coriolis_gravity(self) -> np.ndarray:
        """Return Coriolis + gravity bias forces (nv,)."""
        ...

    @abstractmethod
    def get_constraint_forces(self) -> np.ndarray:
        """Return constraint forces (nv,)."""
        ...

    # ------------------------------------------------------------------
    # Body spatial info
    # ------------------------------------------------------------------

    @abstractmethod
    def get_body_xpos(self, bid: int) -> np.ndarray:
        """Return body frame origin in world frame (3,)."""
        ...

    @abstractmethod
    def get_body_xmat(self, bid: int) -> np.ndarray:
        """Return body frame orientation as 3x3 rotation matrix."""
        ...

    @abstractmethod
    def get_body_com(self, bid: int) -> np.ndarray:
        """Return body center-of-mass in world frame (3,)."""
        ...

    @abstractmethod
    def get_body_com_velocity(self, bid: int) -> np.ndarray:
        """Return COM linear velocity of body *bid* in world frame (3,)."""
        ...

    @abstractmethod
    def get_joint_anchor(self, jnt_id: int) -> np.ndarray:
        """Return joint anchor position in world frame (3,)."""
        ...

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    @abstractmethod
    def get_linear_jacobian(self, point: np.ndarray, bid: int) -> np.ndarray:
        """Return linear (translational) Jacobian at *point* for body *bid*.

        Args:
            point: World-frame point (3,).
            bid: Body index.

        Returns:
            Jacobian of shape (3, nv).
        """
        ...

    @abstractmethod
    def get_com_jacobian(self, bid: int) -> np.ndarray:
        """Return COM linear Jacobian for body *bid*.

        Returns:
            Jacobian of shape (3, nv).
        """
        ...

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    @abstractmethod
    def clear_applied_forces(self) -> None:
        """Zero out all externally applied forces/torques."""
        ...

    @abstractmethod
    def apply_force_and_torque(
        self, bid: int, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """Accumulate a world-frame force and torque on body *bid*.

        Args:
            bid: Body index.
            force: World-frame force (3,).
            torque: World-frame torque (3,).
        """
        ...

    @abstractmethod
    def set_qfrc_applied(self, forces: np.ndarray) -> None:
        """Set generalized applied forces (nv,)."""
        ...


# =========================================================================
# MuJoCo implementation
# =========================================================================


class MuJoCoEngine(PhysicsEngine):
    """MuJoCo implementation of :class:`PhysicsEngine`.

    Args:
        model_path: Path to MJCF XML file.
        timestep: Desired simulation timestep (overrides model default).
    """

    def __init__(self, model_path: str, timestep: float):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

        # Reusable buffer for get_linear_jacobian (avoids per-call np.zeros).
        self._jacp_buf = np.zeros((3, self.model.nv), dtype=np.float64)
        # Reusable buffers for body-COM jacobians (translational + rotational).
        self._jacp_com_buf = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr_com_buf = np.zeros((3, self.model.nv), dtype=np.float64)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nbody(self) -> int:
        return self.model.nbody

    @property
    def timestep(self) -> float:
        return self.model.opt.timestep

    # ------------------------------------------------------------------
    # Body queries
    # ------------------------------------------------------------------

    def body_id(self, name: str) -> int:
        return self._mj.mj_name2id(
            self.model, self._mj.mjtObj.mjOBJ_BODY, name
        )

    def body_mass(self, bid: int) -> float:
        return float(self.model.body_mass[bid])

    def body_parent_id(self, bid: int) -> int:
        return int(self.model.body_parentid[bid])

    def body_jnt_addr(self, bid: int) -> int:
        return int(self.model.body_jntadr[bid])

    def body_jnt_num(self, bid: int) -> int:
        return int(self.model.body_jntnum[bid])

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def get_qvel(self) -> np.ndarray:
        return self.data.qvel.copy()

    def set_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[:] = q

    def set_qvel(self, v: np.ndarray) -> None:
        self.data.qvel[:] = v

    def get_time(self) -> float:
        return float(self.data.time)

    def set_time(self, t: float) -> None:
        self.data.time = t

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def forward(self) -> None:
        self._mj.mj_forward(self.model, self.data)

    def kinematics(self) -> None:
        self._mj.mj_kinematics(self.model, self.data)

    def com_pos(self) -> None:
        self._mj.mj_comPos(self.model, self.data)

    def step(self) -> None:
        self._mj.mj_step(self.model, self.data)

    def reset(self) -> None:
        self._mj.mj_resetData(self.model, self.data)

    def get_mass_matrix(self) -> np.ndarray:
        nv = self.model.nv
        M = np.zeros((nv, nv), dtype=np.float64)
        self._mj.mj_fullM(self.model, M, self.data.qM)
        return M

    def get_coriolis_gravity(self) -> np.ndarray:
        return self.data.qfrc_bias.copy()

    def get_constraint_forces(self) -> np.ndarray:
        return self.data.qfrc_constraint.copy()

    # ------------------------------------------------------------------
    # Body spatial info
    # ------------------------------------------------------------------

    def get_body_xpos(self, bid: int) -> np.ndarray:
        return self.data.xpos[bid].copy()

    def get_body_xmat(self, bid: int) -> np.ndarray:
        return self.data.xmat[bid].reshape(3, 3).copy()

    def get_body_com(self, bid: int) -> np.ndarray:
        return self.data.xipos[bid].copy()

    def get_body_com_velocity(self, bid: int) -> np.ndarray:
        nv = self.model.nv
        jacp = np.zeros((3, nv), dtype=np.float64)
        self._mj.mj_jacBodyCom(self.model, self.data, jacp, None, bid)
        return jacp @ self.data.qvel

    def get_joint_anchor(self, jnt_id: int) -> np.ndarray:
        return self.data.xanchor[jnt_id].copy()

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    def get_linear_jacobian(self, point: np.ndarray, bid: int) -> np.ndarray:
        # Reuse a single buffer; mj_jac fully overwrites jacp every call.
        # jacr=None skips the (unused) rotational Jacobian.
        # NOTE: returns a shared buffer — copy if you must retain it past
        # the next call.
        jacp = self._jacp_buf
        self._mj.mj_jac(self.model, self.data, jacp, None, point, bid)
        return jacp

    def get_com_jacobian(self, bid: int) -> np.ndarray:
        nv = self.model.nv
        jacp = np.zeros((3, nv), dtype=np.float64)
        self._mj.mj_jacBodyCom(self.model, self.data, jacp, None, bid)
        return jacp

    def get_body_com_jacobians(self, bid: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (jacp, jacr) at body COM (xipos). Buffers reused — copy if retained."""
        jacp = self._jacp_com_buf
        jacr = self._jacr_com_buf
        self._mj.mj_jacBodyCom(self.model, self.data, jacp, jacr, bid)
        return jacp, jacr

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def clear_applied_forces(self) -> None:
        self.data.xfrc_applied[:] = 0.0

    def apply_force_and_torque(
        self, bid: int, force: np.ndarray, torque: np.ndarray
    ) -> None:
        self.data.xfrc_applied[bid, :3] += force
        self.data.xfrc_applied[bid, 3:] += torque

    def set_qfrc_applied(self, forces: np.ndarray) -> None:
        self.data.qfrc_applied[:] = forces
