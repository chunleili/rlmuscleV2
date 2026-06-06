"""Vectorized MassEnv for parallel rollout collection.

Mirrors MASS EnvManager (external/MASS/python/EnvManager.cpp) with two
implementations:

  - SerialVecEnv: in-process for-loop (debug / bit-level regression baseline).
  - SubprocVecEnv: multiprocessing workers with Pipe command dispatch.

Both expose the batch API described in docs/specs/2026-04-24-rl-mass-vecenv-spec.md.

Usage:
    from rl_mass.vec_env import make_vec_env
    env = make_vec_env("data/rl_mass/config.json", num_envs=16, mode="subproc")
    states = env.get_states()               # (N, S)
    env.set_actions(actions)                # (N, A)
    env.steps(2)
    mt = env.get_muscle_torques()           # (N, ...)
    dt = env.get_desired_torques()          # (N, ...)
    env.set_activation_levels(activations)  # (N, M)
    eoe = env.is_end_of_episode(0)          # per-index
    env.reset(rsi=True, idx=0)              # per-index

    # P2 packed path (subproc only):
    env.update_muscle_nn(muscle_model.state_dict())  # once per PPO iter
    results = env.run_control_step(actions)          # list of (state, reward, eoe)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import random
import time
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class MassVecEnv:
    """Abstract batch-API over N MassEnv instances. See module docstring."""

    num_envs: int
    num_state: int
    num_action: int
    num_muscles: int
    use_muscle: bool
    simulation_hz: int
    control_hz: int
    num_total_related_dofs: int

    # Batch I/O
    def get_states(self) -> np.ndarray: raise NotImplementedError
    def set_actions(self, acts: np.ndarray) -> None: raise NotImplementedError
    def get_muscle_torques(self) -> np.ndarray: raise NotImplementedError
    def get_desired_torques(self) -> np.ndarray: raise NotImplementedError
    def set_activation_levels(self, acts: np.ndarray) -> None: raise NotImplementedError

    # Stepping
    def steps(self, n: int) -> None: raise NotImplementedError
    def steps_at_once(self) -> None: raise NotImplementedError

    # Per-index queries
    def is_end_of_episode(self, idx: int) -> bool: raise NotImplementedError
    def get_reward(self, idx: int) -> float: raise NotImplementedError
    def reset(self, rsi: bool = True, idx: Optional[int] = None):
        """Reset env(s). Returns fresh state if idx is given, else None."""
        raise NotImplementedError

    # Muscle tuples
    def gather_muscle_tuples(self) -> list: raise NotImplementedError

    # P2 packed path
    def update_muscle_nn(self, state_dict: dict) -> None: raise NotImplementedError
    def run_control_step(self, actions: np.ndarray) -> list[tuple[np.ndarray, float, bool]]:
        raise NotImplementedError

    def close(self) -> None: pass


# ---------------------------------------------------------------------------
# Serial
# ---------------------------------------------------------------------------


class SerialVecEnv(MassVecEnv):
    """In-process serial implementation. Used for debug and bit-level regression."""

    def __init__(self, config_path: str, num_envs: int):
        from rl_mass.env_factory import make_env

        self.envs = [make_env(config_path) for _ in range(num_envs)]
        e0 = self.envs[0]
        self.num_envs = num_envs
        self.num_state = e0.num_state
        self.num_action = e0.num_action
        self.num_muscles = e0.num_muscles
        self.use_muscle = e0.use_muscle
        self.simulation_hz = e0.simulation_hz
        self.control_hz = e0.control_hz
        self.num_total_related_dofs = e0.num_total_related_dofs

    def get_states(self) -> np.ndarray:
        return np.array([e.get_state() for e in self.envs])

    def set_actions(self, acts: np.ndarray) -> None:
        for e, a in zip(self.envs, acts):
            e.set_action(a)

    def get_muscle_torques(self) -> np.ndarray:
        return np.array(
            [e.get_muscle_torques() for e in self.envs], dtype=np.float32
        )

    def get_desired_torques(self) -> np.ndarray:
        return np.array(
            [e.get_desired_torques() for e in self.envs], dtype=np.float32
        )

    def set_activation_levels(self, acts: np.ndarray) -> None:
        for e, a in zip(self.envs, acts):
            e.set_activation_levels(a)

    def steps(self, n: int) -> None:
        for e in self.envs:
            for _ in range(n):
                e.step()

    def steps_at_once(self) -> None:
        for e in self.envs:
            e.steps_at_once()

    def is_end_of_episode(self, idx: int) -> bool:
        return self.envs[idx].is_end_of_episode()

    def get_reward(self, idx: int) -> float:
        return self.envs[idx].get_reward()

    def reset(self, rsi: bool = True, idx: Optional[int] = None):
        if idx is None:
            for e in self.envs:
                e.reset(rsi=rsi)
            return None
        return self.envs[idx].reset(rsi=rsi)

    def gather_muscle_tuples(self) -> list:
        out: list = []
        for e in self.envs:
            out.extend(e.muscle_tuples)
            e.muscle_tuples.clear()
        return out

    def update_muscle_nn(self, state_dict: dict) -> None:
        # No-op: SerialVecEnv uses the main-process MuscleNN.
        pass

    def run_control_step(self, actions: np.ndarray) -> list:
        raise NotImplementedError(
            "SerialVecEnv does not support packed run_control_step; "
            "use batch API (set_actions, get_muscle_torques, ...) directly."
        )


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _worker_loop(child_conn, config_path: str, base_seed: int, worker_id: int):
    """Worker process main loop. Receives (cmd, args), sends back result.

    Runs in a fresh process (`spawn` start method). Must re-import everything.
    Force CPU torch here so each worker's MuscleNN doesn't fight for GPU.
    """
    # CPU-only torch for workers — set BEFORE importing rl_mass.model.
    os.environ["MASS_DEVICE"] = "cpu"
    # Single-thread BLAS per worker — otherwise 16 workers × N BLAS threads = thrash.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Per-worker seed so 16 envs don't produce identical trajectories.
    worker_seed = base_seed + worker_id * 7919  # arbitrary stride prime
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    import torch
    torch.manual_seed(worker_seed)
    torch.set_num_threads(1)

    from rl_mass.env_factory import make_env

    env = make_env(config_path)

    # Worker-side MuscleNN, constructed lazily on first update_muscle_nn.
    muscle_nn = None
    num_sim_per_control = env.simulation_hz // env.control_hz

    def _build_muscle_nn():
        from rl_mass.model import MuscleNN
        m = MuscleNN(env.num_total_related_dofs, env.num_action, env.num_muscles)
        # Defensive: force everything to CPU regardless of how use_cuda resolved.
        m.cpu()
        m.std_muscle_tau = m.std_muscle_tau.cpu()
        m.std_tau = m.std_tau.cpu()
        m.eval()
        return m

    try:
        while True:
            try:
                cmd, args = child_conn.recv()
            except EOFError:
                break

            if cmd == "close":
                child_conn.close()
                return

            try:
                if cmd == "get_attrs":
                    child_conn.send({
                        "num_state": env.num_state,
                        "num_action": env.num_action,
                        "num_muscles": env.num_muscles,
                        "use_muscle": env.use_muscle,
                        "simulation_hz": env.simulation_hz,
                        "control_hz": env.control_hz,
                        "num_total_related_dofs": env.num_total_related_dofs,
                    })
                elif cmd == "get_state":
                    child_conn.send(env.get_state())
                elif cmd == "set_action":
                    env.set_action(args)
                    child_conn.send(None)
                elif cmd == "get_muscle_torques":
                    child_conn.send(env.get_muscle_torques())
                elif cmd == "get_desired_torques":
                    child_conn.send(env.get_desired_torques())
                elif cmd == "set_activation_levels":
                    env.set_activation_levels(args)
                    child_conn.send(None)
                elif cmd == "step":
                    env.step()
                    child_conn.send(None)
                elif cmd == "steps":
                    for _ in range(int(args)):
                        env.step()
                    child_conn.send(None)
                elif cmd == "steps_at_once":
                    env.steps_at_once()
                    child_conn.send(None)
                elif cmd == "is_end_of_episode":
                    child_conn.send(env.is_end_of_episode())
                elif cmd == "get_reward":
                    child_conn.send(env.get_reward())
                elif cmd == "reset":
                    new_state = env.reset(rsi=bool(args))
                    child_conn.send(new_state)
                elif cmd == "gather_muscle_tuples":
                    out = env.muscle_tuples[:]
                    env.muscle_tuples.clear()
                    child_conn.send(out)
                elif cmd == "update_muscle_nn":
                    if muscle_nn is None:
                        muscle_nn = _build_muscle_nn()
                    sd = pickle.loads(args) if isinstance(args, (bytes, bytearray)) else args
                    muscle_nn.load_state_dict(sd)
                    muscle_nn.eval()
                    child_conn.send(None)
                elif cmd == "run_control_step":
                    action = args
                    env.set_action(action)
                    if env.use_muscle:
                        assert muscle_nn is not None, (
                            "run_control_step requires update_muscle_nn first"
                        )
                        mt_np = env.get_muscle_torques().astype(np.float32)
                        mt = torch.from_numpy(mt_np).unsqueeze(0)
                        with torch.no_grad():
                            for _ in range(num_sim_per_control // 2):
                                dt_np = env.get_desired_torques().astype(np.float32)
                                dt = torch.from_numpy(dt_np).unsqueeze(0)
                                act = muscle_nn(mt, dt).squeeze(0).numpy()
                                env.set_activation_levels(act.astype(np.float64))
                                env.step()
                                env.step()
                    else:
                        env.steps_at_once()
                    state = env.get_state()
                    eoe = env.is_end_of_episode()
                    reward = 0.0 if eoe else env.get_reward()
                    child_conn.send((state, float(reward), bool(eoe)))
                else:
                    child_conn.send(RuntimeError(f"Unknown command: {cmd}"))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                child_conn.send(RuntimeError(f"Worker {worker_id} cmd={cmd} failed:\n{tb}"))
    finally:
        try:
            child_conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Subproc
# ---------------------------------------------------------------------------


class SubprocVecEnv(MassVecEnv):
    """Multiprocessing vec env with Pipe command dispatch.

    Supports both the batch-API (P1) and packed run_control_step (P2).
    """

    def __init__(self, config_path: str, num_envs: int, seed: Optional[int] = None):
        self.num_envs = num_envs
        ctx = mp.get_context("spawn")
        base_seed = seed if seed is not None else (int(time.time() * 1000) & 0xFFFF)

        self.parents = []
        self.procs = []
        for i in range(num_envs):
            parent, child = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_worker_loop,
                args=(child, config_path, base_seed, i),
                daemon=True,
            )
            p.start()
            child.close()  # parent keeps only its end
            self.parents.append(parent)
            self.procs.append(p)

        # Fetch shared attrs from worker 0 (all workers identical).
        self.parents[0].send(("get_attrs", None))
        attrs = self._recv_checked(self.parents[0])
        self.num_state = attrs["num_state"]
        self.num_action = attrs["num_action"]
        self.num_muscles = attrs["num_muscles"]
        self.use_muscle = attrs["use_muscle"]
        self.simulation_hz = attrs["simulation_hz"]
        self.control_hz = attrs["control_hz"]
        self.num_total_related_dofs = attrs["num_total_related_dofs"]

        self._closed = False

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _recv_checked(parent):
        r = parent.recv()
        if isinstance(r, BaseException):
            raise r
        return r

    def _broadcast_same(self, cmd: str, arg: Any = None) -> list:
        for p in self.parents:
            p.send((cmd, arg))
        return [self._recv_checked(p) for p in self.parents]

    def _broadcast_per(self, cmd: str, per_args: list) -> list:
        assert len(per_args) == self.num_envs
        for p, a in zip(self.parents, per_args):
            p.send((cmd, a))
        return [self._recv_checked(p) for p in self.parents]

    # ------------------------------------------------------------------
    # Batch I/O
    # ------------------------------------------------------------------

    def get_states(self) -> np.ndarray:
        return np.array(self._broadcast_same("get_state"))

    def set_actions(self, acts: np.ndarray) -> None:
        self._broadcast_per("set_action", [acts[i] for i in range(self.num_envs)])

    def get_muscle_torques(self) -> np.ndarray:
        return np.array(self._broadcast_same("get_muscle_torques"), dtype=np.float32)

    def get_desired_torques(self) -> np.ndarray:
        return np.array(self._broadcast_same("get_desired_torques"), dtype=np.float32)

    def set_activation_levels(self, acts: np.ndarray) -> None:
        self._broadcast_per(
            "set_activation_levels", [acts[i] for i in range(self.num_envs)]
        )

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def steps(self, n: int) -> None:
        self._broadcast_same("steps", int(n))

    def steps_at_once(self) -> None:
        self._broadcast_same("steps_at_once")

    # ------------------------------------------------------------------
    # Per-index queries (single round-trip to one worker)
    # ------------------------------------------------------------------

    def is_end_of_episode(self, idx: int) -> bool:
        self.parents[idx].send(("is_end_of_episode", None))
        return bool(self._recv_checked(self.parents[idx]))

    def get_reward(self, idx: int) -> float:
        self.parents[idx].send(("get_reward", None))
        return float(self._recv_checked(self.parents[idx]))

    def reset(self, rsi: bool = True, idx: Optional[int] = None):
        if idx is None:
            self._broadcast_same("reset", bool(rsi))
            return None
        self.parents[idx].send(("reset", bool(rsi)))
        return self._recv_checked(self.parents[idx])

    # ------------------------------------------------------------------
    # Muscle tuples
    # ------------------------------------------------------------------

    def gather_muscle_tuples(self) -> list:
        results = self._broadcast_same("gather_muscle_tuples")
        out: list = []
        for r in results:
            out.extend(r)
        return out

    # ------------------------------------------------------------------
    # P2 packed path
    # ------------------------------------------------------------------

    def update_muscle_nn(self, state_dict: dict) -> None:
        # Move state_dict to CPU before broadcasting (workers are CPU-only).
        cpu_sd = {k: v.detach().cpu() for k, v in state_dict.items()}
        payload = pickle.dumps(cpu_sd)
        self._broadcast_same("update_muscle_nn", payload)

    def run_control_step(self, actions: np.ndarray) -> list:
        """Packed path: scatter one action per worker, each worker runs the full
        muscle inner loop + physics substeps, returns (state, reward, eoe).
        """
        assert actions.shape == (self.num_envs, self.num_action), (
            f"actions shape {actions.shape} != ({self.num_envs}, {self.num_action})"
        )
        return self._broadcast_per(
            "run_control_step", [actions[i] for i in range(self.num_envs)]
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for p in self.parents:
            try:
                p.send(("close", None))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        for p in self.parents:
            try:
                p.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_vec_env(
    config_path: str,
    num_envs: int,
    mode: str = "subproc",
    seed: Optional[int] = None,
) -> MassVecEnv:
    """Factory. mode: 'serial' | 'subproc'."""
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    if mode == "serial" or num_envs == 1:
        return SerialVecEnv(config_path, num_envs)
    elif mode == "subproc":
        return SubprocVecEnv(config_path, num_envs, seed=seed)
    else:
        raise ValueError(f"Unknown vec_env mode: {mode!r} (choose 'serial' or 'subproc')")
