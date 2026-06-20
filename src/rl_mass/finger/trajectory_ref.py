"""Reference motion from a musclemimic .npz trajectory.

The .npz stores one or more episodes, each as `episode_{i}_traj_qpos`
of shape (frames, nq) already in the target MuJoCo qpos order. We
concatenate all episodes into one frame sequence and serve interpolated
qpos / finite-difference qvel at a query time, matching how the existing
Character feeds (target_pos, target_vel) into SPD and the reward.
"""
from __future__ import annotations

import numpy as np


class TrajectoryRef:
    def __init__(self, npz_path: str, nq: int):
        data = np.load(npz_path, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        frames = []
        for i in range(n_eps):
            key = f"episode_{i}_traj_qpos"
            if key not in data:
                continue
            arr = np.asarray(data[key], dtype=np.float64)
            assert arr.shape[1] == nq, (key, arr.shape, nq)
            frames.append(arr)
        assert frames, f"no episode_*_traj_qpos in {npz_path}"
        self.frames = np.concatenate(frames, axis=0)  # (N, nq)
        self.n_frames = self.frames.shape[0]
        self.nq = nq
        self.dt = float(data["dt"])
        # max_time is the last sample time; queries wrap modulo this.
        self.max_time = self.dt * (self.n_frames - 1)

    def _qpos_at(self, t: float) -> np.ndarray:
        """Linearly interpolated qpos at time t (clamped to [0, max_time])."""
        if self.max_time <= 0.0:
            return self.frames[0].copy()
        t = max(0.0, min(t, self.max_time))
        f = t / self.dt
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, self.n_frames - 1)
        w = f - i0
        return (1.0 - w) * self.frames[i0] + w * self.frames[i1]

    def pos_vel(self, t: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (qpos(t), qvel(t)) with qvel = (qpos(t+dt)-qpos(t))/dt."""
        q = self._qpos_at(t)
        q1 = self._qpos_at(t + dt)
        return q, (q1 - q) / dt
