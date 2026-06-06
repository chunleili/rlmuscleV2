"""TrajectoryRef serves interpolated qpos/qvel from a musclemimic .npz."""
import numpy as np
from rl_mass.trajectory_ref import TrajectoryRef

NPZ = "/home/chunleli/Dev/musclemimic/trajectory_data_bimanual/bimanual_episodes_mjx_20260601_220203.npz"


def main():
    ref = TrajectoryRef(NPZ, nq=76)
    assert ref.max_time > 0.0
    q0, v0 = ref.pos_vel(0.0, ref.dt)
    assert q0.shape == (76,), q0.shape
    assert v0.shape == (76,), v0.shape
    # Interpolation at a mid frame matches a manual lerp of stored frames.
    qa, _ = ref.pos_vel(0.0, ref.dt)
    qb, _ = ref.pos_vel(ref.dt, ref.dt)
    qmid, _ = ref.pos_vel(0.5 * ref.dt, ref.dt)
    assert np.allclose(qmid, 0.5 * (qa + qb), atol=1e-5), np.abs(qmid - 0.5 * (qa + qb)).max()
    # Velocity is the finite difference used by the controller.
    assert np.allclose(v0, (qb - qa) / ref.dt, atol=1e-5)
    print("OK", round(ref.max_time, 3), ref.n_frames)


if __name__ == "__main__":
    main()
