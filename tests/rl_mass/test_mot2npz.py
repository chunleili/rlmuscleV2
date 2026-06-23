"""mot2npz: parse a Storage .mot, select+reorder coords, deg->rad, write npz."""
import os, sys, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from tools.mot2npz import read_mot, mot_to_npz
from rl_mass.finger.trajectory_ref import TrajectoryRef

SAMPLE = """Coordinates
version=1
nRows=3
nColumns=3
inDegrees=yes

endheader
time\tr_shoulder_elev\tr_elbow_flex
0.0\t0.0\t90.0
0.5\t10.0\t90.0
1.0\t20.0\t90.0
"""

def main():
    d = tempfile.mkdtemp()
    mot = os.path.join(d, "x.mot")
    with open(mot, "w") as f:
        f.write(SAMPLE)
    cols, t, data, deg = read_mot(mot)
    assert cols == ["r_shoulder_elev", "r_elbow_flex"], cols
    assert deg is True
    assert t.shape == (3,) and data.shape == (3, 2), (t.shape, data.shape)
    assert abs(t[1] - 0.5) < 1e-12

    out = os.path.join(d, "x.npz")
    mot_to_npz(mot, out, coords=["r_shoulder_elev", "r_elbow_flex"],
               rotational={"r_shoulder_elev", "r_elbow_flex"})
    z = np.load(out)
    assert int(z["n_episodes"]) == 1
    assert abs(float(z["dt"]) - 0.5) < 1e-12
    q = z["episode_0_traj_qpos"]
    assert q.shape == (3, 2), q.shape
    # 90 deg -> pi/2 on the elbow column
    assert abs(q[0, 1] - np.pi / 2) < 1e-9, q[0, 1]
    # 10 deg -> 10*pi/180 on the shoulder column at t=0.5 (row 1)
    assert abs(q[1, 0] - np.deg2rad(10.0)) < 1e-9, q[1, 0]
    # TrajectoryRef loads it
    ref = TrajectoryRef(out, nq=2)
    assert ref.n_frames == 3 and ref.nq == 2
    print("OK")

if __name__ == "__main__":
    main()
