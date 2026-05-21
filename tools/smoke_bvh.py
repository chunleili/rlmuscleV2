"""Torch-free smoke test for a retargeted BVH.

Loads the BVH through Character + MuJoCoEngine (the same path env.py uses),
samples target qpos across the cycle, pushes each into MuJoCo and checks the
pelvis stays finite and at a plausible height.

    PYTHONPATH=src .venv/bin/python tools/smoke_bvh.py data/rl_mass/motion/overhead-squat.bvh
    PYTHONPATH=src .venv/bin/python tools/smoke_bvh.py data/rl_mass/motion/preroll.bvh native
"""
import sys
import numpy as np
from rl_mass.engine import MuJoCoEngine
from rl_mass.character import Character
from rl_mass.joint_utils import MASS_NATIVE_BVH_MAP

path = sys.argv[1] if len(sys.argv) > 1 else "data/rl_mass/motion/overhead-squat.bvh"
naming = sys.argv[2] if len(sys.argv) > 2 else "character1"
bvh_map = MASS_NATIVE_BVH_MAP if naming == "native" else None

engine = MuJoCoEngine("data/rl_mass/humanoid.xml", timestep=1.0 / 600)
char = Character(engine, kp=300.0)
char.load_bvh(path, cyclic=True, bvh_map=bvh_map)
char.reset()

print("parsed:", path)
print("  model nq=%d nv=%d   bvh max_time=%.4fs" %
      (engine.nq, engine.nv, char.bvh.max_time))

dt_ctrl = 1.0 / 30.0
max_time = char.bvh.max_time
ok = True
ys = []
for frac in (0.0, 0.25, 0.5, 0.75, 0.99):
    char.reset()
    t = frac * max_time
    p = char.get_target_positions_mujoco(t, dt_ctrl)
    finite = bool(np.all(np.isfinite(p)))
    engine.set_qpos(p)
    engine.forward()
    py = engine.get_body_xpos(engine.body_id("Pelvis"))[1]
    ys.append(py)
    flag = "ok" if finite and 0.3 < py < 1.6 else "SUSPECT"
    if flag != "ok":
        ok = False
    print("  t=%5.2fs  qpos[%d] finite=%s  pelvis_y=%.3f  %s" %
          (t, p.size, finite, py, flag))

print("  pelvis_y range: %.3f .. %.3f" % (min(ys), max(ys)))

# SPD with zero action must produce zero root torque (sanity on joint mapping).
char.reset()
p0, v0 = char.get_target_pos_and_vel_mujoco(0.0, dt_ctrl)
engine.set_qpos(p0)
engine.set_qvel(v0)
engine.forward()
tau = char.get_spd_forces(engine, p0)
root_zero = bool(np.allclose(tau[:6], 0.0))
print("  SPD root torques zero: %s   max non-root torque=%.3f" %
      (root_zero, float(np.max(np.abs(tau[6:])))))

print("SMOKE", "OK" if (ok and root_zero) else "FAILED")
sys.exit(0 if (ok and root_zero) else 1)
