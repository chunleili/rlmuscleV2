"""Diagnose which verts spike in the Millard run and what they are."""
import os, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "src"))
import warp as wp
from muscle.util.util import load_config
from muscle.muscle_warp import MuscleSim

wp.init(); wp.set_device("cpu")
cfg = load_config("data/muscle/config/bicep.json")
cfg.gui = False; cfg.render_mode = None
sim = MuscleSim(cfg)

rest = sim.pos0_np.copy()
tm = sim.v_tendonmask_np.copy() if sim.v_tendonmask_np is not None else np.zeros(len(rest))
print("n_verts", len(rest), "tendonmask: min/mean/max",
      tm.min(), tm.mean(), tm.max(), " frac>0.5", (tm > 0.5).mean())

# load the deformed millard mesh, find top-displaced verts
import meshio
m = meshio.read(os.path.join(os.path.dirname(__file__), "mesh_fibermillard.ply"))
pos = m.points
d = np.linalg.norm(pos - rest, axis=1)
order = np.argsort(d)[::-1][:20]
lo, hi = rest.min(0), rest.max(0)
axis = int(np.argmax(hi - lo))  # long axis of muscle
print(f"\nlong axis = {axis}; rest range on it [{lo[axis]:.3f}, {hi[axis]:.3f}]")
print("\ntop-20 displaced verts (millard):  vert  disp  tendonmask  axispos(0=lo,1=hi)")
for i in order:
    apos = (rest[i, axis] - lo[axis]) / (hi[axis] - lo[axis])
    print(f"  {i:5d}  {d[i]:.4f}   tm={tm[i]:.3f}   axis={apos:.2f}")
# how does displacement correlate with tendonmask?
print(f"\nmean disp where tm<0.5 (belly): {d[tm<0.5].mean():.4f}  max {d[tm<0.5].max():.4f}")
print(f"mean disp where tm>0.5 (tendon): {d[tm>0.5].mean():.4f}  max {d[tm>0.5].max():.4f}")
