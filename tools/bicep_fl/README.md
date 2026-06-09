# Bicep force-length: Millard constraint vs plain F@d, vs OpenSim

Proves the `fibermillard` constraint reproduces the OpenSim Millard 2012
force-length law, while the plain MASS "F@d" line-muscle (force = `f * direction`,
Gaussian `g_al` / exponential `g_pl`) is a different active force-length profile.

Constitutive-curve comparison: normalized force vs normalized fiber length
`lambda = l_m / l_opt`, at full activation `a = 1` (multiply by `F_max` for Newtons).

## Three laws compared

| Curve | Source |
|-------|--------|
| OpenSim (reference) | `ActiveForceLengthCurve` / `FiberForceLengthCurve`, default Millard2012 |
| Millard (V2 `fibermillard`) | `MillardCurves().fl / .fpe` in `src/muscle/muscle_warp.py` |
| F@d plain fiber (MASS) | `g_al` / `g_pl` in `src/rl_mass/muscle.py` |

## Run

```bash
# 1. OpenSim reference (needs the `opensim` conda env, OpenSim 4.6)
conda run -n opensim python tools/bicep_fl/opensim_ref.py   # -> opensim_ref.csv

# 2. Compare + plot (project uv env)
uv run python tools/bicep_fl/compare.py                      # -> bicep_fl_compare.png/.csv
```

## Result

```
Millard(V2) active vs OpenSim:  RMSE=0.0000  max|err|=0.0000   <- exact match
F@d (MASS)  active vs OpenSim:  RMSE=0.2171  max|err|=0.4899   <- clearly different
```

At the limbs the models disagree most: `f_act(lambda=0.44)` is 0.10 (OpenSim/Millard)
vs 0.54 (F@d); `f_act(lambda=1.81)` is 0.10 vs 0.27. The MASS Gaussian `g_al` is
wider and never reaches the narrow Millard plateau / steep descending limb, and its
passive `g_pl` onset differs from the Millard `f_PE` curve.

Conclusion: the Millard fiber constraint is NOT equivalent to the plain F@d fiber
model; it matches the OpenSim biomechanics reference to machine precision.

## Mesh (visual) comparison

Same bicep tet mesh + attaches, same activation ramp to a=1, only the active
fiber constraint differs (`fiber` plain vs `fibermillard`).

```bash
uv run python tools/bicep_fl/mesh_compare.py --nsteps 600
```

Outputs `mesh_compare.png` (3-panel, colored by displacement) and PLY meshes
`mesh_rest.ply`, `mesh_fiber.ply`, `mesh_fibermillard.ply` (open in MeshLab/Blender).

Activation is a **step input held at a=1** for all steps (rest = a=0). Do NOT use
`activation_ramp()` for this -- it is a `0->0.5->1->0.7->0.3->0` excitation profile
that ends at 0, which would sample the relaxed muscle.

Stability: `num_substeps=100` (each substep is one XPBD jacobi sweep -- `step()`
has no inner solver loop, so more substeps = more jacobi iterations = a converged,
non-distorted mesh). No post-smooth / tet-repair (those artificially modify the
geometry). Applied to BOTH runs so only the constraint type differs.

```
plain F@d   displacement from rest:  mean=0.0131  max=0.0324
Millard     displacement from rest:  mean=0.0279  max=0.0840   (~20% of bbox diag)
F@d vs Millard vertex difference:    mean=0.0157  max=0.0560   (mesh bbox diag 0.41)
```

The Millard constraint contracts/bulges the belly ~3x more than the plain F@d
constraint. Force scales differ (plain `stiffness` vs Millard `sigma0=3e5` Pa), so
this shows behavioral divergence, not a calibrated force match. A few tets still
spike at the tendon-end attach -- a stress concentration against the rigid (1e37)
end attach, not solver under-convergence; the muscle belly itself is smooth.
