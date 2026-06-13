# Code for "RLMuscle: Physics-Based Muscle-Driven Character Animation via Reinforcement Learning"

## Examples

All examples default to writing a **USD mesh sequence** under `output/`.
Run from the repo root with `PYTHONPATH=src` (commands below already include it).

<table>
  <tr>
    <td colspan="3"><h3>SDF Collision</h3></td>
  </tr>
  <tr>
    <td colspan="3">
      Per-vertex analytic / mesh SDF push-out. Simple, fast, stable — but
      <b>no self-intersection handling</b> and contacts are not perfectly erased.
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="src/muscle/collision/sdf/example/u_drop.py">
        <img width="320" src="docs/images/sdf_u_drop.png" alt="SDF U drop">
      </a><br><b>U drop (soft vs plane)</b>
    </td>
    <td align="center" width="33%">
      <a href="src/muscle/collision/sdf/example/rigid_soft.py">
        <img width="320" src="docs/images/sdf_rigid_soft.png" alt="SDF rigid solid">
      </a><br><b>Rigid sphere vs soft body</b>
    </td>
    <td align="center" width="33%">
      <a href="src/muscle/collision/sdf/example/humanV2.py">
        <img width="320" src="docs/images/sdf_humanv2.png" alt="SDF muscle-bone">
      </a><br><b>humanV2 muscle vs animated bone</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>PYTHONPATH=src python -m muscle.collision.sdf.example.u_drop --config data/collisionTest/sdf_drop.json --shape plane</code>
    </td>
    <td align="center" width="33%">
      <code>PYTHONPATH=src python -m muscle.collision.sdf.example.rigid_soft --config data/collisionTest/rigid_soft.json</code>
    </td>
    <td align="center" width="33%">
      <code>PYTHONPATH=src python -m muscle.collision.sdf.example.humanV2 --config data/humanV2/config/humanV2_gmg.json --bone_dir data/humanV2/model/bone --end_frame 20</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%"><sub>→ <code>output/sdf_drop.usd</code></sub></td>
    <td align="center" width="33%"><sub>→ <code>output/rigid_soft.usd</code></sub></td>
    <td align="center" width="33%"><sub>→ <code>output/humanV2_bone_anim.usd</code></sub></td>
  </tr>
  <tr>
    <td colspan="3"><h3>DCD Collision</h3></td>
  </tr>
  <tr>
    <td colspan="3">
      Faithful Warp port of Gaia_UDCF triangle-triangle discrete collision —
      <b>handles self-collision</b> (U arms fold and press without crossing).
      Material is Gaia's MassSpring (XPBD springs + hard volume + inversion).
      DCD cannot perfectly erase contacts either, which requiring CCD. But most penatrations are shallow.
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="src/muscle/collision/udcf/example/u_drop.py">
        <img width="320" src="docs/images/dcd_u_drop.png" alt="DCD U drop">
      </a><br><b>U drop (self-collision)</b>
    </td>
    <td align="center" width="33%"></td>
    <td align="center" width="33%"></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>PYTHONPATH=src python -m muscle.collision.udcf.example.u_drop --steps 60 --init_vel_y 0</code>
    </td>
    <td align="center" width="33%"></td>
    <td align="center" width="33%"></td>
  </tr>
  <tr>
    <td align="center" width="33%"><sub>→ <code>output/udcf_u_drop.usd</code></sub></td>
    <td align="center" width="33%"></td>
    <td align="center" width="33%"></td>
  </tr>
</table>



## Art assets

Large `model/` geometry is not in git — it lives in the private HF dataset
[`chunleili/rlmuscleV2-assets`](https://huggingface.co/datasets/chunleili/rlmuscleV2-assets)
as `<case>/model.zip`, extracting to `data/<case>/model/`.

```bash
pip install huggingface_hub
hf auth login                  # read token; ask owner for repo access (private)
python tools/fetch_assets.py   # download all cases (or pass case names)
```

Upload (maintainer): `hf upload chunleili/rlmuscleV2-assets data/<case>/model.zip <case>/model.zip --repo-type dataset`
