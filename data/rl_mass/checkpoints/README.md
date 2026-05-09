# MASS BVH Sweep — 收敛 ckpt 归档

来源：`docs/experiments/2026-04-26-mass-bvh-sweep.md` 实验结果。
保存策略：4 段达成收敛阈值（max_return ≥ 5.0 且 final_avg ≥ 0.5·max_return）的最佳 ckpt 拷贝。
不收敛的 4 段（run / cart / backflip / pirouette）未归档。

## ckpt 列表

| Motion | iter | max_return | final_avg | wall (h) | 文件 |
|--------|-----:|-----------:|----------:|---------:|------|
| walk | 1000 | 23.21 | 15.54 | 5.0 | `walk_max.pt` + `walk_max_muscle.pt` |
| balance | 1000 | 13.17 | 10.99 | 5.3 | `balance_max.pt` + `balance_max_muscle.pt` |
| kick | 5000 | 9.99 | 6.91 | 20.3 | `kick_max.pt` + `kick_max_muscle.pt` |
| dance | 5000 | 9.26 | 6.87 | 19.9 | `dance_max.pt` + `dance_max_muscle.pt` |

## 文件用途

- `<motion>_max.pt` — `SimulationNN` (policy + value)
- `<motion>_max_muscle.pt` — `MuscleNN` (desired-torque → 284 muscle activations)

## 加载方式

`examples/rl_mass/eval.py` 通过 `-m <prefix>` 加载，会拼接 `.pt` 与 `_muscle.pt`。
LFS 已拉取后（`git lfs pull`）直接用即可：

```bash
# 临时软链到任意 prefix（避免改名）
mkdir -p /tmp/walk_ckpt && \
    ln -sf $PWD/data/rl_mass/checkpoints/walk_max.pt        /tmp/walk_ckpt/x.pt && \
    ln -sf $PWD/data/rl_mass/checkpoints/walk_max_muscle.pt /tmp/walk_ckpt/x_muscle.pt

PYTHONPATH=. MUJOCO_GL=egl uv run python examples/rl_mass/eval.py \
    -d data/rl_mass/configs/walk.json -m /tmp/walk_ckpt/x \
    --video -o output/eval_walk.mp4
```

也可以直接传 ckpt 前缀（无需软链）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MUJOCO_GL=egl uv run python examples/rl_mass/eval.py \
    -d data/rl_mass/configs/walk.json \
    -m data/rl_mass/checkpoints/walk_max \
    --num-episodes 1 --max-steps 300 \
    --save-traj output/anim/walk_traj.npz
```

## 一键 eval + bake（USD/BVH motion 导出）

下游 Houdini 流水线只要 motion（rest skeleton + 每帧 root translation + per-joint rotation），不需要 LBS。脚本：

```bash
# 单个 motion
scripts/rl_mass_anim/eval_and_bake.sh walk

# 批量
scripts/rl_mass_anim/eval_and_bake.sh balance kick dance

# 指定 GPU / 步数
CUDA_VISIBLE_DEVICES=1 MAX_STEPS=600 scripts/rl_mass_anim/eval_and_bake.sh walk
```

每个 motion 输出四件套到 `$OUT_DIR`（默认 `output/anim/`）：

- `<motion>_traj.npz` — body world transforms + qpos（最后一回合）
- `<motion>_activations.csv` — 表头 `frame,<284 muscle name>`，每行一帧 float ∈ `[0, 1]`；列序 = MASS 默认顺序（见 `external/MASS/data/muscle284.xml`）；行数 `T_a = T_motion - 2`。可用 `NO_DIAG=1` 关闭。
- `<motion>_snu_ysy.usd` — UsdSkel SkelAnimation（绑 `snu_ysy_v3.usd` rest skeleton，Storm 可直接渲染）。同时把每帧激活写到每根肌肉 prim 上（`/Human/Muscles/<name>.muscle:activation` time-sampled float ∈ [0,1]，`NO_DIAG=1` 跳过）。最后 2 帧 hold-last（`T_diag = T_motion - 2`）。
- `<motion>_snu_ysy.bvh` — 标准 BVH（HIERARCHY + MOTION，ZXY intrinsic euler，Houdini Mocap Import SOP 直读）

可调环境变量：`MAX_STEPS`（默认 300）、`NUM_EPISODES`（默认 1，仅最后一回合 bake）、`STOCHASTIC=1`、`NO_DIAG=1`（跳过激活值导出）、`OUT_DIR`、`ASSET`、`CKPT_DIR`、`CFG_DIR`。

### 流程（脚本内部三步）

1. **`examples/rl_mass/eval.py` rollout** — 加载 `<motion>_max.pt`（`SimulationNN`）+ `<motion>_max_muscle.pt`（`MuscleNN`），`reset(rsi=False)` 从 BVH t=0 起跑，确定性 policy，每控制步保存 `qpos / body_xpos / body_xquat` → traj NPZ（`--save-traj`），同时 `--diag` 写完整诊断 NPZ（含 `a / l_m_norm / f_a / f_p / R / L / b / tau_des / tau_muscle / muscle_names / muscle_regions / muscle_f0 / t / reward`）到 `_diag_full.npz`。
2. **`examples/rl_mass_anim/bake_snu_ysy.py`** — 读 traj NPZ + `_diag_full.npz` + `snu_ysy_v3.usd` rest skeleton，FK 重定向（`snu` rest offset × MASS 世界旋转），写 USD SkelAnimation + BVH（HIERARCHY+MOTION，ZXY euler）。同时把 `a` 时间采样写到 `/Human/Muscles/<name>.muscle:activation`（284 muscle prims）。
3. **dump CSV 步骤** — 从 `_diag_full.npz` 取 `a + muscle_names`，写 `_activations.csv`（表头 `frame,<284 muscle name>`），删 `_diag_full.npz`。Excel/Houdini Table Import 直读。

### 已知坑

- 激活值 NPZ 中的列序固定 = MASS 训练时的肌肉顺序（`external/MASS/data/muscle284.xml` 文件解析顺序）。同一 ckpt 不同 motion / 同一 motion 不同次运行，列序保持一致。要查名字↔索引：随便加载一个 `_diag_full.npz` 之前的 NPZ 看 `muscle_names`，或临时去掉 `eval_and_bake.sh` 里的 strip 步骤。
- bake 必须用 `data/snu/snu_ysy_v3.usd`（含 `/Human/Joint_tree/Skeleton`），原始 `snu_ysy.usd` 缺 UsdSkel 节点，会报错。
- 当前 4 个 ckpt 中 `walk` 部署最稳（300 步无早停）；`balance / kick / dance` 部署后 1–2s 内会摔倒（root_y < 1.3 触发终止），baked 长度 41/45/55 帧。要更长片段需多回合采样或继续训练。
- `eval.py --save-traj` 只保留**最后一回合**的轨迹，`--num-episodes>1` 不会自动选 best。

## 续训复用

可作为新实验（如 phase-aware termination）的预训练种子：

```bash
PYTHONPATH=. uv run python examples/rl_mass/train.py \
    -d data/rl_mass/configs/walk.json \
    --nn_dir output/rl_mass/walk_v2/nn \
    -m max  # 需先把 data/rl_mass/checkpoints/walk_max.pt → output/rl_mass/walk_v2/nn/max.pt
```

## 说明

- 每个 motion 对应固定 BVH `external/MASS/data/motion/<motion>.bvh`，骨架 `data/rl_mass/humanoid.xml`，肌肉 `external/MASS/data/muscle284.xml`。
- 同一 ckpt 用于不同 BVH 不会有意义（policy 学的是特定动作的相位 → action 映射）。
- 训练超参与 reward 配置：见 `data/rl_mass/configs/<motion>.json` + `examples/rl_mass/train.py` 默认值。
