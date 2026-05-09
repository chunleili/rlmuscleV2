# V2 初始整理 + coupling 迁移

## data/ 清理

- 删 `data/configs/`（与 `data/rl_mass/configs/` 重复，且路径过期）
- 删 `data/checkpoints/`（与 `data/rl_mass/checkpoints/` 完全相同，约 37 MB）
- 删 `data/config.json`、`data/humanoid.xml`（未被引用，rl_mass 子树下已有）
- 从 RLMuscle 拷入 `data/Human/`（Human.usd 41 MB + human_import2.json + data_prepare.hip）
- 删 `data/muscle/config/bicep_xpbd_millard.json`（explicit-force 对照基线，已被 RLMuscle 实验否决）

## Git 与 LFS

- `git init`，分支 `main`
- `.gitattributes` 追踪：`*.usd/usda/usdc/usdz`、`*.pt/pth/ckpt`、`*.hip/hiplc/hipnc`、`*.bvh/fbx/npz`
- `.gitignore` 忽略 `.venv/`、`output/`、`log.md`、`smoke_test*.py`、`run_1k_gpu.py`、`src/rl_mass/verify/` 等

## 选型：example_couple3 + 默认 TETFIBERNORM

调查 RLMuscle 三个 coupling example：

| Example | 后端 | 默认约束 | V2 适配 |
|---|---|---|---|
| `example_couple` | Taichi + Newton | (taichi PBD) | ❌ V2 无 taichi |
| `example_couple2` | Warp + Newton | TETFIBERNORM | ✅ |
| `example_couple3` | Warp + Newton | TETFIBERMILLARD | ✅ + CUDA graph + surface mesh |

couple2 与 couple3 的物理逻辑由 config 决定，example 文件只在外围设施有差。**采用 couple3，重命名为 `example_couple.py`**——拿到 CUDA graph 与 surface-mesh 渲染，同时支持 `--config` 一行切换两种约束。

## 迁移：`src/muscle/couple/` 子包

新增 7 个文件：

| 文件 | 行数 | 来源 | 备注 |
|---|---|---|---|
| `bicep_helpers.py` | 111 | RLMuscle 同名 | ELBOW_AXIS/PIVOT, build_elbow_model |
| `solver_muscle_bone_coupled_warp.py` | 296 | 同上 | SolverMuscleBoneCoupled |
| `controllability.py` | 404 | 同上 | 耦合参数 + sweep eval（solver 也用） |
| `mesh_utils.py` | 95 | 同上（精简自 487 行） | 只保留 `MeshDistortionError`、`check_mesh_quality`、`fix_tet_winding` |
| `usd_io.py` | 574 | 同上 | UsdIO、build_bone_prim_map |
| `activation.py` | 72 | 同上 | activation_dynamics_step_np（controllability 依赖） |
| `example_couple.py` | 332 | RLMuscle `example_couple3.py` | DEFAULT_CONFIG 改为 `bicep_coupled.json`（首选 norm 路径） |

`__init__.py` 留空，包路径 `muscle.couple.*`。

## 现有文件补丁

- `config.py`：`_PATH_FIELDS` 加 `ground_mesh_path`、`coord_mesh_path`
- `muscle_common.py`：`_load_bone` 形参重命名 `bone_usd_path → target_path`，新增缓存方法 `load_bone_geo()`
- `constraints.py`：`create_attach_constraints` / `create_distance_line_constraints` 在头部加 `target_path = params.get('target_path')` + `self.bone_geo, self.bone_pos = self.load_bone_geo(target_path)`

## 依赖

`pyproject.toml` 加 `newton[sim]`，source 锁到 GitHub 主分支：

```toml
newton = { git = "https://github.com/newton-physics/newton.git", branch = "main" }
```

> PyPI 上的 `newton==0.2.0` 存在 API 漂移（SolverMuJoCo 在 add_joint 阶段抛 `stiffness should be a numeric scalar or list`），必须用 GitHub 源版（实测 commit `6648d39`，自报版本 `1.2.0.dev0`）。

## 1000 步烟囱测试

```bash
PYTHONPATH=src .venv/bin/python -m muscle.couple.example_couple --auto --steps 1000 \
  [--config data/muscle/config/<config>.json]
```

| Config | 路径 | 峰值 q (rad) | 峰值 τ (N·m) | 备注 |
|---|---|---:|---:|---|
| `bicep_coupled.json` | TETFIBERNORM | 0.107 | 0 | 几乎无屈曲——与 RLMuscle approach-comparison 文档结论一致（mesh 退化 + position-based 软约束力量不足） |
| `bicep_fibermillard_coupled.json` | TETFIBERMILLARD | -3.000（撞下限）| 20.0（max_torque） | 完整屈伸周期 |

USD 输出 `output/example_couple.anim.usd`（16 MB）。

**结论**：Millard 才是可用路径，建议把 `example_couple.py::DEFAULT_CONFIG` 切回 `bicep_fibermillard_coupled.json`。

## 后续

- 切默认 config 到 Millard
- `archived/bicep_fibermillard.json`、`archived/bicep_hybrid.json` 待评估是否删除
- `millard_curves.py` 现仅在 TETFIBERMILLARD 路径触发，不调用则零开销，保留
