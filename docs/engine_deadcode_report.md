# engine.py 冗余代码报告

方法：给 `wp.launch` 打 hook（记录实际派发的 kernel）+ `sys.setprofile`（记录实际执行的 engine.py Python 方法），逐个跑完全部 demo，取并集。再用全仓 grep（src/tests/tools/data）静态交叉验证。

跑过的 demo（短帧）：bicep、oneway(lion/humanV2)、sdf(u_drop/rigid_soft/humanV2)、udcf u_drop。

经验结论：
- `MuscleSim`（engine.py 内）是唯一真正全量走 engine.py 的路径（bicep + sdf 例子）。
- `OneWaySim`(coupleOneWayGmg/solver.py)、`MassSpringSim`(udcf/massspring_sim.py) 只继承 `XpbdEngine` 基类 + 借用 `ssvd_fn`/`precompute_rest_kernel`，自带各自 solver kernel。所以多数 engine.py 代码这些 demo 根本不碰。

按"是否还有任何引用"分三档。

---

## Tier 1 — 真死（无 demo、无 config、无 tool、无 test 引用）

整簇删除安全。约 ~700 行。

### snh 簇（无任何 config `"type":"snh"`）
| 符号 | 类型 | 行 |
|---|---|---|
| `TETSNH` | const | 46 |
| `create_tet_snh_constraint` | method | 345–380 |
| `elif ctype == 'snh':` build 分支 | | 677–678 |
| `tet_snh_update_xpbd_fn` | @wp.func | 2569–2724 |
| `solve_tetsnh_kernel` | @wp.kernel | 2727–2752 |
| `elif ctype == TETSNH:` dispatch 分支 | | 2938–2943 |

### distanceline 簇（无任何 config）
| 符号 | 类型 | 行 |
|---|---|---|
| `DISTANCELINE` | const | 41 |
| `create_distance_line_constraints` | method | 557–616 |
| `self.distanceline_constraints = []` | | 648 |
| `elif ctype == 'distanceline':` build 分支 | | 666–668 |
| `elif ctype == DISTANCELINE:` (update_attach_targets_kernel) | | 1852–1857 |
| `solve_distanceline_kernel` | @wp.kernel | 2272–2306 |
| `elif ctype == DISTANCELINE:` dispatch 分支 | | 2887–2891 |
| `... distanceline_constraints ...` 条件项 | | 3666 |

### pin 簇（无任何 config，无 create 方法）
| 符号 | 类型 | 行 |
|---|---|---|
| `PIN` | const | 36 |
| `solve_pin_kernel` | @wp.kernel | 2216–2245 |
| `elif ctype == PIN:` dispatch 分支 | | 2877–2881 |

注：`distance_pos_update_xpbd_fn`(@wp.func L1215) 同时被 distanceline 和 pin 两个 kernel 调用——两簇都删后才可删它，否则保留。

### 孤儿工具（全仓零引用）
| 符号 | 类型 | 行 |
|---|---|---|
| `fill_vec3_kernel` | @wp.kernel | 819–821 |
| `update_cons_restdir1_kernel` | @wp.kernel | 854–867 |
| `compute_tet_fiber_rest_length` | method | 88–99 |
| `get_fps` | method | 3190–3194 |
| `rebuild_constraints` | method | 3304–3343 |
| `get_insertion_force` | method | 3709–3725 |
| `save_predicted_positions` | method | 3704–3707 |
| `set_lbs_T_rel` | method | 3727–3736 |
| `apply_lbs` | method | 3738–3748 |
| `lbs_apply_kernel` | @wp.kernel | 1742–1763 |

`fill_float_kernel` 有用（example.py/tools），`fill_vec3_kernel` 无用。
LBS 簇（set_lbs_T_rel/apply_lbs/lbs_apply_kernel）整条无人调，coupleTwoWay 也不用。

---

## Tier 2 — 已接线但所有 config 都禁用的旋钮

按"无 demo 用即冗余"算冗余，但属"接好但被关掉"的 plumbing，不是孤儿。删除即移除特性。

### DGF 簇（无任何 config `"type":"fiberdgf"`）
| 符号 | 位置 |
|---|---|
| `src/muscle/fiber/dgf.py` 整模块 | dgf_active_force_length_wp / dgf_passive_force_length_wp / dgf_force_velocity_wp |
| engine.py import 块 | 18–22 |
| `TETFIBERDGF` const | 45 |
| `fem_flags_fn` 里 `or ctype == TETFIBERDGF` | 900（只删这一项，保留 TETFIBERNORM） |
| `create_tet_fiber_dgf_constraint` | 269–305 |
| `elif ctype == 'fiberdgf':` build 分支 | 673–674 |
| `solve_tetfiberdgf_kernel` | 1944–2012 |
| `elif ctype == TETFIBERDGF:` dispatch 分支 | 2904–2912 |
| `_FIBER_TYPES` frozenset 去掉 TETFIBERDGF | 2986 |

牵连：`dgf_force_velocity_wp` 被 `accumulate_active_fiber_force_kernel`(L1680) 调用。

### legacy 显式 fiber 力 / sigma0 路径
所有 config 顶层 `sigma0: 0.0`。`bicep_fibermillard_coupled.json` 的 `_note` 直接写明这是 "legacy explicit fiber force kernel ... disabled"。
| 符号 | 行 |
|---|---|
| `accumulate_active_fiber_force_kernel` | 1584–1696 |
| `accumulate_active_fiber_force` (method) | 3676–3702 |

### max_accel 力钳
所有 config `max_accel: 0.0`，不可达。
| 符号 | 行 |
|---|---|
| `_clamp_force_by_accel_kernel` | 1399 |
| `accumulate_active_fiber_force` 内 max_accel>0 分支 | 3700 附近 |

注：`_freeze_near_inverted_kernel`(L1529) 受 `cfg.freeze_detF_threshold>0` 且仅 CPU——同属未启用旋钮，归此档。

---

## Tier 3 — 有自带 config + 验证 tool，只是非 README demo

非 demo，但有 working config + 工具。删除会破坏 `data/muscle/config/*.json` 与 `tools/bicep_fl/`。不算冗余。

| 特性 | config | tool |
|---|---|---|
| `fibermillard`（Millard 2012）| `data/muscle/config/bicep_fibermillard_coupled.json` | `tools/bicep_fl/*`（compare/diag/mesh_compare + csv）|
| `pintotarget` | `bicep_drive_pintotarget.json`, `fullbody_drive.json` | — |
| `tetarapsvd` | `bicep_drive_pintotarget.json` | — |

涉及：`create_tet_fiber_millard_constraint`、`solve_tetfibermillard_kernel`、`_init_millard_curves`、`_upload_curve`、`fiber/millard.py`、`create_pintotarget_constraints`、`solve_pintotarget_kernel`、`_mark_pintotarget_stopped`、`solve_tetarap_svd_kernel` 等。

---

## 确认保留（条件触发，配置可达）

- `_build_colored_gs`：`build_constraints` 在 `use_colored_gs` 时直接调（L3469）。
- 这些非死，只是短 demo 路径或 setprofile 未覆盖。

---

## 数据附录

各 demo 实际派发 kernel / 执行 func 数：
- bicep: 14 kernels, 59 funcs（最全）
- sdf rigid_soft: 12, 44 ; sdf u_drop: 9, 44 ; sdf humanV2: 8, 10
- udcf u_drop: 12, 10
- oneway lion: 3, 8（仅基类）; oneway humanV2: 6, 8

全 config 出现的 `"type"` 值：attach / volume / tetarap / fiber / pintotarget / tetarapsvd / fibermillard。
缺席（无任何 config）：snh / fiberdgf / distanceline / pin。
