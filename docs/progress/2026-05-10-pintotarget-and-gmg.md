# Pintotarget hard pin + Geometric Multigrid (GMG)

## 背景

参考 RLMuscle 的 `realtimeMuscle/`,V2 需要补两件事:
1. **Pintotarget**: 把骨头驱动的 fine vert 当 kinematic 锚点直接覆写位置(硬 pin),取代之前
   软 attach 在大形变下被 vol/arap dP 拽走的问题。
2. **GMG**: coarse 网格上跑几步 XPBD 把 fine 的低频体积误差吸收掉,fine 只处理高频细节。

## Pintotarget (hard pin)

约束新类型 `PINTOTARGET = 771624311`(见 `src/muscle/constraints.py`)。

### 数据布局

构造期 (`create_pintotarget_constraints`):
- `pts[0] = src` (fine vert id), `pts[2] = tgt` (bone vert id)
- `restdir = pos0[src] - bone_pos0[tgt]` (rest 偏移,世界坐标)
- `restvector` 在每帧 `update_attach_targets` 里被刷新成当前 `bone_pos[tgt]`

### 求解器 (`muscle_warp.py`)

新 kernel `solve_pintotarget_kernel` 直接位置覆写:
```
pos[src] = bone_pos + rest_offset
```

⚠️ **不能跟 vol/arap 一起在 jacobi 里算** —— `apply_dP` 会把累加 dP 叠在 pin 之后,
撤销 snap 导致 NaN(实测在 bicep step 50 NaN)。修法:
1. 从 `_dispatch_constraints` 里跳过 PINTOTARGET
2. 单独 `apply_pintotarget()` 在 `apply_dP → repair_inverted_tets` 之后、
   `update_velocities` 之前作为 kinematic 后处理跑

`_mark_pintotarget_stopped()` 把 src verts 的 `stopped=1`,跳过 integrate
(否则积分会让 pin 点漂移一帧再被 snap,产生 GMG 假信号)。

### 配置示例

```json
{
  "type": "pintotarget",
  "name": "muscleEndPin",
  "mask_name": "muscleendmask",
  "mask_threshold": 0.75
}
```

无 stiffness/dampingratio 字段 —— 硬约束没有刚度概念。

### 结果

- bicep 200 步:vol_err 比软 attach 略好(step 200:+0.006 vs -0.014)
- fullbody 200 步:全程更稳(step 150 拐点:-0.25 vs -0.61)

## GMG: 独立 CoarseSolver 辅助类

### 架构

- `src/muscle/coarse_solver.py`:
  - `CoarseSolver` 持有自己的 Warp 状态 (pos/pprev/dP/mass/tendonmask/stopped/cons)
    + fine↔coarse bary 映射,只跑 vol + tetarap
  - `MultigridMuscleSim(MuscleSim)` 在 `step()` substep 内注入:
    ```
    integrate(fine)
    coarse.restrict(fine.pos)
    coarse.solve(coarse_iters)
    coarse.prolong(fine.pos, fine.stopped)
    fine constraints + apply_dP + post_smooth + repair + pintotarget
    update_velocities(fine)
    ```

`example_drive.py` 检测 `cfg.coarse_geo_path`,有就实例化 `MultigridMuscleSim`,
没有就用普通 `MuscleSim`。

### Restrict / Prolong

最初做 delta 形式 (`Δcoarse = Σ bary · Δfine`) 但发现:fine 的运动几乎全集中在
pintotarget 锚点上,而我们 prolong 跳过了 stopped 点 —— 结果 coarse 看到的是
"周围 vert 没动" 的虚假信号,GMG 直接失效。

改为**绝对版**:
- `restrict`: `coarse.pos = interp_in_fine_cage(fine.pos)`
- `prolong`: `fine.pos = blend × interp_in_coarse_cage(coarse.pos) + (1-blend) × fine.pos`,跳过 stopped

每 substep `coarse.pprev := coarse.pos` 在 restrict 之前快照,
保证 XPBD 的 pprev/pos 在 coarse solve 内一致。

### Coarse 网格生成 (`tools/build_coarse_per_muscle.py`)

最初用户给的 body-envelope `coarse.geo` 是错域 —— 94% coarse vert 落在
fine tet 外部,bary 严重外插,反馈放大 → NaN。

正确做法:**每块肌肉独立 remesh**。
1. 从每块 fine TetMesh 抽出 surface tris (`build_surface_tris`,
   收集每个三角面只出现一次的边界)
2. `pymeshlab.meshing_decimation_quadric_edge_collapse(targetfacenum=24,
   preservenormal=True, preserveboundary=True, optimalplacement=True)`
3. `tetgen.TetGen(...).tetrahedralize(switches="pYq{q}O5a{max_vol}")`
4. 拼接所有肌肉,带 `muscle_id` 标签

GH200 aarch64 上 gmsh 没有 wheel,改用 `tetgen` Python 包。
依赖在 `pyproject.toml` 的 `mesh-coarsen` extra:
```toml
mesh-coarsen = ["pymeshlab>=2025.7", "tetgen>=0.8"]
```

154/199 块成功:其余薄片肌(rectusAbdominis、pectoralisMajor、
serratusAnterior 等)在 decimation 后非流形,tetgen 报
`Failed to tetrahedralize. You may need to repair surface by making it manifold`。
可接受。

输出 `data/Human/coarse_per_muscle.npz`:2037 verts / 2525 tets / 154 muscles
(平均 16.4 tets/muscle)。`tools/coarse_to_usd.py` 写 USD 可视化检查。

### Bary mapping (`compute_barycentric_mapping`)

KDTree 种子 + k=20 候选 + snap-and-clamp fallback:
- 候选里不命中(bary 不在 [0-eps, 1+eps] 单纯形)的 query,落在最近 tet 上
- 把 bary 三个分量 clip 到 [0,1],再按和 ≤ 1 等比缩放
- **绝对不外插**(防止反馈放大)

`strict_inside=True` 仅在 coarse→fine 映射时使用;实测 strict 模式跑出 1514/2037
unmatched(decimation 漂移),改成 snap-and-clamp 后 0 unmatched。

## Fine → Coarse 物性 gather

⚠️ 之前 coarse 的 `mass` 用 `density × coarse_volume / 4` 算,完全不看 fine,导致
"物理失衡":fine 的 heavy belly / tendon end 在 coarse 上变成均匀分布。

`CoarseSolver._gather_mass_from_fine`:
- 对每个 fine vert v: 取 `fine_in_coarse_idx[v] = T` 和 bary `(1-u-v-w, u, v, w)`
- `coarse_mass[T[k]] += fine_mass[v] · bary[k]`,k=0..3
- 总质量守恒(bary sum=1 由 snap-and-clamp 保证)

`CoarseSolver._gather_tendonmask_from_fine`:
- 对每个 fine tet,用其重心找所属 coarse tet
- 按 fine tet volume 加权累积 tendonmask,再除以总 volume

### 结果

```
[CoarseSolver] mass: gathered from fine, Σ_fine=36.2007 → Σ_coarse=36.2008
               (ratio=1.0000, floored 3 verts at 6.67e-05)
[CoarseSolver] tendonmask: gathered from fine (4 coarse tets received no fine tet)
```

### vol_err 对比 (200 步,fullbody,coarse_iter=10):

| step | geometric mass | gathered mass |
|------|----------------|---------------|
| 1    | -0.488         | **-0.459**    |
| 50   | -0.609         | **-0.572**    |
| 100  | -0.682         | **-0.637**    |
| 150  | -0.347         | -0.544        |
| 200  | -0.488         | -0.561        |

峰值阶段 (step 50/100) 改善 ~0.04~0.05。松弛阶段反而稍劣 —— gather 后
coarse 等效质量更大(承担了真实 fine 总质量 36 kg 而不是几何估算),回弹更慢。
端到端 21 秒,稳定无 NaN。

### 没做

- coarse 端 stiffness 当前是 config 全局常量 —— gather 等价于 identity,跳过
- coarse 暂不跑 fiber 约束 —— tendonmask 只是存着备用
- 4 块肌肉 < 4 tets / 14 块 < 12 tets:用户明确说"不用"优化
- 45 块薄片肌肉 tetgen 失败:接受当前 154/199 覆盖率

## 文件改动概要

- `src/muscle/constraints.py`: PINTOTARGET 类型 + create + 派发
- `src/muscle/muscle_warp.py`: solve_pintotarget_kernel + apply_pintotarget +
  _mark_pintotarget_stopped + update_attach_targets 处理 PINTOTARGET 刷新
- `src/muscle/muscle_common.py`: step() 在 repair 之后调 apply_pintotarget
- `src/muscle/coarse_solver.py`: 新增,~480 行
- `src/muscle/coupleOneWay/example_drive.py`: 检测 coarse_geo_path 切换 sim 类
- `tools/build_coarse_per_muscle.py`: pymeshlab+tetgen pipeline
- `tools/coarse_to_usd.py` / `tools/coarse_vs_fine_usd.py`: USD 可视化检查
- `data/muscle/config/bicep_drive_pintotarget.json`: 新
- `data/muscle/config/fullbody_drive.json`: 加回 muscleEndPin
- `data/muscle/config/fullbody_drive_gmg.json`: 新
- `data/Human/coarse_per_muscle.npz`: 生成产物
- `pyproject.toml`: `mesh-coarsen` extra

---

## GMG → realtimeMuscle 对齐(2026-05-10 后续)

跑通 fullbody GMG 后发现 mesh 仍爆炸乱飞。逐项对照 realtimeMuscle 的 substep,
发现 4 处偏差,全部修齐:

1. **Restrict 改 delta**: `coarse.pos += bary · (fine.pos − fine.pprev)`
   而不是 `coarse.pos = bary · fine.pos`(absolute)。delta 形式让 coarse 只
   响应 fine 的"运动量",不会在松弛阶段被 absolute 重采样误差累出 +1.97 vol_err。
2. **Coarse damping = 0**: realtimeMuscle 的 `project_constraints` 不读 pprev,
   速度阻尼只在 `semi_euler` 里做。原来 V2 coarse XPBD 还在按 0.1 dampingratio 跑。
3. **λ reset per substep**: realtimeMuscle 每个 substep 在 iter 循环前
   `reset_lagrangian(coarse.lagrangian)`。V2 漏了,导致 λ 在 200 步 × 10 substep
   = 2000 次累积,生成虚假力 → 爆炸。修法:`CoarseSolver.solve()` 开头
   `wp.launch(clear_cons_L_kernel, ...)`,跟 fine 端 `clear_cons_L_kernel` 同款。
4. **Pin 时序**: `apply_pintotarget` 原本在 substep **末尾**,意味着 `restrict()`
   读到的 fine.pos − fine.pprev 在 pin 处恒为 0,coarse 完全感知不到 pin 移动。
   修法:integrate 之后立刻调 `apply_pintotarget`,然后再 restrict — 让 pin
   teleport 进入 Δfine,coarse 才能跟着动。fine 求解之后再调一次保最终位置。
5. **Coarse stiffness 镜像 fine**: 删掉 `coarse_stiffness` 配置项,coarse 端的
   volume / tetarap stiffness 直接读 fine config 里同名约束。物理上对齐,
   配置上少一份分歧来源。
6. **Prolong 总覆盖**: `prolong_kernel` 删 `blend × interp + (1-blend) × fine`,
   改为直接 `fine.pos = bary · coarse.pos`(realtimeMuscle 默认行为)。

**结果**(fullbody bone_speed_scale=1.0,200 步,10 coarse iter):
- step 1 vol_err = −0.250(原 absolute restrict 时是 −0.488)
- step 100(acti=1.0 峰值)vol_err = +0.062(原是 −0.682,~10× 改善)
- step 200 vol_err = −0.336
- centroid 跟随骨头 (-0.001 → -0.200)

软体不再像 absolute restrict 时那样跟着骨头刚性同步,而是有适当的弹性滞后 ——
更接近真实肌肉。

---

## SVD-based ARAP(替换 polar form)

bicep 单肌肉 + native bone speed 测试时,V2 的 polar-form ARAP
`||F − R||_F` 在第一个 substep 就把 38/3938 个 tet 翻面,worst det(F) = −17。
跨 200 步累积到 195/3938 翻面、worst det(F) = −33.5(完全爆炸)。

数学上 `||F − R||_F = ||S − I||_F`(因为 F = R · V·diag(σ)·Vᵀ),polar 跟 SVD
等价。但**梯度路径**不同:V2 旧实现用 `H = Bᵀ · (F − R)`,polar 在 det F < 0
时给的 R 是 reflection(det R = −1),梯度方向反向 → 翻面后无法被推回。
realtimeMuscle 的 SVD 路径走 `dC/dF = Σ_k dC/dS_k · U[:,k] · V[:,k]ᵀ`,
配上 sign-fixed `ssvd_fn`(`det U`/`det V` 修正)始终把 F 推回 det > 0 一侧。

**实现**:`solve_tetarap_kernel` 完全替换为 SVD form,旧 polar-form
`tet_arap_update_xpbd_fn` 删除。

**结果**(bicep + 0.1× speed + 50 substep + per-substep bone interp):

| | 旧 polar | 新 SVD |
|---|---|---|
| step 1 末 | 4 翻面, −0.45 | **0** 翻面, det(F) ∈ [0.93, 1.09] |
| step 50 末 | 14 翻面, −3.0 | 1 翻面, −0.35 |
| step 100 末 | 17 翻面, −4.17 | 9 翻面, −2.28 |
| step 200 末 | 195 翻面, **−33.5** | 32 翻面, **−7.81** |

step 1-49 全程 0 翻面,worst-tet det(F) 增长接近线性而非指数发散。

---

## Pintotarget tet-coherent split

`tet 3637`(SVD ARAP 的惯犯)分析显示:它是顶部 3% 体积、宽高比 ~2 的 sliver
tet,**4 个顶点全过 muscleendmask 阈值 0.75**,但 mask 跨度 0.89-1.00。
其它惯犯(3644/45/1276/887/1588/3540)的 4 顶点 muscleendmask 都跨越阈值,
比如 tet 3644 的顶点是 (0.90, 0.72, 0.53, 0.91) —— 2 个超过 0.75 被硬 pin、
2 个不超 → **同一 tet 内同时存在硬 pin 和自由顶点**。骨头一动,pin 顶点瞬间
teleport,自由顶点滞后,tet 内部产生剧烈剪切 → 翻面。

**核心修复**:在 `create_pintotarget_constraints` 加 tet-coherent 后处理 ——
- 计算每个 tet 的 "pinned 顶点数"
- 仅当 4/4 都 pin 才认为这是 "interior tet",其顶点保持 PINTOTARGET 硬 pin
- 任何 1-3/4 的"边界 tet" 上的 pin 顶点 **改成完全自由**(默认),由 ARAP
  通过相邻 interior tet 拖动。可选 `boundary_attach_stiffness > 0` 退回
  ATTACH 软约束(legacy)
- 实测 bicep: 提督端 18 hard / 27 boundary→free,远端 21 hard / 13 free,
  共 39 个 vert 真硬 pin(原 79),137/3938 interior tet

用户洞察:扩张式贪婪("有任一 pin 顶点的 tet 全部加入 pin")**只是把边界外推
一圈**,新边界还会出现"半 pin tet"。所以根治办法不是扩张,而是**把硬 pin
区域收缩到完全 interior 的子集**,让边界 vert 通过 ARAP 弹性带动。

---

## Pin 时序:apply_pintotarget 移到 solve 前

原来 `MuscleSimBase.step()` 顺序是 `integrate → solve_constraints → ... →
apply_pintotarget(末尾)`。问题:每个 substep 内 ARAP 看到的 pin 还在上个
substep 的位置,等 solve 结束后 snap 才把 pin 推到当前 substep 目标 → ARAP
没有机会消化 snap 引入的剪切,只能在下个 substep 反应,然后下个 substep snap
再次推开。

修法:`apply_pintotarget` 在 integrate **之后**、solve **之前**调一次,
让 ARAP 在同一 substep 内"看到 pin 已经在新位置",把剪切立刻拉出来给相邻
elastic 顶点。末尾再调一次清掉求解器的微小飘移(因为 stopped/inv_mass=0
理论上飘移就是零,但显式覆写更稳)。

---

## Per-substep bone interpolation

debug 用增加的 `cfg.debug_per_substep` 启用每个 substep 的 mesh quality
检查后,发现 `update_attach_targets` 每整步只调一次,所以 50 个 substep 都
target 同一帧 bone pose —— 第一个 substep 上 pin teleport 全量发生,后面 49
个 substep 只是被动跟着。等于 substep 没起到细分骨头运动的作用。

修法:
- `BoneAnimation.sample_at_fractional_frame(f)`:lerp 两个相邻整数 frame
  的 4×4 关节矩阵(对小角度近似可接受)
- `MuscleSimBase.step(pre_substep_hook=...)`:整步开头不再调
  `update_attach_targets`,改由钩子在每个 substep 开头插值骨头位置
- `example_drive._make_substep_bone_hook`:为每步生成 hook,在 substep s/N 时
  把骨头插值到 `frame=(step-1+s/N)*dt*fps`
- 加 `cfg.bone_speed_scale`(默认 1.0):整体缩放骨头进度,方便测试慢动作
  极限情况(0.1× 把 worst-case shear 压低 ~90×)

---

## Mesh quality 调试钩子

`MuscleSimBase._debug_check_substep`:每个 substep 末尾调 `check_mesh_quality`
(已在 `src/muscle/couple/mesh_utils.py`),记录每 tet det(F) 范围、翻面计数、
worst tet id + det(F)。`MeshDistortionError` 被 try/except 包住不中断仿真,
方便完整跑完观察发散过程。`cfg.debug_per_substep: true` 开关。

`tools/inspect_tet.py`(临时,在 `/tmp` 下未提交)用于 dump 任意 tet 的
顶点 mask、与骨头距离、几何质量百分位,定位惯犯。

---

## 累计成果对比(bicep, 0.1× bone speed, 50 substeps)

| 配置 | step 1 末 | step 100 末 | step 200 末 | 修复幅度 |
|------|---|---|---|---|
| 旧 polar ARAP, 全 hard-pin, end-only snap | 4 翻面 / −0.45 | ~210 翻面 / ~−65 | **195 翻面 / −33.5** | baseline |
| SVD ARAP only | 0 / [0.93,1.09] | 9 / −2.28 | 32 / −7.81 | 6×/4× |
| **SVD + tet-coherent split + pin-before-solve** | **0** / [0.87,1.16] | ~13 / −2.5 | **14 / −4.85** | **14×/7×** |

centroid 也从 step 200 = −1.49(基线只走半路)进步到 −2.13(完整跟随骨头到位),
vol_err 全程 ±2%,step 200 是 +0.66%。

---

## 文件改动追加

- `src/muscle/muscle_warp.py`: 删除 polar `tet_arap_update_xpbd_fn`,
  `solve_tetarap_kernel` 完全替换为 SVD form;`solve_tetarap_v2_kernel`
  作为中间过渡版本已合并
- `src/muscle/coarse_solver.py`: delta restrict、`clear_cons_L_kernel` 调用、
  prolong 总覆盖、coarse stiffness 镜像 fine、`coarse_only` debug flag、
  fine-mass / fine-tendonmask gather
- `src/muscle/constraints.py`: `create_pintotarget_constraints` 增加
  tet-coherent split + boundary→free / ATTACH 选项
- `src/muscle/muscle_common.py`: `step(pre_substep_hook=...)` 接受 substep
  hook;`apply_pintotarget` 在 solve 前后各调一次;`_debug_check_substep`
  per-substep mesh quality 日志
- `src/muscle/coupleOneWay/bone_anim.py`: `sample_at_fractional_frame`
- `src/muscle/coupleOneWay/example_drive.py`: `_make_substep_bone_hook`,
  step_cnt 提前到 sim.step() 之前赋值
- `data/muscle/config/bicep_drive_pintotarget.json`: `debug_per_substep`,
  `bone_speed_scale`,`num_substeps=50`
- `data/muscle/config/fullbody_drive_gmg.json`: 删 `coarse_stiffness` /
  `coarse_dampingratio`(转为镜像 fine);加 `coarse_only` flag
