> **⚠️ Historical progress log.** The `MultigridMuscleSim` / `coarse_solver.py`
> GMG references below are from the old `coupleOneWay` codebase (since replaced by
> `src/muscle/mg/mg.py`). Current multigrid: [`../humanV2_cage_multigrid.md`](../humanV2_cage_multigrid.md).

# LBS-as-predictor (bone-driven muscles without collision)

## 背景

`fullbody_drive` pre-roll 阶段已经稳定,但骨头开始走路后**肌肉滞后非常严重** ——
muscle body 跟不上骨头,远端 attach 顶点被拉成软糖。根因:没有 collision,
constraint-only solve 在快速运动下只能靠 vol/arap 把肌肉慢慢"拽"过去,而
attach 软约束又被 vol/arap 反向拽住,形成滞后回路。

用户思路:**先做 LBS 把肌肉"刚性"地搬到骨头当前姿态,再让 XPBD constraints
做弹性修正**。LBS 提供强 kinematic prior,constraints 只负责高频修正而不是
唯一的运动来源。

## 整体管线

每个 substep 内:

```
integrate(fine)                 # XPBD 半隐式预测(惯性 + dt·v)
post_integrate_hook(sub_i):     # LBS 预测覆盖
    joint_world = anim.joint_world_at_fractional_frame(f)
    T_rel[b] = inv(M_rest_bone[b]) @ (offset_bone[b] @ joint_world[driver[b]])
    pos[i] = (1−blend) · pos[i] + blend · pos0[i] @ T_rel[vertex_bone[i]]
pintotarget snap
solve_constraints (vol + arap + fiber + attach)
apply_dP / repair / update_velocities
```

`blend=1.0` 完全用 LBS 替代惯性预测;`<1.0` 保留一部分惯性。
默认 `cfg.lbs_blend = 1.0`(实测 fullbody walk 上没有副作用,vol_err 全程
≤5%)。

## Vertex → bone assignment

`src/muscle/coupleOneWay/lbs_skin.py: LBSDriver`

1-hot 最近骨骼:对每个 muscle vert 在 rest-pose 世界空间上跑 cKDTree,找到最近的
**bone vert**,再查它属于哪根 bone slice。这样不需要骨骼骨架结构,纯几何就能搞定。

为什么 1-hot 够用?
- ARAP 是 SVD-based,自带跨 bone 边界的几何平滑;blending weight 反而会引入
  额外的"软糖"层
- attach 约束在肌腱端独立工作,不依赖 LBS 权重
- 实测 walking + pre-roll 两段都没有看到 LBS 边界的可见 artifact

缓存:`data/Human/lbs_weights_<cfg.name>.json`,只存 `vertex_bone[n_verts]`
(int 数组)。KDTree build 在 157k vert × 91 bone 量级要 ~1.5 s,缓存后毫秒级。

## Per-bone rigid transform

行向量约定(全管线统一):

```
M_t_bone   = offset_bone @ joint_world[driver_idx]    # 当前骨头世界变换
T_rel      = inv(M_rest_bone) @ M_t_bone              # rest→current 的相对刚体变换
pos_world  = pos_world_rest @ T_rel                   # 应用到 muscle vert
```

`offset_bone = bone_world_rest @ inv(joint_rest)` 已经在 `BoneAnimation`
构造时算好。

上传 GPU 时打平成 12 float/bone(`r00 r01 r02  r10 r11 r12  r20 r21 r22  t0 t1 t2`),
kernel 用 `T_rel_flat[b*12 + k]` 索引,避免 mat44 在 Warp 上的对齐开销。

`anim.joint_world_at_fractional_frame(f)` 复用已有的 lerp 实现
(`sample_at_fractional_frame` 改为先调它再 apply,确保 fractional bone 和
fractional LBS 看同一个 joint_world)。

## Warp kernel

`muscle_warp.py: lbs_apply_kernel`:

```python
@wp.kernel
def lbs_apply_kernel(pos, pos0, vertex_bone, T_rel_flat, blend):
    i = wp.tid()
    b = vertex_bone[i]
    if b < 0: return                # 未匹配的 vert 保持原值
    p = pos0[i]
    base = b * 12
    qx = p[0]*T[base+0] + p[1]*T[base+3] + p[2]*T[base+6] + T[base+9]
    qy = p[0]*T[base+1] + p[1]*T[base+4] + p[2]*T[base+7] + T[base+10]
    qz = p[0]*T[base+2] + p[1]*T[base+5] + p[2]*T[base+8] + T[base+11]
    pos[i] = (1.0 - blend) * pos[i] + blend * wp.vec3(qx, qy, qz)
```

注意 `qx = p · row0`(行向量;p 是 `pos0[i]`,即 rest 世界位置)。
之前一版写成列向量乘法导致整个 mesh 镜像翻转 —— 行/列约定要一以贯之。

## Hook 接入

`MuscleSimBase.step(pre_substep_hook=..., post_integrate_hook=...)`:
- `pre_substep_hook(sub_i, n_sub)` 在 integrate **之前**调(更新 attach target +
  per-substep bone pose,旧有逻辑)
- `post_integrate_hook(sub_i, n_sub)` 在 integrate **之后**、solve_constraints
  **之前**调(LBS 覆盖预测位置)

`MultigridMuscleSim.step` 镜像同样两参数,用于将来 GMG + LBS 组合(目前 GMG
分支没新接入 LBS,占位)。

`example_drive._make_substep_hooks(sim, anim, cfg, step_index, lbs, lbs_blend)`
返回 `(pre, post)` 二元组:
- `pre` 计算当前 fractional frame,更新 bone_pos_field + attach target
- `post` 用同一 fractional frame 取 joint_world,算 T_rel,upload,
  调 `sim.apply_lbs(vertex_bone_field, blend=lbs_blend)`

`lbs_blend ≤ 0` 时直接返回 `(pre, None)`,LBS 完全关闭(用于和 baseline 对比)。

## LBS-only baseline

为了直观看 LBS 单独工作的样子(完全跳过 XPBD),加了 `--lbs-only` CLI flag:

```python
def _run_lbs_only(sim, anim, cfg, args, frame_at, render_stride, lbs):
    speed = float(getattr(cfg, "bone_speed_scale", 1.0))
    motion_per_step = speed * cfg.dt * anim.fps
    vertex_bone_field = lbs.vertex_bone_field(sim)
    for step in range(1, cfg.nsteps + 1):
        f = step * motion_per_step
        joint_world = anim.joint_world_at_fractional_frame(f)
        sim.set_lbs_T_rel(lbs.compute_T_rel_flat(joint_world))
        sim.apply_lbs(vertex_bone_field, blend=1.0)
        sim.bone_pos_field.assign(anim.sample_at_fractional_frame(f))  # 渲染骨头用
        if step % render_stride == 0:
            sim.render()
```

⚠️ **bug fix**: 第一版漏了 `bone_pos_field.assign(...)`,导致 USD 里骨头
mesh 一直停在 rest pose("lbsonly那个骨骼没动啊")。原因:常规管线在
`pre_substep_hook` 里给 `bone_pos_field` 灌当前帧位置,而 `_run_lbs_only`
完全绕过了 substep 循环,得手动补一次。

LBS-only 跑 73 帧整段只要 ~1 秒(对比完整管线 ~45 秒),非常适合快速预览
walking motion 的骨头驱动效果。

## 结果

`output/fullbody_drive.usd`(constraint-only 旧 baseline):
- pre-roll 末尾肌肉略软,但骨头开始走后**整体明显滞后** ——
  muscle body 像在水里飘,远端 attach 被拉成软糖

`output/fullbody_drive_lbs.usd`(LBS+constraints,`lbs_blend=1.0`):
- pre-roll 和 walking 全程跟住骨头
- vol_err 200 步内全程 ∈ [−5%, 0%]
- 没有可见的 LBS 边界 artifact(ARAP 自然平滑)
- 1-hot weighting 足够,不需要 weight blending

`output/fullbody_drive_lbsonly.usd`(纯 LBS,对照):
- 完全刚性绑定,bone 怎么动 muscle 怎么动,完全没有弹性 —— 是 LBS 上界

## 配置

`data/muscle/config/fullbody_drive.json` 新增:
```json
"lbs_blend": 1.0,
```

CLI:
- `--lbs-blend 0.5`:override 配置里的 blend(0 完全关闭,1 完全覆盖)
- `--lbs-only`:跳过 XPBD,只跑 LBS(快速预览)

## 文件改动

- `src/muscle/coupleOneWay/lbs_skin.py`:新,LBSDriver(KDTree assignment +
  T_rel 计算 + cache I/O + Warp field 绑定)
- `src/muscle/muscle_warp.py`:`lbs_apply_kernel`,`MuscleSim.set_lbs_T_rel()`,
  `MuscleSim.apply_lbs()`,`_lbs_T_rel_field` 字段
- `src/muscle/muscle_common.py`:`step(post_integrate_hook=None)` 参数
- `src/muscle/coarse_solver.py`:`MultigridMuscleSim.step` 镜像参数
- `src/muscle/coupleOneWay/bone_anim.py`:抽出 `joint_world_at_fractional_frame`
  (`sample_at_fractional_frame` 改为先调它再 apply)
- `src/muscle/coupleOneWay/example_drive.py`:`_make_substep_hooks` 返回
  `(pre, post)`,加 `--lbs-blend` / `--lbs-only`,`_run_lbs_only` baseline
- `data/muscle/config/fullbody_drive.json`:`lbs_blend: 1.0`
- `data/Human/lbs_weights_fullbody_drive.json`:vertex→bone 缓存
  (157k vert × 91 bone,运行时若不存在会自动生成)

## 未做 / 后续

- **GMG + LBS 组合**:`MultigridMuscleSim` 已经接好 `post_integrate_hook` 参数
  但 hook 内部没有按 coarse 流程做 prolong/restrict 协同。等 collision 进来
  之后再决定要不要给 coarse 端也加 LBS prior
- **Two-way coupling**:目前是 muscle 完全跟随 bone(`stopped` 的 attach 端是
  硬 pin),没有 muscle 反作用力推 bone。等加 collision 之后会比较自然地引出
  这条路径
- **Blend < 1.0 的工程意义**:目前 `lbs_blend=1.0` 已经够稳,没看到必要保留
  惯性预测。如果未来加 collision 后惯性更重要,可以降到 0.5~0.8 让 XPBD
  保留更多 momentum
