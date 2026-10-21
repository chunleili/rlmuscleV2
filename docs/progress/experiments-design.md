# 实验设计 —— RLMuscle (TVCG 投稿)

围绕两个核心创新点设计实验。Innovation 1(GMG 稳定性)思路明确,Innovation 2(Hill Injection 的生理学合法性)是核心难点。

---

## Innovation 1 — GMG 让 XPBD 模拟更稳定

> 不要只展示 "baseline 爆炸 vs 我们不爆炸"。TVCG 审稿人期待多维度的 quantitative 证据。

### E1.1 稳定性相图(Phase Diagram, **headline figure**)
- 扫描参数空间 $(\Delta t,\; k_{\text{vol}})$,每组运行 30 秒固定姿态 + 重力的静止测试
- 分类为 **稳定 / 振荡 / 爆炸** 三档
- 给 single-level XPBD 与 GMG 各画一张相图
- 视觉效果:"GMG 的稳定区域大数倍"
- 这是 multigrid 类论文的标配

### E1.2 h-independent convergence(**multigrid 的金字招牌,必须有**)
- 扫描网格分辨率 4 级(例 ~5k / 30k / 100k / 468k 四面体)
- 测达到固定残差所需的迭代次数
- 预期:Single-level 迭代数 $\propto N$,GMG 接近常数
- 这条直接区分 "GMG vs. 普通预条件",**没有这条会被审稿人专门挑**

### E1.3 Volume drift 时间曲线
- 固定姿态 + 重力,绘制 $|V/V_0 - 1|$ 随时间(30s+)
- Baseline 单调漂移;GMG 应当有界并收敛到机器精度附近

### E1.4 Activation step response
- $a: 0 \to 1$ 阶跃,测体积/形状 overshoot 与稳定时间
- 展示 "controllability"

### E1.5 Ablation
- Jacobi-only / Gauss–Seidel-only / GS+Jacobi 两阶段 / + GMG
- 撑住 §3.3.2 的 "two-pass scheme" claim

---

## Innovation 2 — Hill-Injection Fiber Constraint(核心)

### 问题诊断与论证策略

TVCG 不是 *Journal of Biomechanics*,审稿人不会要求真人数据。生理学的辩护策略**不应该**是 "我们贴近真人",而应该是:

> **"1D Hill 是生物力学公认的金标准。我们是第一个把 1D Hill 的能量律精确无近似地抬升到 3D XPBD 体积域。因此在退化(1D-like)配置下,我们必须能复现 1D Hill;在 3D 配置下,直接继承 Hill 的生理学合法性。"**

把 "生理学正确" 转化为 **可量化的数值复现问题** —— 图形学审稿人能接受、能验证、能打分。

### 必备 baselines(决定说服力)

| Baseline | 含义 | 我们的优势 |
|---|---|---|
| **Analytical Millard 2012** | Hill 解析曲线 | gold truth |
| **OpenSim 1D** | 生物力学标准实现 | 跨实现交叉验证 |
| **Lee et al. 2018 (PD, rest-length modulation)** | graphics SOTA 的 Hill 近似 | "无近似" 优势 |
| **Linear active-stress** $\mathbf{P}=a\sigma_0\,\mathbf{F}(\mathbf{d}\otimes\mathbf{d})$ | graphics 常用简化 | 揭露 "丢失钟形 F-L 曲线" 的代价 |
| (可选)**数值积分版 fiber constraint** | Gauss quadrature 替代 closed-form | 凸显 closed-form 又准又快 |

### E2.1 单肌等长测试 —— Force–Length–Activation 曲面(**主图**)

**装置**: 取单根肌肉(直接用 biceps),固定 origin 与 insertion 在两个虚拟刚体上,只此一肌,无重力。

**协议**:
- 扫描 $\tilde\ell^M \in [0.4, 1.6]$(覆盖 ascending / plateau / descending 全段)
- 扫描 $a \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$
- 测稳态下 attachment 反作用力 $\|\sum\lambda_{\text{attach}}\nabla C\|$(= tendon force)

**输出**:
1. 平面图:$a=0$ 的 passive F–L、$a=1$ 的 total F–L —— 叠加 analytical / OpenSim / Lee2018 / Linear
2. 三维曲面:$F(a,\tilde\ell^M)$,与 analytical 的差分 heatmap
3. **RMSE 表格**(逐 baseline)。Ours 应 $<1\%$;Lee2018 在 ascending limb 系统偏差;Linear 完全丢钟形

> 一图一表足以证明 **"Hill injection 是 exact 的,prior art 不是"**

### E2.2 闭式积分精度验证
- closed-form degree-10 polynomial vs. Gauss quadrature(4-pt / 8-pt)
- 精度:相对 64-pt 高精度参考的差值
- 速度:per-tet 评估纳秒数
- 撑住 paper 里 "no numerical quadrature" 的 claim

### E2.3 EMG-like 验证(**审稿人最爱的实物对照**)

对标准动作(bicep curl / squat / walking gait)运行 RL,导出每根肌肉 $a(t)$,**与已发表 EMG 文献比较**激活时序:

- **Bicep curl**: biceps 屈肘相激活, triceps 伸肘相激活, 强度随负荷大致线性
- **Squat**: quadriceps + gluteus 在下蹲底部峰值, calf 维持姿态
- **Walking gait**: tibialis anterior 在 swing 相, gastrocnemius 在 push-off 相

**输出**: 把 $a(t)$ 曲线 overlay 在 EMG 文献模板上。**不需要数值完美匹配**,时序+峰值位置对即可。Lee 2019 (MASS) 就是用这种 qualitative EMG comparison。

### E2.4 Bulging 几何验证

- 测肌腹中央横截面积 $A(a)/A(0)$ 在 $a\in[0,1]$ 上的增益
- 对比 in-vivo MRI / ultrasound 文献(biceps bulge ~10-20% 截面积增益)
- 与 Lee 2018 / Linear baseline 比较 bulge 形状
- **关键**: 把 "生理学" 从 1D 推到 3D —— 1D Hill 没有 bulging 概念,3D injection + volume constraint 一起才有这个 emergent behavior

### E2.5(可选)双关节多肌协调
- biceps + brachialis + triceps 共同驱动肘关节屈伸
- antagonist co-activation 在 $\mu_\theta$ 下自然涌现
- 协同肌力分配与文献一致

---

## 推荐 Results 章节叙事顺序

每一小节对应一张主图:

1. **§4.1 1D Hill Fidelity** (E2.1, E2.2) — 数字证明 injection 精确
2. **§4.2 Volumetric Behavior** (E2.4 + bicep-3 那张图) — 视觉 + 几何
3. **§4.3 Solver Stability** (E1.1–E1.4) — 数值方法稳定性论证
4. **§4.4 Motion Tracking + EMG Validation** (E2.3 + RL 整体结果) — 系统级 closing argument
5. **§4.5 Ablations** (GS-only / Jacobi-only / no-MG / Linear-stress) — 各组件贡献量化

---

## 关键落地建议

- **E2.1 的 figure = 整篇论文主图**,值得占 full-column。两条 F-L + 一张 RMSE 表把所有 baseline 钉死
- **E1.2 的 h-independent convergence 图必备**
- EMG 模板从 Besomi 2019 或 Winter *Biomechanics and Motor Control* 取标准曲线,引用要正规
- **Linear active-stress baseline 强烈建议加上** —— 实现成本极低($f_{\text{hill}}\!\to\!a\sigma_0$),性价比最高,让 "我们用 Hill" 的价值 self-evident
- bicep-3.png 已经是 E2.4 的雏形,后续可扩展成多肌 / 多姿态版本

---

## 待办

- [ ] E2.1 实现单肌等长 rig 与 baseline 扫描脚本
- [ ] E1.1 实现稳定性相图扫描自动化
- [ ] E1.2 多分辨率网格生成(5k / 30k / 100k / 468k)
- [ ] EMG 文献调研:Besomi 2019, Winter 等正规出处
- [ ] Lee 2018 baseline 在我们的 XPBD 框架内的实现
- [ ] Linear active-stress baseline(成本极低,先做)
