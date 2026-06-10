# Muscle Force → Joint Torque

Each muscle is a polyline of anchors $\{\mathbf{p}_i\}$ skinned to rigid bodies, so anchor positions follow body kinematics.

**Length.** Normalized muscle–tendon length and (rigid-tendon) fiber length:

$$
\ell_{mt}=\frac{1}{\ell_{mt0}}\sum_i\lVert\mathbf{p}_i-\mathbf{p}_{i-1}\rVert,\qquad
\bar{\ell}_m=\frac{\ell_{mt}-\ell_t^0}{\ell_m^0}.
$$

**Force.** Hill-type, $a\in[0,1]$ the policy activation:

$$
F(a)=f_0\big(a\,g_{\!A}(\bar{\ell}_m)+g_{\!P}(\bar{\ell}_m)\big),\qquad
g_{\!A}(\bar\ell)=e^{-(\bar\ell-1)^2/\gamma}.
$$

It acts as tension along the line: anchor $i$ gets $\mathbf{f}_i=F\,\mathbf{d}_i$, $\mathbf{d}_i$ the sum of unit directions to its neighbors.

**Torque.** Project anchor forces through the transpose point Jacobian $\mathbf{J}_i=\partial\mathbf{p}_i/\partial\mathbf{q}$. Active force is linear in $a$, so total muscle torque splits:

$$
\boldsymbol{\tau}_{\text{musc}}(\mathbf{a})=\mathbf{L}\,\mathbf{a}+\mathbf{b},\qquad
\mathbf{L}_{:,m}=\sum_i\mathbf{J}_i^{\top}f_0\,g_{\!A}\,\mathbf{d}_i,\quad
\mathbf{b}=\sum_{m,i}\mathbf{J}_i^{\top}f_0\,g_{\!P}\,\mathbf{d}_i.
$$

$\mathbf{L}$ = per-unit-activation moment arms, $\mathbf{b}$ = passive torque. Root DOFs excluded.
