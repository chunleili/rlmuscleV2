● 正规方程(normal equations)= 最小二乘问题取极小时的一阶条件。推导:

  目标: 最小化残差平方
  $$f(\mathbf{x}^c) = \lVert \mathbf{P}\mathbf{x}^c - \tilde{\mathbf{x}}^f \rVert_2^2 =
  (\mathbf{P}\mathbf{x}^c - \tilde{\mathbf{x}}^f)^\top(\mathbf{P}\mathbf{x}^c - \tilde{\mathbf{x}}^f)$$

  展开:
  $$f = \mathbf{x}^{c\top}\mathbf{P}^\top\mathbf{P}\mathbf{x}^c -
  2,\mathbf{x}^{c\top}\mathbf{P}^\top\tilde{\mathbf{x}}^f + \tilde{\mathbf{x}}^{f\top}\tilde{\mathbf{x}}^f$$

  求梯度,令 = 0(极小点处导数为零):
  $$\nabla f = 2\mathbf{P}^\top\mathbf{P}\mathbf{x}^c - 2\mathbf{P}^\top\tilde{\mathbf{x}}^f = 0$$

  这就是正规方程:
  $$\mathbf{P}^\top\mathbf{P},\mathbf{x}^c = \mathbf{P}^\top\tilde{\mathbf{x}}^f$$

  解出(PᵀP 可逆,块对角且每块满秩):左乘 (PᵀP)⁻¹:
  $$\mathbf{x}^c = (\mathbf{P}^\top\mathbf{P})^{-1}\mathbf{P}^\top,\tilde{\mathbf{x}}^f$$

  即 eq:restriction。

  几何直觉: P x^c 张成 cage 子空间。最优解 = 把 x̃^f 正交投影到该子空间。残差 P x^c − x̃^f 须垂直于子空间 ⇒
  Pᵀ(P x^c − x̃^f) = 0 ⇒ 同正规方程。P(PᵀP)⁻¹Pᵀ 就是投影矩阵。

  要不要把这步简短写进 paper(一句"normal equations"加注)?还是当前够了?