---
title: "从 Adam 到在线矩阵 Whitening"
description: "从 Adam 的对角二阶矩出发，理解在线矩阵 Whitening、KL-Root-Kron 及其 Kronecker 实现。"
date: "2026-09-01"
lang: "zh-CN"
author: "Haoyu Tang"
bibliography: "ref.bib"
---

# 从 Adam 到在线矩阵 Whitening

Adam 的核心操作之一，是利用梯度的二阶矩对更新做归一化。对第 $i$ 个参数坐标，Adam 维护 $\bm v_{t,i}\approx \mathbb E[\bm g_i^2]$，计算参数更新 $\Delta\bm \theta_i\propto-\frac{\bm m_i}{\sqrt{\bm v_i}}$.

如果把梯度二阶矩写成 $\bm H=\mathbb E[\bm g\bm g^\top]$，那么 Adam 实际上只保留了 $\bm H$ 的对角部分
$$
\bm H_{\mathrm{diag}}=\operatorname{diag}\bigl(\mathbb E[\bm g_1^2],\dots,\mathbb E[\bm g_d^2]\bigr),
$$
因此 Adam 的预条件器可以写成
$$
\bm P_{\mathrm{Adam}}=\bm H_{\mathrm{diag}}^{-1/2}.
$$

从这个视角出发，不免得到一个很自然的问题：

> 如果不局限于二阶矩对角元，而是保留梯度不同方向之间的相关性，会得到什么样的优化器？

不妨直接推广，
$$
\bm P=\bm H^{-1/2},\qquad \bm H=\mathbb E[\bm g\bm g^\top],
$$
由于 $\bm H$ 是梯度 covariance，它天然是对称半正定矩阵。为了让通常意义下的 inverse square root 有定义，下面先假设 $\bm H\succ0$；如果 $\bm H$ 存在退化方向，实际实现中通常对它加入一个很小的 damping，使用 $(\bm H+\varepsilon I)^{-1/2}$。在这个假设下，$\bm P$ 也是对称正定矩阵，即 $\bm P\succ0$.

此时参数更新变成
$$
\Delta\bm \theta=-\eta \bm P\bm m=-\eta \bm H^{-1/2}\bm m.
$$
注意这里的 $-1/2$ 次方和 Newton 法里的 $\bm H^{-1}$ 是两件不同的事情。Newton 的 $\bm H$ 是 Hessian，并通过 $-\bm H^{-1}\bm g$ 求解局部二次模型；这里的 $\bm H$ 是梯度二阶矩，目标更接近 whitening / RMS normalization。

## 1. 与其直接计算 $\bm H^{-1/2}$，不如求解 whitening 方程

直接维护 $\bm P=\bm H^{-1/2}$ 的问题很明显：矩阵 inverse square root 很贵。但 $\bm P=\bm H^{-1/2}$ 有一个等价条件 $\bm P\bm H\bm P=\bm I$. 通过这个转化
$$
\bm P\bm H\bm P=\bm P\,\mathbb E[\bm g\bm g^\top]\bm P=\mathbb E[(\bm P\bm g)(\bm P\bm g)^\top],
$$
可以定义预条件后的梯度 $\bm y=\bm P\bm g$，那么理想状态就是
$$
\mathbb E[\bm y\bm y^\top]=\bm I.
$$
也就是说
$$
\boxed{
\bm P=\bm H^{-1/2}
\quad\Longleftrightarrow\quad
\text{对预条件化的梯度做白化.}
}
$$


## 2. 在线训练中，可以直接估计 $\bm P\bm H\bm P$

实际训练过程中通常没有精确的 $\bm H=\mathbb E[\bm g\bm g^\top]$，但每一步都有当前梯度 $\bm g_t$. 于是可以构造 $\bm C_t=(\bm P\bm g_t)(\bm P\bm g_t)^\top$，它满足
$$
\mathbb E[\bm C_t]=\bm P\mathbb E[\bm g_t\bm g_t^\top]\bm P=\bm P\bm H\bm P ,
$$
所以 $\bm C_t$ 是 $\bm P\bm H\bm P$ 的一个估计。因而 $\bm P\bm H\bm P=\bm I$ 就可以在线写成 $\bm C_t-\bm I\to 0$. 

需要注意，真正要求的是 $\mathbb E[\bm C_t]=\bm I$，而 $\bm C_t-\bm I$ 只是一个 noisy stochastic residual。

到这里，原问题已经可以简化为 
$$
\boxed{
\bm C_t=(\bm P_t\bm g_t)(\bm P_t\bm g_t)^\top,
\qquad
\bm C_t-\bm I\to 0.
}
$$

## 3. $\bm C_t-\bm I$ 包含了什么信息？

这一节，我们先来看看这个 $\bm C_t-\bm I$ 可以如何理解。由于 $\bm C_t$ 是一个对称阵，不妨对其做特征分解，
$$
\bm C_t=\bm U\bm \Lambda \bm U^\top,\quad \bm \Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_d),
$$
这里 $\bm U$ 的列向量给出了当前 $\bm C_t$ 的特征向量，而 $\lambda_i$ 则是对应的特征值。如果 $\lambda_i>1$，说明这个方向被放得太大，应当压缩；如果 $\lambda_i<1$，说明这个方向仍然太小，应当放大。因此 $\bm C_t-\bm I$ 实际上就是一个各向异性的白化误差。

相比之下，Adam 只能看到 $\operatorname{diag}(\bm H)$，因此只能逐元素调整；而此处 $\bm C_t$ 中的 off-diagonal 信息则允许 optimizer 识别旋转后的高方差方向和低方差方向。

## 4. 如何实际迭代更新 $\bm P$？

现在真正的问题只剩下 $\bm C_t\to \bm I$ 应该对应什么样的 $\bm P$ 更新？

最直接当然是定义一个平方损失函数
$$
L(\bm P)=\frac12\|\bm P\bm H\bm P-\bm I\|_F^2.
$$
但直接对上式求梯度需要计算 $\bm H$，复杂度很高，而且更新后也未必始终保持 $\bm P\succ0$ 这样一个约束。

考虑到 GPU 做乘法运算比较高效，一个直接想法是乘法迭代，
$$
\bm P^+=\bm A^\top\bm P\bm A,
$$
那么只要 $\bm P\succ0$ 并且 $\bm A$ 可逆，就有 $\bm P^+\succ0$. 于是问题进一步变成应该选择什么 matrix function $f$ 来定义 $\bm A=f(\bm C_t)$，使得 $\bm C_t\to \bm I$ 对应 $\bm P\to \bm H^{-1/2}$。

## 5. 从 matrix function 看理想 correction

暂时先考虑一个最简单的 whitening 问题。假设随机向量 $\bm y$ 的 covariance 是 $\bm C_t=\mathbb E[\bm y\bm y^\top]$，如果直接做 $\bm y^+=\bm A \bm y$，那么 $\bm C_t^+=\bm A \bm C_t \bm A^\top$. 若取对称矩阵 $\bm A=\bm C_t^{-1/2}$，自然直接就有 $\bm C_t^+=\bm I$. 所以 inverse square root 本身就是最自然的一种 whitening correction.

这提示我们可以把 preconditioner update 写成某种
$$
\bm A=\bm C^{-\alpha},
$$
不过，将这一操作放进在线优化器后，情况会有所不同。训练过程中观察到的 $\bm C$ 往往不是精确的全局协方差，而只是由当前 mini-batch 得到的随机估计，因此直接做 $\bm A=\bm C^{-1/2}$ 过于激进。因此，更自然的做法不是每一步都“完全 whiten”，而是只向 whitening fixed point 移动一部分。

一种简洁的构造是 
$$
\bm A=\bm C^{-\eta/2},\quad 0<\eta\le 1,
$$
当然，真正计算 $\bm C^{-\eta/2}$ 本身仍然需要 matrix function，工程上不划算。这时候就可以来引入近似。

## 6. 对 matrix function 做近似

如果当前 $\bm C$ 已经不离 $\bm I$ 太远，令 $\bm C=\bm I+\bm E$，则 $\bm C^{-\eta/2}=(\bm I+\bm E)^{-\eta/2}$. 做一阶 Taylor 展开有
$$
\bm C^{-\eta/2}=\bm I-\frac{\eta}{2}\bm E+O(\bm E^2),
$$
于是自然得到
$$
\boxed{\bm A=\bm I-\frac{\eta}{2}(\bm C-\bm I)}
$$
作为最低成本的一阶 matrix-function approximation. 

这个形式特别适合 optimizer，其中的矩阵 $\bm C$ 来自一个 Gram matrix，运算均为正常的加减法，没有特征分解、求逆、矩阵对数这些相对昂贵的操作。

当然如果愿意多付出一些计算开销，可以继续保留二阶项：
$$
\bm C^{-\eta/2}\approx \bm I-\frac{\eta}{2}(\bm C-\bm I)+\frac{\eta(\eta+2)}{8}(\bm C-\bm I)^2,
$$
特别地，取 $\eta=1$，
$$
\bm C^{-1/2}\approx \bm I-\frac12(\bm C-\bm I)+\frac38(\bm C-\bm I)^2.
$$

可以把以上方法理解成
$$
\boxed{
\text{用低阶 polynomial 近似 inverse-root matrix function}
}
$$
而这和 Muon / Newton-Schulz 一类方法的工程哲学其实很接近：避免 eig/SVD，把 matrix function 改写成少量 GEMM。

## 7. 如何在实现中维护 preconditioner 更新？

上面的这种乘法迭代可以直接作用在 $\bm P$ 上，但在实际实现中，我们还希望每一步都显式地保持 $\bm P$ 的对称正定结构。一个自然的参数化方式是引入其平方根因子 $\bm R$，写成 $\bm P=\bm R^\top \bm R$。这样，更新不再直接修改 $\bm P$，而只需要做更新 $\bm R^+=\bm R\bm A$.

由于 $(\bm R\bm A)^\top(\bm R\bm A)=\bm A^\top \bm R^\top \bm R\bm A$，而这里的 $\bm A=f(\bm C)$ 是 $\bm C$ 的对称 matrix function，满足 $\bm A^\top=\bm A$，所以新的因子确实对应于 $\bm P^+=\bm A\bm P\bm A$.

于是，一个极简版的在线算法可以写成

$$
\begin{aligned}
\bm y_t&=\bm R_t^\top \bm R_t\bm g_t,\\
\bm C_t&=\bm y_t\bm y_t^\top,\\
\bm A_t&=\bm I-\frac{\eta}{2}(\bm C_t-\bm I),\\
\bm R_{t+1}&\leftarrow \bm R_t\bm A_t.
\end{aligned}
$$

然后用 $\bm P_t=\bm R_{t+1}^\top \bm R_{t+1}$ 预条件来做真正的参数更新。值得注意的是，这里第一个 $\bm y_t$ 是用当前的 raw gradient $\bm g_t$ 来做计算，而真正的更新 $\bm u_t$ 则可以用新的预条件 $\bm P_t$ 来处理梯度的动量 EMA 来算。这是因为动量中包含参数跨时间的 correlation，因此更新 $\bm P_t$ 用 raw gradient 更能反映当前的 whitening 状态。

## 8. 从向量预条件器到矩阵参数

到目前为止，我们讨论的都是向量情形：
$$
\bm g\in\mathbb R^d,\qquad \bm P\in\mathbb R^{d\times d},
$$
然而，在真实神经网络中，参数通常是形如 $\bm W\in\mathbb R^{m\times n}$ 的矩阵，将其展平维度会变成 $d=mn$，对应的 full preconditioner 需要具有 $\bm P\in\mathbb R^{mn\times mn}$ 这样的 shape，显然计算开销过大，无法直接实现。因此，需要引入结构化 approximation。一种自然的选择是 Kronecker factorization
$$
\bm P_{\mathrm{full}}=\bm P_b\otimes \bm P_a,
$$
其中 $\bm P_a\in\mathbb R^{m\times m},\bm P_b\in\mathbb R^{n\times n}$. 对于矩阵梯度 $\bm G\in\mathbb R^{m\times n}$，利用 Kronecker 与向量化的恒等式，有
$$
\bm P_{\mathrm{full}}\operatorname{vec}(\bm G)=\operatorname{vec}(\bm P_a\bm G\bm P_b).
$$
于是可以直接定义预条件后的矩阵梯度 $\bm Y=\bm P_a\bm G\bm P_b$，这样就不必构造巨大的 $mn\times mn$ covariance，而只需分别考察两个 mode 上的 covariance：
$$
\bm C_a=\frac1n\bm Y\bm Y^\top,\quad \bm C_b=\frac1m\bm Y^\top \bm Y.
$$
理想的 fixed point 对应于
$$
\mathbb E[\bm C_a]=\bm I_m,
\qquad
\mathbb E[\bm C_b]=\bm I_n.
$$
可以证明，这两方程与 idealized KL-Shampoo 的 Kronecker covariance fixed point 等价。于是，两边都可以进行与前面相同的一阶更新
$$
\bm A_a=\bm I-\frac{\eta_a}{2}(\bm C_a-\bm I),\quad \bm A_b=\bm I-\frac{\eta_b}{2}(\bm C_b-\bm I),
$$
同时维护
$$
\bm P_a=\bm R_a^\top \bm R_a,\qquad \bm P_b=\bm R_b^\top \bm R_b,
$$
并做更新
$$
\bm R_a\leftarrow \bm R_a\bm A_a,
\qquad
\bm R_b\leftarrow \bm R_b\bm A_b,
$$
就得到一个实际可实现的 Kronecker 版本。只需存储两个较小的矩阵，而不需要面对 $mn\times mn$ 的完整计算量。后者空间上需要 $O(m^2n^2)$，而 Kronecker 版本只需 $O(m^2+n^2)$，计算上也从 $O(m^2n^2)$ 降到 $O(m^2n+mn^2+m^3+n^3)$，显著降低了开销。

## 9. KL-Root-Kron 的正式推导

以上到此的推导，都是针对 [@You Jiacheng](https://x.com/YouJiacheng) 的一份报告[@you]，在 [@谢天](https://github.com/Unakar) 的理解基础上，补充而来的。

下面回到原报告的逻辑。报告并非从 inverse-root matrix function 出发，而是先在 Gaussian-KL 目标上定义 KL-Root-Kron，再利用 SPD manifold 上的 affine-invariant metric 推出相应的在线更新。具体来说，报告首先定义
$$
\mathcal{J}(\bm P)=D_{\mathrm{KL}}\left(\mathcal N(0,\bm H)\Vert\mathcal N(0,\bm P^{-2})\right),
$$
将 Gaussian-KL 展开，可以得到
$$
\mathcal{J}(\bm P)=\frac12\operatorname{tr}(\bm P^2\bm H)-\log\det \bm P+\text{const}.
$$

另一方面，利用 KL 散度对可逆线性变换的不变性，$\mathcal{J}(\bm P)$ 也可以写成
$$
\mathcal{J}(\bm P)=D_{\mathrm{KL}}\left(\mathcal N(0,\bm P\bm H\bm P)\Vert\mathcal N(0,\bm I)\right).
$$
因此，这个目标仍然是在形式化 $\bm P\bm H\bm P\rightarrow \bm I$.

问题在于，若直接对 $\bm P$ 使用 Euclidean gradient，仍然会出现 $\bm P^{-1}$ 项，
$$
\nabla_{\bm P}\mathcal{J}(\bm P)=\frac12(\bm P\bm H+\bm H\bm P)-\bm P^{-1},
$$
解依然为 $\bm P=\bm H^{-1/2}$，依然需要求逆。接下来真正关键的一步，是在对称正定矩阵上使用 affine-invariant metric（AIRM）。在这个度量下，Riemannian gradient 是 Euclidean gradient 的 $\bm P$-共轭，即 $\bm P(\nabla_{\bm P}\mathcal{J}(\bm P))\bm P$。令当前样本对应的 $C=\bm P\bm g\bm g^\top \bm P$，便有
$$
\operatorname{grad}_{\rm AIRM}\mathcal{J}=\frac12\left[\bm P(C-\bm I)+(C-\bm I)\bm P\right],
$$
正好消去原本的 $\bm P^{-1}$ 项。接着采用 congruence update $\bm P^+=\bm A\bm P\bm A$，其中取 $\bm A=\bm I-\frac{\eta}{2}(\bm C-\bm I)$。对这个更新做一阶展开：
$$
\bm A\bm P\bm A=\bm P-\frac{\eta}{2}\left[(\bm C-\bm I)\bm P+\bm P(\bm C-\bm I)\right]+O(\eta^2),
$$

可见它正好匹配负的 AIRM natural-gradient direction。

从严格几何推导看，$\bm A=\bm I-\frac{\eta}{2}(\bm C-\bm I)$ 是与 AIRM natural gradient 一阶匹配的 congruence update；而从 matrix-function 角度看，它又恰好是 $C^{-\eta/2}$ 在 $C\approx I$ 附近的一阶近似。

[^ref]
