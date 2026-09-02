---
title: "从 Adam 到在线矩阵 Whitening"
description: "从 Adam 的对角二阶矩出发，理解在线矩阵 Whitening、KL-Root-Kron 及其 Kronecker 实现。"
date: "2026-09-01"
lang: "zh-CN"
author: "Haoyu Tang"
bibliography: "ref.bib"
---

# 从 Adam 到在线矩阵 Whitening

Adam 的核心操作之一，是利用梯度的二阶矩对更新做归一化。对第 $i$ 个参数坐标，Adam 维护 $v_{t,i}\approx \mathbb E[g_i^2]$，计算参数更新 $\Delta\theta_i\propto-\frac{m_i}{\sqrt{v_i}}$.

如果把梯度二阶矩写成 $H=\mathbb E[gg^\top]$，那么 Adam 实际上只保留了 $H$ 的对角部分
$$
H_{\mathrm{diag}}=\operatorname{diag}\bigl(\mathbb E[g_1^2],\dots,\mathbb E[g_d^2]\bigr),
$$
因此 Adam 的预条件器可以写成
$$
P_{\mathrm{Adam}}=H_{\mathrm{diag}}^{-1/2}.
$$

从这个视角出发，不免得到一个很自然的问题：

> 如果不局限于二阶矩对角元，而是保留梯度不同方向之间的相关性，会得到什么样的优化器？

不妨直接推广，
$$
P=H^{-1/2},\qquad H=\mathbb E[gg^\top],
$$
由于 $H$ 是梯度 covariance，它天然是对称半正定矩阵。为了让通常意义下的 inverse square root 有定义，下面先假设 $H\succ0$；如果 $H$ 存在退化方向，实际实现中通常对它加入一个很小的 damping，使用 $(H+\varepsilon I)^{-1/2}$。在这个假设下，$P$ 也是对称正定矩阵，即 $P\succ0$.

此时参数更新变成
$$
\Delta\theta=-\eta Pm=-\eta H^{-1/2}m.
$$
注意这里的 $-1/2$ 次方和 Newton 法里的 $H^{-1}$ 是两件不同的事情。Newton 的 $H$ 是 Hessian，并通过 $-H^{-1}g$ 求解局部二次模型；这里的 $H$ 是梯度二阶矩，目标更接近 whitening / RMS normalization。

## 1. 与其直接计算 $H^{-1/2}$，不如求解 whitening 方程

直接维护 $P=H^{-1/2}$ 的问题很明显：矩阵 inverse square root 很贵。但 $P=H^{-1/2}$ 有一个等价条件 $PHP=I$. 通过这个转化
$$
PHP=P\,\mathbb E[gg^\top]P=\mathbb E[(Pg)(Pg)^\top],
$$
可以定义预条件后的梯度 $y=Pg$，那么理想状态就是
$$
\mathbb E[yy^\top]=I.
$$
也就是说
$$
\boxed{
P=H^{-1/2}
\quad\Longleftrightarrow\quad
\text{对预条件化的梯度做白化.}
}
$$


## 2. 在线训练中，可以直接估计 $PHP$

实际训练过程中通常没有精确的 $H=\mathbb E[gg^\top]$，但每一步都有当前梯度 $g_t$. 于是可以构造 $C_t=(Pg_t)(Pg_t)^\top$，它满足
$$
\mathbb E[C_t]=P\mathbb E[g_tg_t^\top]P=PHP,
$$
所以 $C_t$ 是 $PHP$ 的一个估计。因而 $PHP=I$ 就可以在线写成 $C_t-I\to 0$. 

需要注意，真正要求的是 $\mathbb E[C_t]=I$，而 $C_t-I$ 只是一个 noisy stochastic residual。

到这里，原问题已经可以简化为 
$$
\boxed{
C_t=(P_tg_t)(P_tg_t)^\top,
\qquad
C_t-I\to 0.
}
$$

## 3. $C-I$ 包含了什么信息？

这一节，我们先来看看这个 $C-I$ 可以如何理解。由于 $C$ 是一个对称阵，不妨对其做特征分解，
$$
C=U\Lambda U^\top,\quad \Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_d),
$$
这里 $U$ 的列向量给出了当前 $C$ 的特征向量，而 $\lambda_i$ 则是对应的特征值。如果 $\lambda_i>1$，说明这个方向被放得太大，应当压缩；如果 $\lambda_i<1$，说明这个方向仍然太小，应当放大。因此 $C-I$ 实际上就是一个各向异性的白化误差。

相比之下，Adam 只能看到 $\operatorname{diag}(H)$，因此只能逐元素调整；而此处 $C$ 中的 off-diagonal 信息则允许 optimizer 识别旋转后的高方差方向和低方差方向。

## 4. 如何实际迭代更新 $P$？

现在真正的问题只剩下 $C\to I$ 应该对应什么样的 $P$ 更新？

最直接当然是定义一个平方损失函数
$$
L(P)=\frac12\|PHP-I\|_F^2.
$$
但直接对上式求梯度需要计算 $H$，复杂度很高，而且更新后也未必始终保持 $P\succ0$ 这样一个约束。

考虑到 GPU 做乘法运算比较高效，一个直接想法是乘法迭代，
$$
P^+=APA,
$$
那么只要 $P\succ0$ 并且 $A$ 可逆，就有 $P^+\succ0$. 于是问题进一步变成应该选择什么 matrix function $f$ 来定义 $A=f(C)$，使得 $C\to I$ 对应 $P\to H^{-1/2}$。

## 5. 从 matrix function 看理想 correction

暂时先考虑一个最简单的 whitening 问题。假设随机向量 $y$ 的 covariance 是 $C=\mathbb E[yy^\top]$，如果直接做 $y^+=Ay$，那么 $C^+=ACA^\top$. 若取对称矩阵 $A=C^{-1/2}$，自然直接就有 $C^+=I$. 所以 inverse square root 本身就是最自然的一种 whitening correction.

这提示我们可以把 preconditioner update 写成某种
$$
A=C^{-\alpha},
$$
不过，将这一操作放进在线优化器后，情况会有所不同。训练过程中观察到的 $C$ 往往不是精确的全局协方差，而只是由当前 mini-batch 得到的随机估计，因此直接做 $A=C^{-1/2}$ 过于激进。因此，更自然的做法不是每一步都“完全 whiten”，而是只向 whitening fixed point 移动一部分。

一种简洁的构造是 
$$
A=C^{-\eta/2},\quad 0<\eta\le 1,
$$
当然，真正计算 $C^{-\eta/2}$ 本身仍然需要 matrix function，工程上不划算。这时候就可以来引入近似。

## 6. 对 matrix function 做近似

如果当前 $C$ 已经不离 $I$ 太远，令 $C=I+E$，则 $C^{-\eta/2}=(I+E)^{-\eta/2}$. 做一阶 Taylor 展开有
$$
C^{-\eta/2}=I-\frac{\eta}{2}E+O(E^2),
$$
于是自然得到
$$
\boxed{A=I-\frac{\eta}{2}(C-I)}
$$
作为最低成本的一阶 matrix-function approximation. 

这个形式特别适合 optimizer，其中的矩阵 $C$ 来自一个 Gram matrix，运算均为正常的加减法，没有特征分解、求逆、矩阵对数这些相对昂贵的操作。

当然如果愿意多付出一些计算开销，可以继续保留二阶项：
$$
C^{-\eta/2}\approx I-\frac{\eta}{2}(C-I)+\frac{\eta(\eta+2)}{8}(C-I)^2,
$$
特别地，取 $\eta=1$，
$$
C^{-1/2}\approx I-\frac12(C-I)+\frac38(C-I)^2.
$$

可以把以上方法理解成
$$
\boxed{
\text{用低阶 polynomial 近似 inverse-root matrix function}
}
$$
而这和 Muon / Newton-Schulz 一类方法的工程哲学其实很接近：避免 eig/SVD，把 matrix function 改写成少量 GEMM。

## 7. 如何在实现中维护 preconditioner 更新？

上面的这种乘法迭代可以直接作用在 $P$ 上，但在实际实现中，我们还希望每一步都显式地保持 $P$ 的对称正定结构。一个自然的参数化方式是引入其平方根因子 $R$，写成 $P=R^\top R$。这样，更新不再直接修改 $P$，而只需要做更新 $R^+=RA$.

由于 $(RA)^\top(RA)=A^\top R^\top RA$，而这里的 $A=f(C)$ 是 $C$ 的对称 matrix function，满足 $A^\top=A$，所以新的因子确实对应于 $P^+=APA$.

于是，一个极简版的在线算法可以写成

$$
\begin{aligned}
y&=R^\top Rg,\\
C&=yy^\top,\\
A&=I-\frac{\eta}{2}(C-I),\\
R&\leftarrow RA.
\end{aligned}
$$

然后用新的 $P=R^\top R$ 预条件来做真正的参数更新。值得注意的是，这里第一个 $y$ 是用当前的 raw gradient $g$ 来做计算，而真正的更新 $u$ 则可以用新的预条件来处理梯度的动量 EMA 来算。这是因为动量中包含参数跨时间的 correlation，因此更新 $P$ 用 raw gradient 更能反映当前的 whitening 状态。

## 8. 从向量预条件器到矩阵参数

到目前为止，我们讨论的都是向量情形：
$$
g\in\mathbb R^d,\qquad P\in\mathbb R^{d\times d},
$$
然而，在真实神经网络中，参数通常是形如 $W\in\mathbb R^{m\times n}$ 的矩阵，将其展平维度会变成 $d=mn$，对应的 full preconditioner 需要具有 $P\in\mathbb R^{mn\times mn}$ 这样的 shape，显然计算开销过大，无法直接实现。因此，需要引入结构化 approximation。一种自然的选择是 Kronecker factorization
$$
P_{\mathrm{full}}=P_b\otimes P_a,
$$
其中 $P_a\in\mathbb R^{m\times m},P_b\in\mathbb R^{n\times n}$. 对于矩阵梯度 $G\in\mathbb R^{m\times n}$，利用 Kronecker 与向量化的恒等式，有
$$
P_{\mathrm{full}}\operatorname{vec}(G)=\operatorname{vec}(P_aGP_b).
$$
于是可以直接定义预条件后的矩阵梯度 $Y=P_aGP_b$，这样就不必构造巨大的 $mn\times mn$ covariance，而只需分别考察两个 mode 上的 covariance：
$$
C_a=\frac1nYY^\top,\quad C_b=\frac1mY^\top Y.
$$
理想的 fixed point 对应于
$$
\mathbb E[C_a]=I_m,
\qquad
\mathbb E[C_b]=I_n.
$$

可以证明，这两个 stationary equations 与 idealized KL-Shampoo 的 Kronecker covariance fixed point 等价。于是，每个 factor 都可以进行与前面相同的一阶更新：

$$
A_a
=
I-\frac{\eta_a}{2}(C_a-I),
$$

$$
A_b
=
I-\frac{\eta_b}{2}(C_b-I),
$$

同时维护

$$
P_a=R_a^\top R_a,
\qquad
P_b=R_b^\top R_b,
$$

并更新

$$
R_a\leftarrow R_aA_a,
\qquad
R_b\leftarrow R_bA_b.
$$

这就得到一个实际可实现的 Kronecker 版本：只需存储两个较小的 factor，而不需要面对 $mn\times mn$ 的 full matrix。
