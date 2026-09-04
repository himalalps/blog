---
title: "从 Linear Attention 到 Tensor Transformer：如果 Transformer 不再需要显式 Activation？"
description: "从 DeepSeek-V4 对 SwiGLU 开销的讨论出发，借助 Bilinear FFN、Linear Attention 与多项式函数族，分析 Transformer 是否需要显式激活函数。"
date: "2026-09-03"
lang: "zh-CN"
author: "Haoyu Tang"
bibliography: "ref.bib"
---

# 从 Linear Attention 到 Tensor Transformer：如果 Transformer 不再需要显式 Activation？

DeepSeek-V4 报告 [@deepseekai2026deepseekv4highlyefficientmilliontoken] 里有这么一段话：

> We propose replacing SwiGLU with a low-cost element-wise activation that involves no exponential or division operations. This directly reduces the overhead of post-GEMM processing, preventing the GEMM pipeline from being stalled by activation function computation, thereby enhancing overall computational throughput and resource utilization.

当下的前沿 MoE 模型基本都采用 SwiGLU，其中的非线性激活函数 SiLU 需要计算 $\on{Sigmoid}(x)=1/(1+\exp(-x))$，包含 GPU 并不高效的指数和除法操作。但即使如此，DeepSeek-V4 也只是把替换 SwiGLU 作为 proposal，并没有在最终架构中真正对这一模块动刀子。

最近看到的一篇文章 [@tensor-transformer-variants]，则尝试直接去掉这里的 SiLU，文章的实验结果也有些出人意料：对于约 500M 参数的模型，用 bilinear layer 替代 SwiGLU 几乎没有明显的损失，这里的损失甚至小于把 Softmax Attention 替换为 Bilinear Attention. 本文尝试来理解这个结果，并进一步讨论可能的潜在探索方向。

## 1. Bilinear FFN：去除 SiLU 的 SwiGLU

先来看 Tensor Transformer 对 FFN 做的改动。现代 LLM 中常见的 SwiGLU 可以写成
$$
\operatorname{SwiGLU}(\bm x)=\bm W_o\left[\operatorname{SiLU}(\bm W_g\bm x)\odot\bm W_u\bm x\right].
$$
其中 $\bm W_g$ 对应 gate branch，$\bm W_u$ 对应 up branch. Tensor Transformer 使用的 Bilinear FFN 则直接去掉 SiLU
$$
\operatorname{Bilinear}(\bm x)=\bm W_o\left[(\bm W_1\bm x)\odot(\bm W_2\bm x)\right].
$$
看上去只是少算了一个 $\operatorname{SiLU}$，但从函数形式上看这却是一个很激进的改动：整个 FFN 不再包含任何显式的非多项式激活函数. 笔者看到这里的第一反应是，这不就退化成线性层了吗？

答案其实是否定的。虽然 $\bm W_1\bm x$ 和 $\bm W_2\bm x$ 都是 $\bm x$ 的线性函数，但二者的逐元素乘积是二次函数。设 FFN 的 intermediate size 为 $m$，对于第 $k$ 个中间维度有
$$
(\bm W_1\bm x)_k(\bm W_2\bm x)_k=\sum_{i,j}(\bm W_1)_{ki}(\bm W_2)_{kj}\bm x_i\bm x_j.
$$
进一步考虑 Bilinear FFN 输出 $\bm y$ 的第 $l$ 个分量。用 $\bm W_o$ 对上述 $m$ 个中间维度加权求和，有
$$
\begin{aligned}
\bm y_l&=\sum_{k=1}^{m}(\bm W_o)_{lk}(\bm W_1\bm x)_k(\bm W_2\bm x)_k\\
&=\sum_{i,j}\left[\sum_{k=1}^{m}(\bm W_o)_{lk}(\bm W_1)_{ki}(\bm W_2)_{kj}
\right]\bm x_i\bm x_j\\
&=\sum_{i,j}\bm T_{lij}\bm x_i\bm x_j,
\end{aligned}
$$
其中 $\bm T_{lij}=\sum_{k=1}^{m}(\bm W_o)_{lk}(\bm W_1)_{ki}(\bm W_2)_{kj}\in\mathbb R^{d_{\mathrm{out}}\times d\times d}$ 是一个三阶 tensor，三个矩阵 $\bm W_o,\bm W_1,\bm W_2$ 实际上给出了 $\bm T$ 的一个低秩分解。Bilinear FFN 因而是经过低秩参数化的 quadratic map，去掉 SiLU 并未把 MLP 变成一个单线性层，而是把用这里的逐元素积实现了二阶的非线性映射，而多层叠加后也可以因而产生更高阶的多项式交互。

从这个角度再看 SwiGLU，会发现其中本来就同时包含两种非线性来源：SiLU 提供显式的激活函数，而 gate 与 up branch 的逐元素乘法提供隐式乘性非线性。这其实在六年前就已经被 Noam Shazeer 进行过对比实验 [@shazeer2020gluvariantsimprovetransformer]，在 Loss 以及下游任务上相比最优的带激活函数的 GLU，这种 Bilinear FFN 也只是差了一些，总体上是 GEGLU≈SwiGLU>ReGLU≈Bilinear>GLU>普通 ReLU/GELU/Swish FFN. 但值得注意的是，当时的实验还没有进入 MoE 时代，整体规模也还比较小。

经典 universal approximation theorem 告诉我们，对于标准单隐藏层前馈网络，在适当条件下，非多项式 activation 与其稠密函数逼近能力密切相关。SiLU、ReLU 和 GELU 都属于这里的非多项式 activation.

Bilinear FFN 显然不同。单层映射 $\bm W_o[(\bm W_1\bm x)\odot(\bm W_2\bm x)]$ 严格来说只是二次多项式，因此固定为单层时，它当然不具有普通非多项式 activation MLP 那样的函数族。但这并不意味着深层网络也只能表达低阶，如果每层都是二次映射，在暂时忽略 residual 和 normalization 的情况下，组合起来会使这里的阶数随深度快速增长：第 $1$ 层最高为 $2$ 阶，第 $2$ 层最高为 $4$ 阶，到第 $L$ 层则可以达到 $2^L$ 阶。

传统 MLP 直接使用非线性激活，通过增加 width 就可以构造复杂函数；Bilinear network 则更多依靠 $\bm x_i\bm x_j$ 及其深层组合所产生的高阶 interaction. 当前一些模型的深度已经达到几十上百，那么这么深的一个模型中如果换用 Bilinear FFN 其实也已经可以产生非常丰富的高阶多项式。

## 2. Bilinear Attention：基于二阶 Map 的 Linear Attention

Tensor Transformer 还把这种构造推广到了 Attention 中。先考虑标准 Attention
$$
\bm Y=\operatorname{Softmax}(\bm Q\bm K^\top)\bm V,
$$
Bilinear Attention 不再对一组 query-key score 使用 Softmax，而是先构造两组 query 和 key 投影，再将这里的两张 score matrix 逐元素相乘
$$
\bm A=(\bm Q_1\bm K_1^\top)\odot(\bm Q_2\bm K_2^\top),\qquad\bm Y=\bm A\bm V,
$$
其中 $\bm Q_1,\bm Q_2$ 是两个 query branch，$\bm K_1,\bm K_2$ 是对应的 key branch，$\odot$ 表示 Hadamard product. 对于第 $i$ 个 query 位置和第 $j$ 个 key 位置，Attention weight 为
$$
\begin{aligned}
\bm A_{ij}&=(\bm Q_1\bm K_1^\top)_{ij}(\bm Q_2\bm K_2^\top)_{ij}\\
&=(\bm q_{1,i}^\top\bm k_{1,j})(\bm q_{2,i}^\top\bm k_{2,j}),
\end{aligned}
$$
这里 $\bm q_{1,i}$ 是 $\bm Q_1$ 的第 $i$ 行，$\bm k_{1,j}$ 是 $\bm K_1$ 的第 $j$ 行，$\bm q_{2,i}$ 和 $\bm k_{2,j}$ 同理. 这和 Bilinear FFN 的结构完全一致：每个 branch 本身都是线性的，二者相乘后则产生二阶 interaction. 利用 Kronecker product 恒等式
$$
(\bm a^\top\bm b)(\bm c^\top\bm d)=(\bm a\otimes\bm c)^\top(\bm b\otimes\bm d),
$$
可以为两个 branch 的组合定义二阶 feature map
$$
\phi(\bm q)=\bm q_1\otimes\bm q_2,\qquad\phi(\bm k)=\bm k_1\otimes\bm k_2,
$$
这样就得到 $\bm A_{ij}=\phi(\bm q_i)^\top\phi(\bm k_j)$，进而有 $\bm A=\bm\Phi(\bm Q)\bm\Phi(\bm K)^\top$，这是标准的 Linear Attention 形式。

不过，[@tensor-transformer-variants] 的初步实验表明，用 Bilinear Attention 替换 Softmax Attention 会带来相当明显的性能损失。一个直观解释是二者使用了不同阶数的 kernel：如果两个 branch 的维度都是 $d_h$，那么 $\phi(\bm q)=\bm q_1\otimes\bm q_2\in\mathbb R^{d_h^2}$，对应一个有限维的二阶 feature map；而 $\exp(\bm q^\top\bm k)$ 经过泰勒展开后，可以解释为一个无限维 kernel。

但 polynomial degree 可能并不是这里最关键的区别。与 Bilinear FFN 类似，上一层的表示本身已经可以是输入的高阶函数，继续叠加 Bilinear Attention 仍会不断提高整个网络能够形成的 interaction 阶数。Attention 里真正更重要的，是 Softmax 在每一层施加的 routing 机制。

为了看清这一点，先把标准 Softmax Attention 写成逐位置的形式。设 $i$ 表示 query 位置，$j$ 和 $\ell$ 表示 key 位置，$\bm v_j$ 是 $\bm V$ 的第 $j$ 行，并记 query-key logit 为 $s_{ij}=\bm q_i^\top\bm k_j$。忽略常见的 $1/\sqrt{d_h}$ 缩放后，有
$$
\bm y_i=\frac{\sum_j \exp(s_{ij})\bm v_j}{\sum_\ell \exp(s_{i\ell})}=\sum_j p_{ij}\bm v_j,
$$
其中
$$
p_{ij}=\frac{\exp(s_{ij})}{\sum_\ell\exp(s_{i\ell})},\qquad\sum_j p_{ij}=1.
$$
可以看出，Softmax weight 不只由第 $i$ 个 query 和第 $j$ 个 key 决定，还取决于其余 keys 提供了哪些竞争项。而 Bilinear Attention 则直接使用 dot-product similarity，而没有跨 key 的 normalization。

这会直接改变 token 之间的竞争方式。对于 Softmax Attention，任意两个 key 位置 $j$ 和 $\ell$ 的相对权重满足
$$
\frac{p_{ij}}{p_{i\ell}}=\exp(s_{ij}-s_{i\ell}).
$$
logit 上有限的差异经过指数映射后可以变成悬殊的权重比，因此 Softmax 很容易形成尖锐的选择分布。Bilinear Attention 则不天然提供这种 exponential sharpening，且 $\bm A_{ij}$ 可以为负，$\sum_j\bm A_{ij}$ 也不固定为 $1$，所以输出尺度还会直接受到 query/key norm、score distribution 和序列长度的影响。

而今天很多前沿模型采用的线性注意力 (GDN/KDA) kernel 其实是更低阶的直接映射，复杂度比这里的 Bilinear Attention 还要更低。那么既然更简单的线性注意力能 scale 上去，没有道理表达能力更强的 Bilinear Attention 反而不行，笔者想到前沿模型往往是把线性注意力和全注意力在深度上面做混合，也许这里的实验中只用 Bilinear Attention 相比 Softmax Attention 的性能下降，并不是因为 Bilinear Attention 本身不够强，而是缺少了 hybrid 的设计。

## 3. 从 Hybrid Attention 到 Hybrid FFN

现有前沿模型已经给出了一种解法，并不要求 Linear Attention 完全替代 Full Attention，而是在层数上做 hybrid. 一种常见配置是让 Linear Attention 与 Full Attention 的层数比例达到 $3:1$，甚至让 Linear Attention 占比更高。大多数层使用便宜的 recurrent state 负责信息传播，每隔若干层插入一次 Full Attention，补充精确的 token selection 和全局交互。实践已经说明，受限模块可以占据绝大多数层，只要模型中仍然保留少量表达能力完整的路径。

同样的思路也可以搬到 FFN. 从 SwiGLU 切换到 Bilinear FFN，只是把 $\operatorname{SiLU}(\bm a)\odot\bm b$ 换成 $\bm a\odot\bm b$，multiplicative skeleton 仍然完整保留，实验中的 gap 也比 Attention 一侧更小。因此进一步地，是否能让大部分 FFN 使用 Bilinear map，再保留少量 SwiGLU 作为非多项式的补充路径？

最直接的方案当然也可以沿深度做 hybrid，例如每三层里用 Bilinear 来替代原本的 SwiGLU，然后再插入一层使用完整的 SwiGLU. 但 MoE 本身已经提供了另一种更自然的比例结构。

## 4. MoE 天然适合做 Hybrid FFN

典型 MoE 中，每个 routed expert 本质上都是一个 SwiGLU FFN：
$$
E_j(\bm x)=\bm W_{o,j}\left[\operatorname{SiLU}(\bm W_{1,j}\bm x)\odot\bm W_{2,j}\bm x\right].
$$

现有的 MoE 与普通 dense FFN 的不同之处在于，它天然把计算拆成两类：数量很多但每个 token 只激活少数的 routed experts，以及数量很少但对所有 token 始终激活的 shared experts. 这恰好对应 hybrid Attention 中“大量受限路径 + 少量完整路径”的结构，只不过比例不再沿层数划分，而是在同一层的不同 expert 之间划分。

由于 MoE 的绝大多数参数都位于 routed FFN 中，一个直接的设计就是让大量 routed experts 使用 Bilinear FFN，而让少量 shared experts 保留 SwiGLU. 这样就可以在每一层中直接实现 hybrid 的比例，而不必沿深度做插入。

这个结构还有一点不同：按层做 hybrid 时，一个 token 需要经过若干受限层后才会遇到 Full Attention；而在 MoE 版本中，每个 token 在每一层都同时经过 Bilinear Routed Experts 和 SwiGLU Shared Experts. 非多项式路径不是周期性出现，而是始终存在。

从函数类角度，若所有模块都是 bilinear，整个网络更接近一个 polynomial system. 加入 SwiGLU shared expert，则相当于每层都保留一条经过 $\operatorname{SiLU}(\bm W\bm x)$ 的 non-polynomial path，模型因而不再局限于纯 polynomial composition.

## 5. 会有什么收益？

回到最初的系统动机，把 SwiGLU 换成 Bilinear FFN 并不会直接减少参数量。两个输入 projection 和一个输出 projection 仍然存在，主要 GEMM 的形状也没有改变；如果只看参数量和通常的 FLOPs 统计，这次替换省下的只是 SiLU 对应的逐元素运算。

但 FLOPs 并不等于实际运行时间，它不反映访存、kernel launch，以及 GEMM 与后处理算子之间的流水线停顿。这个差异在细粒度 MoE 中尤其可能被放大：单个 expert 的 GEMM 规模较小，低精度矩阵乘法又已经得到高度优化，此时 post-GEMM 阶段的 SiLU 即使只占很少的理论 FLOPs，也可能形成不可忽略的 wall-clock 开销。正是 DeepSeek-V4 在开头强调的系统收益启发了本文去除指数/除法的想法。

当然，从 SwiGLU 去除 SiLU 的后果也是系统性的，EP 相关实现也得重新设计，尤其是通信/计算 overlap 的部分，routed shared expert 比例也会影响这里的 overlap 能否掩盖开销，也许有很多和现有架构不一样的点值得发现与尝试。

[^bib]

[^ref]

[^giscus]
