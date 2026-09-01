---
title: "模型架构：1. MoE"
description: "模型架构系列第一篇，围绕 MoE 的基本结构、SwiGLU 与细粒度专家设计，梳理共享专家、路由函数、负载均衡和专家并行等关键取舍。"
date: "2026-08-30"
lang: "zh-CN"
author: "Haoyu Tang"
bibliography: "ref.bib"
---

# 模型架构：1. MoE

> 在回顾不同模型的架构时，笔者发现前沿的主流模型很多设计有不少共通之处。接下来一系列文章，笔者尝试从不同模块角度来分别浅析不同的设计选择，算是一个梳理，也欢迎读者指出任何错漏之处。这第一篇文章，就从 MoE 开始。

## 基本结构

今天的主流前沿模型，基本都采用了 MoE 架构。这可以看作把一层 block 分成并行的路径，每个路径单独包含一个原始 Transformer [@vaswani2023attentionneed] 里的 FFN 层。后者本质上只是一个两层的 MLP，对输入 $x\in\R^{d}$（行向量）有
$$
\begin{equation}
\bm y=\phi(\bm x \bm W_u + \bm b_1) \bm W_d + \bm b_2,
\end{equation}
$$
其中 $\bm W_u\in\R^{d\times \dff}, \bm W_d\in\R^{\dff\times d}$，而 $\phi$ 是非线性激活函数，[@vaswani2023attentionneed] 里用的是 $\on{ReLU}(x)=\max(0,x)$. 这里有一个 $\dff$，最初版本 $d=512, \dff=2048$，也就是 $\dff=4d$. 

那么一个显然的问题是为啥这里需要做一个中间层上的 up/down projection，而不是直接在原本的 $d$ 维空间里做 MLP 呢？ 一种解释 [@2025whyupdown] 是为了避免由于非线性激活函数 $\phi$ 造成的矩阵降秩，对于一个 $d$ 维的输入，假设变换到 $\dff$ 维，使用 ReLU 激活，那么此时降秩到低于 $d$ 的概率是一个组合问题，仍激活的个数记为 $K$，则有 $K\sim {\rm Binomial}(n,1/2)$，那么降秩的概率为
$$
\begin{equation}
{\rm Pr}[K<d]=\sum_{k=0}^{d-1} \binom{\dff}{k} 2^{-\dff},
\end{equation}
$$
可以验证，当 $\dff>3d$ 时，降秩概率就已经接近 0 了，而此处取 4 是一个更适合的经验值。

更现代的主流 MoE 设计里，单个 MLP 的计算公式通常为：
$$
\begin{equation}
\bm y=[\operatorname{SiLU}(\bm x\bm W_g)\odot \bm x\bm W_u] \bm W_d,
\end{equation}
$$
其中 $\bm W_g,\bm W_u\in\R^{d\times \dff},\bm W_d\in\R^{\dff\times d}$.

这相比原版有一些改动：
1. 去除每一层的 bias，一方面 bias 内存比较密集但对于性能影响不大因而去除可以节省访存，另一方面 bias 可能引入计算稳定性问题。
2. 使用 SwiGLU 替代原始的 FFN，SwiGLU 里一方面引入了门控机制，另一方面换用了 SiLU 激活函数 $x\sigma(x)$.

而因为此处相比原始的 FFN 多了一个矩阵 $\bm W_g$，为了控制整体的 FLOPs 不变，一些模型会选择将 $\dff$ 缩小到原来的 $2/3$，也就是此时的 $\dff=8d/3$. 当然精确的 $8d/3$ 未必是适合硬件的设计（切成 tile 有尾块），最终取的值可能是通过性能 benchmark 取得的更优点。

不过这也不意味着 SwiGLU 就一定是最优的选择，DeepSeek-V4 的 report 里就认为这个计算开销比较大，需要做指数/除法这些不是很适合 GPU 并行加速的运算。不过目前各家模型似乎都还是用这个设计，可能暂时也没有找到更好的替代品。一些这个基础上的修补可能是类似 Kimi-K3 那样的 tanh soft cap，控制 gate/up branch 的输出范围，避免尺度爆炸。

## 稀疏计算

介绍完单个 expert 的设计之后，就需要考虑如何将多个并行的 expert 组合起来。MoE 的核心是稀疏计算：每个 token 只激活一部分专家。这样可以在不增加每个 token 的计算量的情况下，增加模型的总参数量，从而提升模型的表达能力。

大体上，现有的 MoE 都可以看作以下的逻辑
$$
\begin{equation}
\bm y = \bm x + \sum_{i=1}^{N_s} E_i^{(s)}(\bm x) + \sum_{i=1}^{N_r} g_{i} E_i^{(r)}(\bm x),
\end{equation}
$$
其中 $N_s$ 是共享专家的数量，$N_r$ 是路由专家的数量，$E_i^{(s)}$ 和 $E_i^{(r)}$ 分别是共享专家和路由专家的计算函数，$g_{i}$ 是路由专家的激活权重（通常只有 Top-k 不为 0）。共享专家对所有 token 都是常开的，而路由专家则根据 token 的特征动态选择。有些模型没有共享专家，可以看作 $N_s=0$；而传统的 dense 模型则可以看作 $N_r=0$.

### LatentMoE

普通 MoE 的每个 expert 都直接接收完整的 hidden state，因此当 routed expert 很多时，矩阵乘法和跨卡通信都会随 $d$ 一起变大。LatentMoE 只把 routed 分支送入较窄的 latent space：
$$
\begin{equation}
\bm z=\bm x\bm W^{\downarrow},\qquad
\bm h=\sum_{i=1}^{N_r}g_iE_i^{(r)}(\bm z),\qquad
\bm y=\bm x+\sum_{i=1}^{N_s} E_i^{(s)}(\bm x)+ \bm h\bm W^{\uparrow}.
\end{equation}
$$
其中 $\bm W^{\downarrow}\in\R^{d\times\ell},\bm W^{\uparrow}\in\R^{\ell\times d}$，且通常 $\ell<d$. shared expert 仍可在完整的 $d$ 维上工作，只有 routed experts 在 $\ell$ 维内计算；代价是多了一对投影矩阵的计算。

Kimi K3 进一步提出 **Stable LatentMoE** [@kimiteam2026kimik3openfrontier]：在计算 Top-k 加和 $\bm h$ 之后再用 $\on{RMSNorm}$ 控制尺度，避免稀疏路由叠加投影后造成尺度漂移。

## 负载均衡

前述介绍里还省略了一个关键问题，如何计算每个 routed expert 的激活权重 $g_i$. 路由器通常是先计算每个 token 对各 expert 的分数，再取 top-k，做归一化。

考虑某一个 token $x$，有
$$
\begin{equation}
\bm s=\phi(\bm{x}\bm{W}_r),\quad g_i=\frac{\bm s_i}{\sum_{j\in\mathcal{T}}\bm s_j}, i\in\mathcal{T},\quad \mathcal{T}=\on{argtop}_k(\bm s+\bm b),
\end{equation}
$$
其中 $\bm W_r\in\R^{d\times N_r}$ 是路由矩阵，$\phi$ 是一个激活函数，通常用的是 $\on{Sigmoid}(\cdot)$，但也有模型如 DeepSeek-V4 用的是 $\on{Sqrt}(\on{Softplus}(\cdot))$，$\bm b\in\R^{N_r}$ 是 Aux-loss-free/QB 等方法里的 bias，如果模型没有采用 Loss-free，则恒有 $\bm b=0$.

一些文章中把这里的 $\bm W_r$ 看作各个 expert 的一个质心，但实际目前的各家模型里面 $\bm W_r$ 还是和实际各个 routed expert 解耦的，不过也有一些 paper[@wu-etal-2026-union] 在尝试对这个 router 和 expert 结合上做探索，笔者认为这方面还是很有探索空间的。

若只优化语言模型损失，router 很容易塌缩到少数 expert：热门 expert 产生队列和通信热点，冷门 expert 又得不到足够梯度。因此负载均衡是在容量、路由质量和通信成本之间取折中。

最经典的方法是 **Aux loss** [@switchtransformer]。设一个训练 batch 中 expert $i$ 被选中的 token 比例为 $F_i$，router 激活权重之和为 $P_i$，常见形式为
$$
\begin{equation}
L_{\mathrm{aux}}=\alpha N\sum_{i=1}^{N}F_i P_i.
\end{equation}
$$
它简单、可微，但会给主目标额外的梯度方向：$$\alpha$$ 太大损害路由质量，太小又压不住热点。

**Aux-loss-free** [@wang2024auxiliarylossfreeloadbalancingstrategy] 把均衡从目标函数移到路由决策中：为每个 expert 维护只影响 top-k 排序的 bias，依据近期实际负载增减
$$
\begin{equation}
\bm b\leftarrow \bm b-\gamma\on{sign}(\bm F-\bm Q),
\end{equation}
$$
这里 $Q$ 是目标负载比例，一般取 $[1/N_r, 1/N_r, \dots]$，$\gamma$ 是更新步长。最终权重仍使用原始 router 分数计算，因此不会把人为正则项混入 token 的训练梯度。

**Quantile balancing** [@kexuefm-11626] 则利用当前 batch 里的 router score 经验分位数，反推出一个使某个 expert 被选中的阈值/偏置。为了避免过拟合到某个 batch 里的情况，实际训练中会做 EMA. 本质上是将 Expert Choice 转化为 Bias 形式：以每个 Expert 的第 $mk/n+1$ 大元素（等价为求 $1−k/n$ 分位数）作为阈值 $\bm \beta_i=-\bm b_i$，将 Expert Choice 里每个 Expert 选 Top-$mk/n$ 重新表述为 Token Choice 的 $\bm s_i−\bm \beta_i>0$ 才激活。为了规避信息泄漏问题，该方法还将 $\bm\beta$ 的更新延迟到激活决策之后。

## 专家并行

在大规模训练时，往往会使用专家并行 (Expert Parallelism, EP) 来加速 MoE 训练。但这样又会引入不同节点间的通信开销，DeepSeek-V4 里介绍了一种方式[@deepseekai2026deepseekv4highlyefficientmilliontoken]，尝试将专家拆分成 waves，将通信开销隐藏在计算之后。

具体来说，通信上包括 Dispatch (把计算分配到各个节点上) 和 Combine (把各节点计算结果聚合起来) 两个阶段。而计算上包括 Linear1, Act, Linear2 三个阶段。前一个 wave 中在做计算时，就可以开始下一个 wave 的通信，这样类似一个流水线一样，把通信时间隐藏在计算时间中。整体上能有 1.5~2 倍的加速。

Kimi-K3 里还引入了**动态冗余 Expert** 方法，这是为了避免 token 在不同 expert 之间的路由天然不均匀。即使全局 token 数固定，不同 rank 最终接收到的 token 数仍可能差很多。设序列长度为 $S$，每个 token 选择 $K$ 个 expert，则总 routed token-expert pair 数为 $SK$。如果 EP size 为 $R$，理想情况下每个 rank 应处理 $ S K/R $ 个 token-expert pair. 这里的结论是：对于任意 token 路由分布，总能找到一个完全均衡的分配方案，并且每个 rank 最多只需要额外放置大约 $E/R$ 个冗余 expert. 

这样带来的好处也是显然的，通信开销不再受影响，而是变成了一个固定值；每个 rank 上投入计算的 shape 也是可以提前定下的，不受 token 分布影响。

## 工程设计

在大规模 MoE 训练中，判断通信能否被计算完全掩盖，需要看计算吞吐 $C$ 与通信带宽 $B$ 之间的比例是否匹配。
若实际计算量和通信量分别为 $V_{\mathrm{comp}}$ 与 $V_{\mathrm{comm}}$。如果希望通过计算与通信重叠，将通信开销完全隐藏，那么需要满足

$$
\frac{C}{B}
\le
\frac{V_{\mathrm{comp}}}{V_{\mathrm{comm}}}.
$$

右侧本质上描述了一个 workload 自身的 computation-to-communication ratio：每传输一个 Byte 的数据，能够对应多少 FLOPs 的计算。只有硬件的计算/带宽比例不超过这个值，通信才有机会被计算完全覆盖。

对于一个普通的 SwiGLU，忽略占比较小的激活函数相关计算量，有 $V_{\mathrm{comp}} = 6dm_r$ FLOPs. 这里 $m_r$ 是单个 expert 的 intermediate size.

另一方面，EP 中 token 经过两阶段通信：Dispatch 将 hidden state 发往对应 expert，使用 FP8，需要 $d$ Bytes；Combine 将 expert 输出传回原 GPU，使用 BF16，需要 $2d$ Bytes. 因此总通信量约为 $V_{\mathrm{comm}} = 3d$ Bytes.

代入 DeepSeek-V4 的配置，$m_r=3072$，有
$$
\frac{V_{\mathrm{comp}}}{V_{\mathrm{comm}}}
=
\frac{6dm_r}{3d}
=
2m_r=6144 \mathrm{FLOPs/Byte},
$$
这个数字可以直接用来理解硬件应该如何配置。在这种 workload 下，每增加 1GB/s 的通信带宽，大约可以匹配 6.144TFLOP/s 的计算能力。

这也意味着，网络带宽并不是越高越好。当带宽已经达到上述平衡点，使通信能够被计算完全隐藏之后，再继续增加网络带宽，并不会带来对应比例的性能提升。此时真正的瓶颈已经从通信转移到了计算侧。

因此，对于面向 MoE 的硬件设计，更合理的目标是围绕具体 workload 的 computation-communication ratio 寻找平衡点。计算单元、HBM、片间互联以及封装资源应该按照这一比例协同设计。

## 前沿模型

这里尝试统计各前沿模型的 MoE 设置，此处单个 routed/shared expert 的 intermediate size 定为 $m_r$，可以把实际被激活的所有 expert 的 $m_r$ 加和当作原始的 $\dff$，那么就可以计算对应的 routed ratio 和 active ratio.

:::table{vlines="1,4,6"}
| 模型                                                                                            |  $d$  | $m_r$ |  $k/N_r$ | $k m_r$ | $r_{\text{route}}$ | $N_s$ | $r_{\text{active}}$ |
| ----------------------------------------------------------------------------------------------- | :---: | :---: | -------: | :-----: | ------------------ | :---: | ------------------- |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)                      | 7168  | 3072  | 16 / 896 |  49152  | 6.857*             |   2   | 7.714*              |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json)        | 8192  | 2048  | 10 / 512 |  20480  | 2.500              |   1   | 2.750               |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)     | 7168  | 3072  |  6 / 384 |  18432  | 2.571              |   1   | 3.000               |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)                         | 6144  | 2048  |  8 / 256 |  16384  | 2.667              |   1   | 3.000               |
| [Hy4-preview](https://huggingface.co/tencent/Hy4-preview/blob/main/config.json)                 | 6144  | 2048  |  8 / 256 |  16384  | 2.667              |   1   | 3.000               |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)             | 4096  | 2048  |  8 / 288 |  16384  | 4.000              |   1   | 4.500               |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) | 4096  | 2048  |  6 / 256 |  12288  | 3.000              |   1   | 3.500               |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)      | 2560  |  640  | 10 / 512 |  6400   | 2.500              |   1   | 2.750               |

$$
\begin{equation}
r_{\text{route}}=\frac{k m_r}{d},
\qquad
r_{\text{active}}=\frac{(k+N_s)m_r}{d}.
\end{equation}
$$

可以看到除了 Kimi-K3 和 GLM-5.3-Flash 之外，其他模型的 routed ratio 都在 2.5~3.0 之间，基本都是取的符合 8/3 附近一个比较适合硬件的值。

Kimi-K3 因为采用了 Stable LatentMoE，表中的 7.714 是实际激活 intermediate width 的和原始 residual width 比例，如果从计算量角度考虑，实际比例并没有那么多，
$$
\begin{equation}
r_{\text{eff}}=\frac{P_{\text{active}}}{3d^2}\approx \frac{3N_sdm_r+3k\ell m_r+2d\ell}{3d^2}=\frac{N_sm_r}{d}+\frac{k\ell m_r}{d^2}+\frac{2\ell}{3d},
\end{equation}
$$
上式三项分别是 shared experts, latent routed experts 以及额外的 down/up projections 的计算量占比。这里的系数 3 对应 SwiGLU 中的三个矩阵。代入 Kimi-K3 的配置 $d=7168,\ell=3584,m_r=3072,k=16,N_s=2$，大概可以得到 $r_{\text{eff}}\approx 4.62$. 单独考虑 routed path，这个比例在 3.76，实际上和其他模型相比略大一些但并没有很夸张。

比较例外的倒是 GLM-5.3-Flash，它相比 GLM-5.3 没有调整 expert 的 intermediate size，而是把 $d$ 调小到原来的 $2/3$，导致最后得到的 $r_{\text{route}}$ 以及 $r_{\text{active}}$ 都比原本大了 $3/2$. 笔者的理解是 GLM-5.3-Flash 因为采用了 mHC，所以自然可以把 hidden size 调小，降低其他每个部分的成本，而同时意味着把更多的容量分配给 FFN.

## 整体占比

当前主流模型里，MoE 层占总参数的绝对大头，但由于存在稀疏计算，实际激活参数中 MoE 占比并没有那么高，下表试统计当前一些前沿模型的参数情况。

:::table{fit vlines="1,5"}
| 模型                                                                                            |  $d$  | $m_r$ | $k/N_r$ | $N_s$ |  $P_{\text{exp}}$   | $P_{\text{router}}$ |        $P_{\text{other}}$        | $P_{\text{MoE}}^{\text{layer}}$ | $P_{\text{MoE}}^{\text{layer,active}}$ |
| ----------------------------------------------------------------------------------------------- | :---: | :---: | ------: | :---: | :-----------------: | ------------------: | :------------------------------: | :-----------------------------: | :------------------------------------: |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)                      | 7168  | 3072  |  16/896 |   2   | 33.03M(r) 66.06M(s) |              *6.42M | 51.38M(latentproj) 3584(RMSNorm) |             29.785B             |                718.41M                 |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json)        | 8192  | 2048  |  10/512 |   1   |       50.33M        |               4.19M |        8192(shared Gate)         |             25.824B             |                557.85M                 |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)     | 7168  | 3072  |   6/384 |   1   |       66.06M        |              *2.75M |                -                 |             25.436B             |                465.18M                 |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)                         | 6144  | 2048  |   8/256 |   1   |       37.75M        |              *1.57M |                -                 |             9.703B              |                341.31M                 |
| [Hy4-preview](https://huggingface.co/tencent/Hy4-preview/blob/main/config.json)                 | 6144  | 2048  |   8/256 |   1   |       37.75M        |              *1.57M |                -                 |             9.703B              |                341.31M                 |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)             | 4096  | 2048  |   8/288 |   1   |       25.17M        |              *1.18M |                -                 |             7.274B              |                227.67M                 |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) | 4096  | 2048  |   6/256 |   1   |       25.17M        |              *1.05M |                -                 |             6.469B              |                177.21M                 |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)      | 2560  |  640  |  10/512 |   1   |       4.915M        |               1.31M |        2560(shared Gate)         |             2.523B              |                 55.38M                 |

上表中 router 参数量标*的意味着还有 loss-free 的 bias 参数，这个参数的 shape 和 $N_r$ 一样，相比整体的router参数很小。Kimi-K3 系列因为在 routed path 上用了 Stable LatentMoE，所以 shared expert 和 routed expert 的参数量不一样，其他模型的 shared expert 与 routed expert 参数量都是一致的。Qwen 的两个模型在 shared expert 上还额外加了一个标量的 gate 来做控制，这个参数量和 $d$ 相同，也就是每层统一对 shared expert 做缩放。

:::table{fit vlines="1,3,7"}
| 模型                                                                                            |  $L$  | $L_{\text{MoE}}$ | $P_{\text{MoE}}^{\text{layer}}$ | $P_{\text{MoE}}$ | $P_{\text{model}}$ | $P_{\text{MoE}}/P_{\text{model}}$ | $P_{\text{MoE}}^{\text{layer,active}}$ | $P_{\text{MoE}}^{\text{active}}$ | $P_{\text{model}}^{\text{active}}$ | $P_{\text{MoE}}^{\text{active}}/P_{\text{model}}^{\text{active}}$ |
| ----------------------------------------------------------------------------------------------- | :---: | :--------------: | ------------------------------: | :--------------: | :----------------: | :-------------------------------: | :------------------------------------: | :------------------------------: | :--------------------------------: | :---------------------------------------------------------------: |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)                      |  93   |        92        |                         29.785B |     2740.2B      |      2779.9B       |              98.57%               |                718.41M                 |             66.094B              |              105.81B               |                              62.46%                               |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json)        |  92   |        92        |                         25.824B |     2375.8B      |      2419.8B       |              98.18%               |                557.85M                 |             51.322B              |              95.288B               |                              53.86%                               |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)     |  61   |        61        |                         25.436B |     1551.6B      |      1573.0B       |              98.64%               |                465.17M                 |             28.376B              |              49.781B               |                              57.00%                               |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)                         |  78   |        75        |                          9.703B |     727.72B      |      743.38B       |              97.89%               |                341.31M                 |             25.598B              |              41.251B               |                              62.06%                               |
| [Hy4-preview](https://huggingface.co/tencent/Hy4-preview/blob/main/config.json)                 |  78   |        77        |                          9.703B |     747.13B      |      769.91B       |              97.04%               |                341.31M                 |             26.281B              |              49.058B               |                              53.57%                               |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)             |  45   |        42        |                          7.274B |     305.51B      |      313.89B       |              97.33%               |                227.67M                 |             9.5622B              |              17.940B               |                              53.30%                               |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) |  43   |        43        |                          6.469B |     278.15B      |      284.33B       |              97.83%               |                177.21M                 |             7.6200B              |              13.802B               |                              55.21%                               |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)      |  48   |        48        |                          2.523B |     121.09B      |      126.19B       |              95.96%               |                55.380M                 |             2.6583B              |              7.7562B               |                              34.27%                               |

此处统计均不统计 MTP 参数量，而保留了普通的 embedding/LM head 以及视觉模型保留 vision encoder 相关参数。Qwen3.8-Flash-Next 中的 ngram embedding 也排除在外。

可以看出，在这些模型中，MoE 层的参数占了整体的绝大部分，均在 95% 以上，但在每 token 实际激活参数中，MoE 占比则没有那么夸张，最多的也只是不到 2/3. 这更说明了稀疏计算的实用性，扩大模型参数量的同时，控制每 token 实际计算量不会太夸张。

同样值得注意的是，由于最初的几层做 load-balancing 的效果往往不太好，一些模型会选择在前几层用 dense FFN，而 DeepSeek 虽然每一层都用了 MoE，但却在前几层使用了 hash routing 的固定路由方式。

[^bib]

[^ref]

[^giscus]
