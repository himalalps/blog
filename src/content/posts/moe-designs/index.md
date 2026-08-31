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

## 推理视角

## 前沿模型

这里尝试统计各前沿模型的 MoE 设置，此处单个 routed/shared expert 的 intermediate size 定为 $m_r$，可以把实际被激活的所有 expert 的 $m_r$ 加和当作原始的 $\dff$，那么就可以计算这里的 routed ratio 和 active ratio.

| 模型               |  $d$  | $m_r$ |  $k/N_r$ | $k m_r$ | $r_{\text{route}}$ | $N_s$ | $r_{\text{active}}$ |
| ------------------ | :---: | ----: | -------: | ------: | ------------------ | :---: | ------------------- |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)            | 7168  |  3072 | 16 / 896 |   49152 | 6.857*             |   2   | 7.714*              |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json)  | 8192  |  2048 | 10 / 512 |   20480 | 2.500              |   1   | 2.750               |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)    | 7168  |  3072 |  6 / 384 |   18432 | 2.571              |   1   | 3.000               |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)            | 6144  |  2048 |  8 / 256 |   16384 | 2.667              |   1   | 3.000               |
| [Hy4-preview](https://huggingface.co/tencent/Hy4-preview/blob/main/config.json)        | 6144  |  2048 |  8 / 256 |   16384 | 2.667              |   1   | 3.000               |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)      | 4096  |  2048 |  8 / 288 |   16384 | 4.000              |   1   | 4.500               |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)  | 4096  |  2048 |  6 / 256 |   12288 | 3.000              |   1   | 3.500               |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) | 2560  |   640 | 10 / 512 |    6400 | 2.500              |   1   | 2.750               |

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

比较让人意外的倒是 GLM-5.3-Flash，它相比 GLM-5.3 没有调整 expert 的 intermediate size，而是把 $d$ 调小到原来的 $2/3$，导致最后得到的 $r_{\text{route}}$ 以及 $r_{\text{active}}$ 都比原本大了 $3/2$. 笔者的理解是 GLM-5.3-Flash 因为采用了 mHC，所以自然可以把 hidden size 调小，降低其他每个部分的成本，而同时意味着把更多的容量分配给 FFN.

[^bib]

[^ref]
