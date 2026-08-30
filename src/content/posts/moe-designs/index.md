---
title: "模型架构：1. MoE"
description: "模型架构系列第一篇，围绕 MoE 的基本结构、SwiGLU 与细粒度专家设计，梳理共享专家、路由函数、负载均衡和专家并行等关键取舍。"
date: "2026-08-30"
lang: "zh-CN"
bibliography: "ref.bib"
---

# 模型架构：1. MoE

> 笔者在回顾不同模型的架构时，发现前沿的主流模型很多设计有不少共通之处。接下来一系列文章（希望不鸽），尝试从不同模块角度来分别浅析不同的设计选择，算是一个梳理，也欢迎读者指出任何错漏之处。这第一篇文章，就从 MoE 开始。

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

更现代的主流 MoE 设计里，相比原版会有一些改动：
1. 去除每一层的 bias，一方面 bias 内存比较密集但对于性能影响不大因而去除可以节省访存，另一方面 bias 可能引入计算稳定性问题。
2. 使用 SwiGLU 替代原始的 FFN，SwiGLU 里一方面引入了门控机制，另一方面换用了 SiLU 激活函数 $x\sigma(x)$.

最终得到的单个 MLP 的计算公式为：
$$
\begin{equation}
\bm y=[\operatorname{SiLU}(\bm x\bm W_g)\odot \bm x\bm W_u] \bm W_d,
\end{equation}
$$
其中 $\bm W_g,\bm W_u\in\R^{d\times \dff},\bm W_d\in\R^{\dff\times d}$.

而因为此处相比原始的 FFN 多了一个矩阵 $\bm W_g$，为了控制整体的 FLOPs 不变，一些模型会选择将 $\dff$ 缩小到原来的 $2/3$，也就是此时的 $\dff=8d/3$. 当然这样精确计算的结果未必是适合硬件的设计（切成 tile 有尾块），最终取的值可能是通过性能 benchmark 取得的更优点。

### LatentMoE



## 稀疏计算

介绍完单个 expert 的设计之后，就需要考虑如何将多个并行的 expert 组合起来。MoE 的核心是稀疏计算：每个 token 只激活一部分专家。这样可以在不增加每个 token 的计算量的情况下，增加模型的总参数量，从而提升模型的表达能力。

大体上，现有的 MoE 都可以看作以下的逻辑
$$
\begin{equation}
\bm y = \bm x + \sum_{i=1}^{N_s} E_i^{(s)}(\bm x) + \sum_{i=1}^{N_r} g_{i,t} E_i^{(r)}(\bm x),
\end{equation}
$$
其中 $N_s$ 是共享专家的数量，$N_r$ 是路由专家的数量，$E_i^{(s)}$ 和 $E_i^{(r)}$ 分别是共享专家和路由专家的计算函数，$g_{it}$ 是路由专家的激活权重。共享专家对所有 token 都是常开的，而路由专家则根据 token 的特征动态选择。有些模型没有共享专家，可以看作 $N_s=0$；而那些传统的 dense 模型则可以看作 $N_r=0$.

## 负载均衡


<!-- 传统路由是 softmax 后取 top-k，再用 softmax 概率做加权。gpt-oss 用了一个更简单的变体 [@gptoss]：**对每个专家的 logit 独立做 sigmoid，取 top-k 后按 sigmoid 值归一化**。

区别听起来微妙，实际影响不小：sigmoid 的分数天然有界、且不需要对所有专家做全局归一化，训练更稳，浅层的路由分布也更平滑。配合"路由用 FP32 计算"这类数值细节，是目前比较省心的方案。

MiniMax-M1 则走得更远：干脆不用共享的 router 矩阵，而是**每个专家自己带一个打分头**（attractive router），从结构上绕开全局 softmax 的耦合。

MoE 训练最大的麻烦是负载不均：router 塌缩到少数几个专家，剩下的专家学不到东西。经典解法是加辅助损失（auxiliary loss）惩罚不均衡，但它和主损失是"拉扯"关系，系数调不好会伤质量。

DeepSeek-V3 把这件事做成了**免辅助损失的均衡** [@deepseekv3]：

- 给每个专家的 router 分数加一个 bias 项，**bias 只参与 top-k 的选择，不参与最终的加权**，因此不会污染主损失；
- 每个 step 结束后看各专家的实际负载，过载的专家 bias 调小、闲置的调大——一个纯工程意义上的控制回路。

配合足够冗余容量，V3 直接做到了 **dropless**：不做 token dropping，每个 token 一定被目标专家处理，避免训练信号被丢弃。Qwen3、K2 等模型也都采用了这一套思路 [@qwen3; @kimik2]（部分仍保留小系数的序列级均衡 loss 作为补充）。 -->

## 训练工程

<!-- MoE 的并行比 dense 模型复杂得多，很多架构决定其实是通信决定的：

- **Node-limited routing**（DeepSeek-V3）：限制每个 token 的 8 个专家最多分布在 4 个节点上，直接控制 all-to-all 通信的扇出；
- **EP 与 DP/TP 的混合划分**：专家分散到多卡，注意力部分按传统 DP/TP 切，路由部分做 all-to-all；
- **Upcycling**：与其从零训练，不如从 dense 模型复制 FFN 初始化专家（Qwen3 即从 Qwen2.5 upcycle 而来 [@qwen3]），省下的算力拿去后训练；
- **低精度**：MoE 的显存压力大，DeepSeek-V3 的 FP8 训练、gpt-oss 的 MXFP4 推理，都是围绕"专家权重太多"这个问题。 -->

## 推理视角

<!-- 一个常见误解是"37B 激活 = 37B 显存"。实际上**所有专家的权重都要驻留显存**（或付出 offload 的延迟代价），MoE 是典型的"大显存、低算力"模型：

- **Prefill** 阶段批大，专家利用率高，接近理想收益；
- **Decode** 阶段批小、访存为主，MoE 的收益更多体现在高并发吞吐而非单请求延迟；
- 小尺寸 MoE（Qwen3-30B-A3B [@qwen3]、gpt-oss-20b [@gptoss]）本质是用总参数换激活效率：30B 的存储、3B 的计算，适合边缘与高并发部署。 -->

<!-- ## 一些观察

- **专家没有想象中"分工明确"**。对训练好的模型做路由分析，多数专家的路由熵很高，"每个专家掌握一种知识"的图景大体不成立；倒是推理链上专家覆盖率明显上升，有工作把这称为模型的"crowd thinking"。
- **路由冗余是特性不是 bug**。多个专家学到相近功能，换来的是对 router 噪声的鲁棒性，也解释了为什么逐个"剪专家"往往剪不动。
- **MoE 与 dense 的边界在移动**。同等训练算力下 MoE 的质量优势有多方证据支撑，于是各家旗舰全面 MoE 化；但 dense 在小尺寸、低显存场景仍有不可替代性。

MoE 的设计空间还远没有收敛：路由函数、专家粒度、均衡策略、与注意力稀疏化（如 DSA）的组合，每一项都还在快速演进。这一代模型的共识大概只有一句话——**专家要又多又小，均衡要靠控制回路而不是损失函数**。 -->

## 参考文献

[^ref]
