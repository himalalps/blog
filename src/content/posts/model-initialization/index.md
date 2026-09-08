---
title: "模型训练的初始化"
description: "从二阶矩传播推导 LeCun、Xavier 与 Kaiming 初始化，分析 Transformer 的残差缩放，并用特征变化与谱范数解释 μP 的初始化和学习率如何随模型宽度变化。"
date: "2026-09-08"
lang: "zh-CN"
author: "Haoyu Tang"
bibliography: "ref.bib"
---

# 模型训练的初始化：从 LeCun、Xavier、Kaiming 到 $\mu$P

讨论 Transformer 初始化时，首先要区分三个问题：网络刚初始化时，信号能否正常传播；网络较深时，残差与归一化能否维持合理的尺度；模型变宽后，同一套训练超参数是否仍然对应相近的学习行为。

LeCun、Xavier 和 Kaiming 初始化主要处理第一个问题。Transformer 的残差缩放和归一化可以处理第二个问题。$\mu$P（Maximal Update Parametrization）则把第三个问题纳入设计：它同时规定初始化、前向乘数和不同参数的学习率如何随宽度变化。

## 统一记号

本文采用列向量约定，
$$
\bm x\in\mathbb R^{d_{\mathrm{in}}},\qquad
\bm W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},\qquad
\bm y=\bm W\bm x\in\mathbb R^{d_{\mathrm{out}}},
$$
$\bm W$ 的行对应输出，列对应输入。如果要引入 bias 则写作 $\bm y=\bm W\bm x+\bm b$，其中 $\bm b\in\mathbb R^{d_{\mathrm{out}}}$. 如果需要同时处理 $T$ 个 token，就可以组合起来列向量排列 $\bm X\in\mathbb R^{d_{\mathrm{in}}\times T}$，于是 $\bm Y=\bm W\bm X$. 

默认权重初始化各维度相互独立，且
$$
\mathbb E[\bm W_{li}]=0,\qquad
\operatorname{Var}(\bm W_{li})=\sigma_W^2,
$$
一般有三种采样方式，高斯分布、均匀分布、截断正态分布。若使用高斯分布，则 $\bm W_{li}\sim\mathcal N(0,\sigma_W^2)$，好处是采样结果多样，缺点是采样结果无界，一些绝对值过大可能不利于后续优化；均匀分布可以写成 $U[-\sqrt3\sigma_W,\sqrt3\sigma_W]$，好处是有界，但采样结果也比较单一。截断正态分布则是在正态分布采样基础上，只保留落在截断区间内的结果，如果落在外部则重新采样直到落在指定区间内。注意因为做了截断，所以方差会出现变化，需要根据截断区间来调整正态分布取的方差，当采用 2 倍标准差截断时，对应的比值为
$$
\gamma=\frac{\int_{-2}^{2}e^{-x^2/2}x^2\mathrm{d} x}{\int_{-2}^{2}e^{-x^2/2}\mathrm{d} x}\approx 0.773,
$$
所以应该扩大标准差为 $\sigma_W/\sqrt{\gamma}\approx1.294\sigma_W.$

一个权重的输入，可以是原始数据，也可以是上一层的隐藏表示。下文先假设 $\bm x$ 与当前层的随机权重 $\bm W$ 独立；对于各层权重独立初始化、没有权重共享的前馈网络，上一层的输出自然也满足独立条件。

不妨先假设输入具有相同、有限且非零的二阶矩
$$
0<\mathbb E[\bm x_i^2]<\infty,
\qquad i=1,\ldots,d_{\mathrm{in}},
$$
这里**不要求输入零均值，也不要求各坐标相互独立或服从高斯分布**。如果各输入坐标的二阶矩不同，也可以用它们的平均值来衡量输入尺度
$$
\frac1{d_{\mathrm{in}}}\sum_{i=1}^{d_{\mathrm{in}}}\mathbb E[\bm x_i^2]
=\frac{\mathbb E\|\bm x\|_2^2}{d_{\mathrm{in}}}.
$$

二阶矩与方差的关系为
$$
\mathbb E[\bm x_i^2]
=\operatorname{Var}(\bm x_i)+\mathbb E[\bm x_i]^2.
$$
只有均值为零时，两者才相等。零均值、单位方差的输入对应 $\mathbb E[\bm x_i^2]=1$，这自然很好，但并不是本文推导所必需。

## 2. LeCun 初始化

先把矩阵乘法展开
$$
\bm y_l=(\bm W\bm x)_l
=\sum_{i=1}^{d_{\mathrm{in}}}\bm W_{li}\bm x_i,
$$
在初始化权重与输入独立的条件下，固定 $\bm x$，只对权重随机性取期望。零均值给出 $\mathbb E_{\bm W}[\bm y_l\mid\bm x]=0$；平方项则需要展开两次求和
$$
\begin{equation}
\label{eq:conditional-output-second-moment}
\begin{aligned}
\mathbb E_{\bm W}[\bm y_l^2\mid\bm x]
&=\mathbb E_{\bm W}\left[
\left(\sum_i\bm W_{li}\bm x_i\right)
\left(\sum_j\bm W_{lj}\bm x_j\right)\middle|\bm x\right]\\
&=\sum_{i,j}\bm x_i\bm x_j\,
\mathbb E_{\bm W}[\bm W_{li}\bm W_{lj}]\\
&=\sum_i\sigma_W^2\bm x_i^2
=\sigma_W^2\|\bm x\|_2^2.
\end{aligned}
\end{equation}
$$
第三个等号成立是因为当 $i\ne j$ 时，有 $\mathbb E[\bm W_{li}\bm W_{lj}]=0$，而当 $i=j$ 时，该期望为 $\sigma_W^2$。因此，这里的等式并不要求输入的不同坐标相互独立。接下来再对输入取期望，得到适用于各坐标二阶矩不同的情形的一般公式
$$
\mathbb E[\bm y_l^2]
=\mathbb E_{\bm x}\!\left[\mathbb E_{\bm W}[\bm y_l^2\mid\bm x]\right]
=\sigma_W^2\sum_{i=1}^{d_{\mathrm{in}}}\mathbb E[\bm x_i^2],
$$
不妨假设求和中的每一项都相等，因此对任一输入坐标 $i$，有
$$
\mathbb E[\bm y_l^2]
=d_{\mathrm{in}}\sigma_W^2\mathbb E[\bm x_i^2].
$$
若希望单坐标的二阶矩保持不变，即 $\mathbb E[\bm y_l^2]=\mathbb E[\bm x_i^2]$，由前面的单坐标公式得到
$$
d_{\mathrm{in}}\sigma_W^2=1
\quad\Longrightarrow\quad
\boxed{\operatorname{Var}(\bm W_{li})=\frac1{d_{\mathrm{in}}}}.
$$
这就是通常所说的 LeCun 初始化 [@lecun1998efficient-backprop]。可以理解为，每个输出项依赖于 $d_{\mathrm{in}}$ 个输入坐标，所以需要除以 $d_{\mathrm{in}}$ 来保持单坐标的二阶矩不变。

不过，保持单坐标尺度不等于保持整个向量的模长，有
$$
\begin{aligned}
\mathbb E\|\bm y\|_2^2
&=\sum_{l=1}^{d_{\mathrm{out}}}\mathbb E[\bm y_l^2]
=d_{\mathrm{out}}\mathbb E[\bm y_l^2]\\
&=d_{\mathrm{out}}d_{\mathrm{in}}\sigma_W^2\mathbb E[\bm x_i^2]\\
&=d_{\mathrm{out}}\sigma_W^2\mathbb E\|\bm x\|_2^2,
\end{aligned}
$$
此时代入 LeCun 初始化的 $\sigma_W^2=1/d_{\mathrm{in}}$，便有 $\mathbb E\|\bm y\|_2^2=\frac{d_{\mathrm{out}}}{d_{\mathrm{in}}}\mathbb E\|\bm x\|_2^2$，从 $d$ 维映射到 $4d$ 维时，如果每个坐标的二阶矩不变，平方模长的期望就会增加到四倍。

若目标改为保持向量模长的期望，即 $\mathbb E\|\bm y\|_2^2=\mathbb E\|\bm x\|_2^2$，则由一般的平方模长公式得到
$$
d_{\mathrm{out}}\sigma_W^2=1
\quad\Longrightarrow\quad
\boxed{\operatorname{Var}(\bm W_{li})=\frac1{d_{\mathrm{out}}}},
$$
这就是 LeCun 初始化的对偶，也是 [@kexuefm-7180] 里几何推导得到的结果。

## 3. Xavier 初始化

前面只考虑了前向传播，接下来看看反向传播的情况。设 $\mathcal L$ 为标量损失，记输出梯度为 $\bm g_y=\partial\mathcal L/\partial \bm y$、输入梯度为 $\bm g_x=\partial\mathcal L/\partial \bm x$，同样采用列向量约定，有
$$
\begin{aligned}
(\bm g_x)_i
&=\sum_{l=1}^{d_{\mathrm{out}}}
\frac{\partial\mathcal L}{\partial\bm y_l}
\frac{\partial\bm y_l}{\partial\bm x_i}
=\sum_{l=1}^{d_{\mathrm{out}}}(\bm g_y)_l\bm W_{li}\\
&=(\bm W^\top\bm g_y)_i,
\qquad \bm g_x=\bm W^\top\bm g_y.
\end{aligned}
$$
不妨假设输出梯度各坐标的二阶矩相同，并在初始化时忽略 $\bm W$ 与反传梯度的相关性。仿照前面的展开，消去交叉项，有
$$
\mathbb E[(\bm g_x)_i^2]\approx\sum_{l=1}^{d_{\mathrm{out}}}
\mathbb E[\bm W_{li}^2]\mathbb E[(\bm g_y)_l^2]
=d_{\mathrm{out}}\sigma_W^2\mathbb E[(\bm g_y)_l^2].
$$
注意反传梯度本身依赖于网络的前向计算，因此这里的独立性只是近似。

因此，保持前向单坐标二阶矩需要 $d_{\mathrm{in}}\sigma_W^2\approx1$，保持反向单坐标二阶矩需要 $d_{\mathrm{out}}\sigma_W^2\approx1$。当输入、输出宽度不相等时，一个标量方差一般无法同时满足两者。

一个折中的办法是让这两个乘子的算术平均等于 1，于是
$$
\frac{d_{\mathrm{in}}\sigma_W^2+d_{\mathrm{out}}\sigma_W^2}{2}=1
\quad\Longrightarrow\quad
\boxed{\operatorname{Var}(\bm W_{li})=\frac2{d_{\mathrm{in}}+d_{\mathrm{out}}}}.
$$
这就是 Xavier 初始化 [@glorot2010initialization]。当 $d_{\mathrm{in}}=d_{\mathrm{out}}$ 时，它与 LeCun 初始化相同；非方阵时，则只保证两个乘子的平均值为 1。

例如 $d_{\mathrm{out}}=4d_{\mathrm{in}}$，忽略非线性时得到：

| 初始化方差                                     | 前向乘子 | 反向乘子 |
| ---------------------------------------------- | :------------: | :------------: |
| LeCun：$1/d_{\mathrm{in}}$                     |      $1$       |      $4$       |
| Xavier：$2/(d_{\mathrm{in}}+d_{\mathrm{out}})$ |     $2/5$      |     $8/5$      |
| fan-out：$1/d_{\mathrm{out}}$                  |     $1/4$      |      $1$       |

可以看到，Xavier 在前向与反向的二阶矩之间作了折中，是否合适取决于要控制哪个量。

## 4. Kaiming 初始化

前面分析的是线性层，如果输出接入激活函数，就需要把非线性造成的尺度变化一起算进去。以 ReLU 为例，记 $\bm z=\bm W\bm x$、$\bm y=\operatorname{ReLU}(\bm z)$，若 $\bm z_l$ 的分布关于零对称，则
$$
\begin{aligned}
\mathbb E[\operatorname{ReLU}(\bm z_l)^2]
&=\mathbb E[\bm z_l^2\mathbf 1_{\{\bm z_l>0\}}]\\
&=\frac12\left(
\mathbb E[\bm z_l^2\mathbf 1_{\{\bm z_l>0\}}]
+\mathbb E[\bm z_l^2\mathbf 1_{\{\bm z_l<0\}}]\right)\\
&=\frac12\mathbb E[\bm z_l^2].
\end{aligned}
$$
这里 $\mathbf 1_{\{\cdot\}}$ 是指示函数，第二个等号成立是因为正、负半轴对二阶矩的贡献相同，而零点没有贡献。再代入前面的线性层公式，要求激活后的二阶矩满足 $\mathbb E[\bm y_l^2]=\mathbb E[\bm x_i^2]$，有
$$
\mathbb E[\bm y_l^2]
=\frac12\mathbb E[\bm z_l^2]
=\frac12d_{\mathrm{in}}\sigma_W^2\mathbb E[\bm x_i^2],
\qquad
\boxed{\operatorname{Var}(\bm W_{li})=\frac2{d_{\mathrm{in}}}}.
$$
这就是适用于 ReLU 的 Kaiming 初始化 [@he2015rectifiers]。这里采用 fan-in 来保持前向二阶矩，相应的 fan-out 版本为 $2/d_{\mathrm{out}}$，用于控制反向信号的尺度。

注意 ReLU 输出一般不再是零均值，因此二阶矩与方差并不相等。例如，设单个预激活坐标 $\bm z_l\sim\mathcal N(0,\sigma_z^2)$，其中 $\sigma_z^2=\mathbb E[\bm z_l^2]>0$ 是其方差，使用积分变量 $t$ 可得
$$
\begin{aligned}
\mathbb E[\operatorname{ReLU}(\bm z_l)]
&=\int_0^\infty\frac{t}{\sqrt{2\pi \sigma_z^2}}
\exp\!\left(-\frac{t^2}{2\sigma_z^2}\right)\,\mathrm dt\\
&=\frac{\sigma_z^2}{\sqrt{2\pi \sigma_z^2}}
=\sqrt{\frac{\sigma_z^2}{2\pi}},\\
\operatorname{Var}(\operatorname{ReLU}(\bm z_l))
&=\mathbb E[\operatorname{ReLU}(\bm z_l)^2]
-\mathbb E[\operatorname{ReLU}(\bm z_l)]^2\\
&=\frac{\sigma_z^2}{2}-\frac{\sigma_z^2}{2\pi}.
\end{aligned}
$$

对于 GELU 和 SiLU，激活前后的二阶矩比例还会随输入尺度变化，因而不能直接使用 ReLU 的固定系数。记一般激活函数为 $\phi$，令 $\bm y=\phi(\bm W\bm x)$，仍假设各输入坐标的二阶矩相同。用高斯变量近似线性层的预激活，并令 $Z\sim\mathcal N(0,1)$，一般需要研究
$$
\mathbb E[\bm y_l^2]
\approx\mathbb E\left[\phi\!\left(\sqrt{d_{\mathrm{in}}\sigma_W^2\mathbb E[\bm x_i^2]}\,Z\right)^2\right].
$$
因此，Transformer 中不同矩阵需要分别考虑。Q、K、V 投影后面没有 ReLU，SwiGLU 又包含两个分支的逐元素乘法，需要计算联合二阶矩，不能统一给标准差乘上 $\sqrt2$。

以 SwiGLU 为例，设输入、输出宽度为 $d$，中间宽度为 $d_{\mathrm{ff}}$，两个分支的权重 $\bm W_{\mathrm{gate}},\bm W_{\mathrm{up}}\in\mathbb R^{d_{\mathrm{ff}}\times d}$，输出投影 $\bm W_{\mathrm{down}}\in\mathbb R^{d\times d_{\mathrm{ff}}}$，列向量形式为
$$
\operatorname{FFN}(\bm x)
=\bm W_{\mathrm{down}}
\left[\operatorname{SiLU}(\bm W_{\mathrm{gate}}\bm x)\odot(\bm W_{\mathrm{up}}\bm x)\right].
$$
令 $\bm g=\bm W_{\mathrm{gate}}\bm x$、$\bm u=\bm W_{\mathrm{up}}\bm x$，中间乘积为 $\bm v=\operatorname{SiLU}(\bm g)\odot\bm u$，其中 $\odot$ 表示逐元素乘法。对中间坐标 $k=1,\ldots,d_{\mathrm{ff}}$，有
$$
\mathbb E[\bm v_k^2]
=\mathbb E[\operatorname{SiLU}(\bm g_k)^2\bm u_k^2]
\approx\mathbb E[\operatorname{SiLU}(\bm g_k)^2]\,
\mathbb E[\bm u_k^2].
$$
这里的近似等号需要两个分支的输出近似独立。即使 gate 和 up 权重独立初始化，也只保证固定输入时的条件独立；对随机输入取平均后，共享的输入模长仍可能带来相关性。因此，SwiGLU 的 gain 需要结合两个分支计算。

这三种初始化可以整理如下：

| 方法                 | 常见权重方差                           | 主要控制对象               |
| -------------------- | -------------------------------------- | -------------------------- |
| LeCun                | $1/d_{\mathrm{in}}$                    | 线性前向的单坐标二阶矩     |
| Xavier               | $2/(d_{\mathrm{in}}+d_{\mathrm{out}})$ | 前向与反向逐坐标尺度的折中 |
| Kaiming，ReLU fan-in | $2/d_{\mathrm{in}}$                    | 加入 ReLU 后的前向二阶矩   |

在隐藏层输入、输出宽度按固定比例共同增长时，这三者的方差都与 $1/d_{\mathrm{in}}$ 同阶，只在常数因子上有区别。

## 5. 残差与归一化

前面控制了单层的二阶矩，但在 Transformer 中，各层输出还要通过残差相加，因此尺度会沿深度累积。

设模型宽度为 $d$，网络有 $L$ 个 Transformer block，每个 block 包含 attention 和 FFN 两个子层，总子层数为 $B=2L$。用 $s=0,\ldots,B-1$ 表示子层编号，$F_s$ 表示对应的 attention 或 FFN 运算，$\alpha_L$ 是其输出在残差相加前的缩放系数。Pre-Norm 子层可以写成
$$
\bm x_{s+1}=\bm x_s+\alpha_L F_s(\operatorname{Norm}(\bm x_s)).
$$
记 $\bm r_s=F_s(\operatorname{Norm}(\bm x_s))$。对残差流的任一坐标 $i=1,\ldots,d$，有
$$
\begin{aligned}
\mathbb E[\bm x_{s+1,i}^2]
&=\mathbb E[(\bm x_{s,i}+\alpha_L\bm r_{s,i})^2]\\
&=\mathbb E[\bm x_{s,i}^2]
+2\alpha_L\mathbb E[\bm x_{s,i}\bm r_{s,i}]
+\alpha_L^2\mathbb E[\bm r_{s,i}^2],
\end{aligned}
$$
若初始化时交叉项可以忽略，再沿深度对上式求和，相邻子层的二阶矩相消，有
$$
\begin{equation}
\label{eq:residual-accumulation}
\begin{aligned}
\mathbb E[\bm x_{B,i}^2]-\mathbb E[\bm x_{0,i}^2]
&=\sum_{s=0}^{B-1}\left(\mathbb E[\bm x_{s+1,i}^2]-\mathbb E[\bm x_{s,i}^2]\right)\\
&\approx\alpha_L^2\sum_{s=0}^{B-1}\mathbb E[\bm r_{s,i}^2].
\end{aligned}
\end{equation}
$$
若 $\mathbb{E}[\bm r_{s,i}^2]$ 都为常数量级，右侧便与 $B\alpha_L^2$ 同阶。要让总增量保持常数阶，需要让 $\alpha_L$ 按 $B^{-1/2}$ 的阶数缩放，这其实也就是 $L^{-1/2}$.

例如 nanoGPT [@karpathy-nanogpt]中，普通线性层和 embedding 用标准差 0.02 的正态初始化，attention 和 FFN 输出投影的 `c_proj.weight` 则用 $0.02/\sqrt{2L}$，这也符合上述推导。而这里的 0.02 则是沿用了 GPT-2 官方实现中的初始化标准差 [@openai-gpt2]。

Pre-Norm 归一化的是残差分支的输入，并不直接约束相加后的残差流 $\bm x_{s+1}$，从式 $\eqref{eq:residual-accumulation}$ 也可看出随着深度增加，对应残差流的二阶矩也会随之增加。

## 6. 参数更新与特征变化

前一个部分考虑的是深度增加时的残差累积。接下来固定深度，看看模型变宽后参数更新会怎样变化。

如果一组不同宽度的稠密输入满足 $\frac1{d_{\mathrm{in}}}\sum_i\mathbb E[\bm x_i^2]=\Theta(1)$，即平均坐标二阶矩不随宽度增长而发散或趋于零，就有
$$
\mathbb E\|\bm x\|_2^2
=\sum_{i=1}^{d_{\mathrm{in}}}\mathbb E[\bm x_i^2]
=\Theta(d_{\mathrm{in}}).
$$
但期望的尺度不能直接推广到典型样本，因为可能存在
$$
Z=\begin{cases}
0,&\text{概率为 }0.99,\\
10,&\text{概率为 }0.01.
\end{cases}
$$
这样的分布，此时每个坐标的二阶矩都是 1，但大量样本平方模长接近0，且维度增加也无法让样本的平方模长集中到期望附近。当然这些也并非下文推导所必需。

对一个样本的线性层 $\bm y=\bm W\bm x$，记 $\bm g=\partial\mathcal L/\partial \bm y$，则
$$
\begin{aligned}
\bm G_{li}
&=\frac{\partial\mathcal L}{\partial\bm W_{li}}
=\frac{\partial\mathcal L}{\partial\bm y_l}
\frac{\partial\bm y_l}{\partial\bm W_{li}}
=\bm g_l\bm x_i
=(\bm g\bm x^\top)_{li},\\
\bm G&=\bm g\bm x^\top
\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}.
\end{aligned}
$$
设学习率为 $\eta>0$。如果用 SGD 更新 $\Delta \bm W=-\eta \bm g\bm x^\top$，先固定该层输入 $\bm x$，本层参数更新对特征的直接影响是
$$
\begin{equation}
\label{eq:direct-feature-update}
\boxed{\Delta \bm y_{\mathrm{direct}}
=\Delta \bm W \bm x
=-\eta(\bm g\bm x^\top)\bm x
=-\eta\bm g(\bm x^\top\bm x)
=-\eta \bm g\,\|\bm x\|_2^2}.
\end{equation}
$$
由式 $\eqref{eq:conditional-output-second-moment}$，初始化时输出单坐标的 RMS 为 $\sigma_W\|\bm x\|_2=\Theta(\sigma_W\sqrt{d_{\mathrm{in}}})$。而由式 $\eqref{eq:direct-feature-update}$，更新量中第 $l$ 个输出坐标的每一项为 $-\eta\bm g_l\bm x_i^2$，其中 $i$ 遍历输入坐标。对固定样本和输出坐标 $l$，这些项具有相同的因子 $-\eta\bm g_l$，且 $\bm x_i^2\ge0$，因此不会相互抵消，而是按 $\sum_i\bm x_i^2=\Theta(d_{\mathrm{in}})$ 累积，可以看到，相比之前已经差出平方量级。

同时，将梯度外积的平方范数展开，有
$$
\|\bm G\|_F^2
=\sum_{l,i}(\bm g_l\bm x_i)^2
=\left(\sum_l\bm g_l^2\right)\left(\sum_i\bm x_i^2\right)
=\|\bm g\|_2^2\|\bm x\|_2^2.
$$
单层更新对损失的一阶贡献是
$$
\begin{equation}
\label{eq:direct-loss-update}
\Delta\mathcal L_W
\approx\langle \bm G,\Delta \bm W\rangle_F
=-\eta\|\bm G\|_F^2
=-\eta\|\bm g\|_2^2\|\bm x\|_2^2.
\end{equation}
$$
式 $\eqref{eq:direct-feature-update}$ 和 $\eqref{eq:direct-loss-update}$ 分别描述特征变化与损失变化。即使总损失变化保持常数阶，部分隐藏层的特征变化仍可能随宽度趋于零。$\mu$P 希望各层的特征变化都稳定且不消失，在宽度缩放允许的阶数下取最大更新，并不是把数值学习率取得越大越好 [@yang2021feature-learning]。

注意本节推导均固定了本层输入，只计算本层参数更新的直接贡献。完整网络还包含上游特征变化，一阶近似为 $\Delta \bm y\approx\Delta \bm W \bm x+\bm W\Delta \bm x$，因此单层结果还不能直接描述完整的训练过程。

[^bib]

[^ref]

[^giscus]
