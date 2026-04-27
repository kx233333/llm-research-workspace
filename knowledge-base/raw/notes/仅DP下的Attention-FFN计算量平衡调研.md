---
title: "仅 DP 下的 Attention vs FFN 计算量平衡：估算方法、解耦方案与显存分析"
date: 2026-04-22
tags:
  - LLM训练
  - 数据并行
  - DP
  - Attention
  - FFN
  - 负载均衡
  - 长上下文
  - 显存分析
  - 变长序列
  - 解耦
status: 整理中
---

# 仅 DP 下的 Attention vs FFN 计算量平衡

> **问题聚焦**：假设只使用数据并行（DP），放弃 TP/PP/CP，在变长序列训练/推理中 Attention（$O(s^2 h)$）和 FFN（$O(s h^2)$）计算量不对称时，各方案如何估算负载？解耦二者能否更好平衡？显存分布差异有多大？

> 与之前笔记 [[Attention-FFN计算量不平衡与解耦调研]] 的区别：本笔记**只考虑 DP** 这一最简并行模式，不讨论 TP/PP/CP 带来的复杂性，更聚焦"跨 rank 分配数据"本身的问题。

---

## 一、DP 场景下的不平衡本质

### 1.1 DP 的同步模型

```
DP rank 0: [micro-batch 0] ─ forward ─ backward ─┐
DP rank 1: [micro-batch 1] ─ forward ─ backward ─┼─ all-reduce gradient ─ update
...                                               │
DP rank N: [micro-batch N] ─ forward ─ backward ─┘
                                                  ↑
                                       所有 rank 必须到达这一屏障
```

**硬性约束**：
- 所有 DP rank 每一步必须处理**相同数量的 micro-batch**（梯度累积要求）
- 梯度聚合是强同步点——最慢 rank 决定步长
- 任何计算量不均衡直接转化为 **DP Bubble**（等待开销）

### 1.2 单层每 token FLOPs 分解（Forward）

设 batch size $B$、序列长度 $s$、hidden dim $h$，则每个打包序列（single packed sequence）的每层 FLOPs：

$$
\text{FLOPs}_{\text{layer}}(s) = \underbrace{\alpha s^2}_{\text{Core Attention}} + \underbrace{\beta s}_{\text{其余线性层}}
$$

其中（粗略常数）：
- $\alpha \sim 4h$（核心注意力 $\text{softmax}(QK^T)V$）
- $\beta \sim 12h^2$（QKV projection + Output projection + FFN 全部）

**两项相等时的临界长度**：

$$
\alpha s^2 = \beta s \implies s^* = \frac{\beta}{\alpha} \approx \frac{12h^2}{4h} = 3h
$$

- $h = 4096$ → $s^* \approx 12K$
- $h = 8192$（34B 级别）→ $s^* \approx 24K$

**关键观察**（来自 [CAD/DistCA 论文](https://arxiv.org/abs/2510.18121)）：

> "Core attention is compute-heavy but stateless; the rest is linear in tokens for both compute and memory."

- Attention 的 FLOPs ∝ $s^2$（quadratic）
- 其余层（QKV projection / FFN / layernorm 等）FLOPs ∝ $s$（linear）
- **Activation memory** 主要由"其余层"贡献，∝ $s$（现代 FlashAttention 不存 $P = \text{softmax}(QK^T)$，仅保留 LSE）

---

## 二、DP 负载的本质矛盾：两个守恒量无法同时平衡

### 2.1 CAD 论文的形式化

对于按文档打包的 micro-batch，设 batch 中文档长度列表为 $\{l_i\}_{i=1}^n$：

| 量 | 表达式 | 物理含义 |
|---|-------|---------|
| 计算 FLOPs | $\alpha \sum l_i^2 + \beta \sum l_i$ | 总计算量 |
| 激活显存 | $\gamma \sum l_i$ | 总显存（FlashAttn 下）|

**要让两个 DP rank 计算和显存都平衡，必须同时满足**：

$$
\sum_i l_i = \sum_j l_j'  \quad \text{AND} \quad \sum_i l_i^2 = \sum_j l_j'^2
$$

这是一个 **bin-packing 双目标问题**，在实际变长序列分布下**几乎不可解**——特别是当 batch 中存在极长序列时（outlier）。

### 2.2 为什么这是根本矛盾

考虑极端例子：一个 512K 的极长序列 vs 128 个 4K 短序列：

| 指标 | 长序列 rank | 短序列 rank |
|-----|-----------|-----------|
| $\sum l$（token 数）| 512K | 512K |
| $\sum l^2$（~FLOPs）| $512\text{K}^2 = 2.6 \times 10^{11}$ | $128 \times (4\text{K})^2 = 2.1 \times 10^9$ |
| Attention FLOPs 比 | **~120×** 更多 | 基准 |
| 激活显存 | 相同 | 相同 |

→ token 数相同，但 **Attention FLOPs 差 120 倍**，长序列 rank 必然成为 straggler。

---

## 三、DP 下的四类估算+平衡方案

下表是 DP 视角下主流系统的对比，完全省略 TP/PP/CP 相关机制：

```
方案               估算什么     平衡什么           解耦Attn/FFN    显存处理
──────────────────────────────────────────────────────────────────────────
Naive DP           $\sum l$     token 数          ❌              均等（但计算不均）
WLB-LLM            $W_a(s) + W_l(s)$  总延迟     ❌              Var-Len 打包
ByteScale          FLOPs        预测时间          ❌              加 DP-Balance 策略
LobRA              Memory + Time 异构副本         ❌              按桶分发
CAD / DistCA       $\alpha l^2$ vs $\beta l$  分开平衡  ✅ 核心注意力独立  按 token 线性
```

---

### 3.1 Naïve DP：均分 token 数

**估算方法**：假设每 micro-batch token 数相同即可平衡。

**做法**：
- 将 global batch 按顺序切成 $N$ 份（固定 token 数 / 固定样本数）
- 不考虑序列长度分布

**DP Bubble 来源**：
1. 同等 token 数下，长文档 rank 的 Attention FLOPs 大几十倍（$\sum l_i^2$ 悬殊）
2. 数据 skewness：ByteScale 对 Byted 数据集的观察——0.05% 的样本贡献了 12.1% 的 token，1% 的样本贡献了 44.3% 的 token

**实测**（ByteScale 数据）：
- 32K 上下文训练，micro-batch 间 FLOPs 差异可达 **3–10×**
- 单纯 DP 下 bubble 占比可超过 30%

**适用条件**：只适用于 $s \ll 3h$（短序列均匀分布场景）。

---

### 3.2 WLB-LLM（arXiv:2503.17924）：双项 profiling 估算

> TLDR：变长文档打包 + per-document sharding，平均 1.23× 加速

**核心洞察**：计算延迟不完全由 $s^2$ 决定。对 $s \lesssim 12K$ 场景，线性项（GEMM + 通信 + element-wise）仍占主要时间。

**估算模型**（来自论文 §4.1）：

$$
W(s) = \underbrace{W_a(s)}_{\text{attention 项，} \propto s^2} + \underbrace{W_l(s)}_{\text{线性项，} \propto s}
$$

其中 $W_a, W_l$ 都是**通过离线 profiling 拟合**的函数（不是纯理论公式）。

**平衡目标**（变长打包优化问题）：

$$
\min_{x_{ij}} \max_{j=1..M} \sum_{i=1}^{N} x_{ij}(W_a(d_i) + W_l(d_i))
$$

**Outlier Document Delay**：
- 多级等待队列保留极长文档
- 队列满（= micro-batch 数）时才释放
- 保证每个 micro-batch 含恰好一个长文档 → 各 micro-batch 的 $W_a$ 主导项接近

**结论**：不需要解耦 Attn/FFN，只要精确估算二者总 latency 并统一优化即可。**前提**：序列长度分布不是极端长尾（outlier 数量有限）。

---

### 3.3 ByteScale（arXiv:2502.21231）：DP-Balance 调度

> TLDR：12000+ GPU 上 2048K 上下文训练，DP/PP 统一调度，7.89× 加速

**估算方法**：
- 对每个序列 $s_i$ 计算 $\text{FLOPs}(s_i) = \alpha s_i^2 + \beta s_i$
- **允许不同 DP rank 处理不同数量的 micro-batch**（打破梯度累积的"每 rank 等量"假设）
- 数学等价性保持：全局 batch 的梯度总和不变

**DP-Balance 策略核心洞察**（Insight 2）：

> It is only necessary to maintain load balance at each time step when pipeline parallelism is not applied.

对纯 DP 场景：
1. 将序列按长度降序排序
2. 按 FLOPs 分桶（每桶总 FLOPs 相等，故长序列桶含更少样本）
3. **同一时刻各 rank 从同一桶抽取序列**（保证同时刻 FLOPs 相似）
4. 处理短序列的 rank 可分配**更多** micro-batch，使总执行时间对齐

**关键差异于 WLB-LLM**：ByteScale 放弃"每 rank micro-batch 数相等"的硬约束，通过"慢 rank 少做几个 + 快 rank 多做几个"实现平衡。

**估算精度**：基于 profiling + 线性回归（类似 MegaScale-Infer），未尝试闭合解析解。

---

### 3.4 LobRA（arXiv:2509.01193）：异构 FT 副本 + 按长度桶分发

> TLDR：多租户 LoRA fine-tuning，减少 GPU-seconds 45–60%

**估算方法**：建立基于 bucket 的运行时模型 $T(\text{sequences}; \mathcal{S}_i)$，其中 $\mathcal{S}_i$ 是 replica 的并行配置。

**核心思想**：
- 放弃"同构 DP"假设，改用**异构副本**
- 每个副本用不同的并行配置（如 TP=1/2/4/8）
- 长序列 → 高 TP 副本（更多 GPU，避免 OOM）
- 短序列 → 低 TP 副本（更少 GPU，效率更高）
- 全局 DP 其实由多个异构 replica 群组成

**优化问题**（LobRA Eq.1）：

$$
\min_{p_i, d_{i,j}} \max_{i \in [1,S]} T\left(\left\{\lceil d_{i,j}/p_i \rceil\right\}_{j=1}^{r_i}; \mathcal{S}_i\right)
$$

约束：
- 所有序列必被处理
- 未选中的配置不分配数据
- GPU 总数不超

**对 Attn/FFN 分离的态度**：❌ 不分离二者，但通过**配置层面的异构**隐性缓解——长序列 replica 用高 TP 天然减小每 rank 的 Attn 计算压力。

---

### 3.5 CAD / DistCA（arXiv:2510.18121）：核心注意力显式解耦

> TLDR：核心注意力搬到独立 GPU 池，512K 上下文 1.35× 端到端加速，**近乎完美的计算和显存平衡**

**这是 DP 场景下最彻底的解耦方案**。

#### 3.5.1 洞察：核心注意力的两个关键性质

| 性质 | 含义 | 启用什么 |
|------|-----|---------|
| **Stateless** | CA（$\text{softmax}(QK^T)V$）无可训练参数，只有瞬时数据 | 可随意搬到任何 GPU，无权重同步 |
| **Composable** | 给定 $Q$ 和对应的 $KV$，token-level 可任意切分/合并 | 不同文档的 shard 可融合到一个 kernel |

**FlashAttention 的实证特性**（CAD 论文 Figure 5）：
- kernel MFU 与**融合 chunk 内总 token 数**相关，与源文档无关
- shard 长度 ≥ 128（kernel tile size）时 MFU 接近满
- 这意味着可以把来自不同文档的 shards 打包进一次 kernel call

#### 3.5.2 显式解耦的架构

```
普通 DP：
  rank 0: [Embedding][Attn][FFN][Attn][FFN]...[Output]  ← 全流程一卡
  rank 1: [Embedding][Attn][FFN][Attn][FFN]...[Output]
  rank 2: ...
  rank 3: ...
  ↓
  all-reduce gradient

CAD/DistCA（in-place attention server）：
  rank 0: [ctx-indep layers]  ──all-to-all──→ ├ CA Task A (任意 shard 组合)
  rank 1: [ctx-indep layers]  ──            ──┤ 
  rank 2: [ctx-indep layers]  ──            ──┤ CA Task B
  rank 3: [ctx-indep layers]  ──            ──┴ CA Task C
          ↑                                    ↓
          └───── 每个 rank 同时充当两个角色 ─────┘
                （时分复用，而非专门的 attention pool）
```

**关键设计**：**"in-place" attention server**——不把 CA 放到专门的 GPU 池，而是让每个 GPU 周期性切换角色。原因（论文 §4.1）：

> FFN layers account for the majority of memory consumption due to their large hidden states; conversely, the core attention is stateless. Dedicating a separate group of GPUs to core attention would leave their memory largely unused.

#### 3.5.3 调度算法（Communication-Aware Greedy Scheduling）

目标：

$$
\begin{aligned}
&\text{minimize} \quad \text{LoadImbalance}(\text{FLOPs}) \\
&\text{s.t.} \quad \text{CommVolume}(\text{bytes}) \text{ 最小}
\end{aligned}
$$

算法：
1. **Target load**：$\bar{F} = \frac{\sum \text{FLOPs}}{n}$（n = attention server 数）
2. **Profiler**：离线跑 CA kernel 在 $(Q, KV)$ 长度网格上的吞吐，运行时用 bilinear 插值预测
3. **Greedy migration**：从 surplus server 向 deficit server 迁移 shard，选 priority score 最高的：

$$
E = \frac{\Delta F_{\max}}{V_{\text{comm}}}
$$

其中 $\Delta F_{\max} = \min(F_{\text{Item}}, S_{\text{source}}, D_{\text{dest}})$。

4. **终止条件**：所有 server 负载进入 $\bar{F}(1 \pm \epsilon)$ 容差内

**关键贡献**：把原本"文档级" bin-packing 问题变成 **token-shard 级** bin-packing，搜索空间大大扩大，几乎必有解。

#### 3.5.4 实验结果

| 配置 | 加速比 vs WLB-LLM（DP-only, 即 3D w/o PP）|
|------|----------------------------|
| Llama-8B, Pretrain, 512K | 1.07–1.20× |
| Llama-8B, ProLong, 512K | 1.05–1.12× |
| Llama-34B, MaxDocLen↑ | 加速比也随之↑ |

**根本原因**：WLB-LLM 在 DP 规模扩大时 memory divergence 越来越高（1.08–1.17× memory overhead），CAD 则保持近乎完美平衡。

---

## 四、解耦 Attn/FFN 在纯 DP 下是否更合适？

### 4.1 理论优势

| 比较点                  | 耦合方案（WLB-LLM/ByteScale）        | 解耦方案（CAD/DistCA）                  |
| -------------------- | ------------------------------ | --------------------------------- |
| **估算粒度**             | 文档级（$\sum l_i^2$ + $\sum l_i$） | token-shard 级                     |
| **搜索空间**             | 文档分配给哪个 rank（离散）               | shard 分配给哪个 attention server（连续）  |
| **双目标平衡**            | 计算和显存必须耦合，有时无解                 | 天然解耦：计算在 server 侧平衡，显存在原 rank 侧平衡 |
| **对长文档 outlier 敏感度** | 高（一个 2M 文档可能打爆某 rank）          | 低（可切成 ≥128 token 的任意 shard）       |
| **kernel 效率损失**      | 无（保持整文档）                       | shard 长度 ≥128 即无损                 |

### 4.2 实践代价

| 代价 | 量级 |
|------|------|
| 额外通信 | $O(\sum l)$ 字节（$Q, KV$ shard 传输），CAD 实测可被 ping-pong overlap 完全隐藏 |
| 工程复杂度 | 高：需要调度器、profiler、fused CA kernel、all-to-all 通信库 |
| 调度开销 | CPU 上运行 greedy 算法，CAD 报告可被下一个 batch 的 GPU 计算 overlap |
| 内存碎片 | CAD 论文明确提到：Llama-34B 4D 实验有 memory fragmentation，建议后续用静态分配 + CUDA Graphs 修复 |

### 4.3 结论：何时值得解耦？

**强烈建议解耦的场景**：
- **MaxDocLen ≥ 3h**（如 12K for Llama-8B，24K for Llama-34B），此时 $s^2$ 项占主导
- 序列长度分布高度长尾（存在少量 >128K 的极长序列）
- DP 规模 ≥ 8（bubble 放大效应显著）

**不建议解耦的场景**：
- 短序列均匀分布（$s < h$，FLOPs 差异小于 2×）
- 极小规模训练（单机 8 卡），调度 overhead 可能超过解耦收益
- 对内存预算敏感（CAD 需要额外通信 buffer）

**DP 规模判断**（基于 CAD Figure 4b）：
- DP=1: imbalance idle = 0%（无意义）
- DP=4: imbalance idle = **19%**（考虑解耦）
- DP=8: imbalance idle = **55%**（必须解耦）

---

## 五、显存分布在 DP 下的具体差异

### 5.1 每 rank 显存分解（训练，bf16，单 GPU 视角）

设 DP rank 上持有一个 micro-batch 包含文档长度 $\{l_i\}_{i=1}^n$，总 token 数 $L = \sum l_i$：

```
权重（所有 rank 相同）：
  全部参数（不分片）  |█████████████████| 固定（例：34B → 68GB bf16）
  
优化器状态：
  Adam m+v+fp32 master |███████████████████████| 约 3× 权重

梯度：
  bf16 梯度           |██████████| 同权重

激活（随序列变化）：
  ── 其余线性层激活  |████████████| ≈ n_layer × 2 × L × h × b_bytes
                    （FlashAttn 下，Attention 不存 P 矩阵，只存 LSE）
  ── LSE（Attn）    |▌| n_layer × 2 × L × n_heads × 4 bytes（极小）
  
KV 通信 buffer（仅 CP 用，CAD 下为解耦 buffer）：
  Q, K, V shard     |█|（为被迁移的 CA-task 保留）
```

### 5.2 激活显存随 micro-batch 长度总数线性变化

以 Llama-34B（$h=8192$, $n_{\text{layer}}=48$）为例，激活显存计算（FlashAttn）：

$$
M_{\text{act}} \approx n_{\text{layer}} \times (34 L h) \text{ bytes（bf16 saved activations）}
$$

| micro-batch 总 token 数 $L$ | 激活显存 | 与权重（68GB）比 |
|--------|---------|-----------------|
| 32K | $\sim 435$ GB × (1/activation_ratio) ≈ **~15 GB** | 22% |
| 128K | ~60 GB | 88% |
| 512K | ~240 GB（**OOM**！）| 350% |

→ 这就是为什么长上下文训练必须用 CP/TP 或 activation recomputation。

### 5.3 解耦前后的显存再平衡

#### 5.3.1 耦合（Naïve DP）

| rank | micro-batch tokens | 激活显存 | Attention FLOPs |
|-----|------------------|---------|----------------|
| 0 | 32K（1 个 32K 文档）| 15 GB | $1024 \times 10^6$ |
| 1 | 32K（8 个 4K 文档）| 15 GB | $128 \times 10^6$ |
| **显存差异** | — | **1.00×**（平衡）| **8.0×**（严重不平衡）|

#### 5.3.2 解耦（CAD/DistCA）

| rank | ctx-indep tokens | CA server load | 激活显存 | Total FLOPs |
|-----|-----------------|--------------|---------|------------|
| 0 | 32K | 部分接受其他 rank 的 CA-task | 15 GB + CA buffer | 平衡 |
| 1 | 32K | 部分接受其他 rank 的 CA-task | 15 GB + CA buffer | 平衡 |
| **显存差异** | — | — | **~1.00×**（平衡）| **~1.00×**（平衡）|

→ **显存依然按 token 数线性分配，保持平衡**；计算分得更均匀（token-shard 级 bin-packing）。

### 5.4 显存差异量化对比

ByteScale 和 CAD 的实测数据：

| 方案 | 变长场景 memory overhead | 说明 |
|------|-----------------------|------|
| Fixed-size packing | 0%（基线）| 计算不平衡 |
| Variable-length chunking（WLB-like）| **+8–17%**（CAD Fig 4a）| memory 向某些 rank 倾斜 |
| Per-document CP | 需额外 KV memory，最后 rank 最多 | 随 CP 度数扩大 |
| **CAD/DistCA** | **~0%**（near-perfect balance）| 计算和显存**同时**平衡 |

---

## 六、关键估算公式速查

### 6.1 单序列（忽略非 FLOPs 开销）

$$
\text{FLOPs}(s) = \underbrace{4 \cdot s^2 \cdot h}_{\text{Core Attn}} + \underbrace{12 \cdot s \cdot h^2}_{\text{QKV Proj + Out Proj + FFN + ...}}
$$

### 6.2 Batch 内多文档（DP rank 级）

$$
\text{FLOPs}_{\text{rank}} = 4 h \sum_{i=1}^{n} l_i^2 + 12 h^2 \sum_{i=1}^n l_i
$$

$$
\text{ActMem}_{\text{rank}} \approx \gamma \sum_i l_i  \quad (\gamma \sim n_{\text{layer}} \cdot 34 \cdot h \text{ bytes under FlashAttn})
$$

### 6.3 CAD 解耦后

**CA server 侧**（无显存压力）：
$$
\text{FLOPs}_{\text{CA-server}} = 4h \sum_{t \in T_s} |q(t)| \cdot |kv(t)|
$$

**Rank 侧**（只跑 context-indep layers）：
$$
\text{FLOPs}_{\text{rank}}^{\text{linear}} = 12 h^2 \sum_i l_i, \quad \text{ActMem}_{\text{rank}} \approx \gamma \sum_i l_i
$$

→ 计算和显存都变成**纯 token 数的线性函数**，可以完美平衡。

---

## 七、小结

### 1. DP 的核心矛盾在于"token 数"和"FLOPs"是两个守恒量

FlashAttention 让激活显存 ∝ $\sum l_i$，但 Attention FLOPs ∝ $\sum l_i^2$。当序列长度差异大时，这两个量**不可能同时在各 rank 间平衡**。

### 2. 不解耦方案走"统一估算 + 调度"路线

- WLB-LLM：用 profiling 拟合 $W_a(s) + W_l(s)$，变长打包 + outlier delay
- ByteScale：同样 profiling，但打破每 rank micro-batch 数相等的假设
- LobRA：引入配置异构（多种并行度的 replica 共存）

这些方案在**中等序列长度**（$s \lesssim 3h$，线性项仍占主导时）有效。

### 3. 解耦 Attn/FFN（CAD/DistCA）是长上下文 DP 的最终答案

- 核心注意力无状态 → 可以任意调度
- token-shard 级切分 + kernel composability → 搜索空间大，必可平衡
- **同时实现**计算平衡 + 显存平衡（而非二选一）
- 长上下文（512K+）、大 DP（≥8）、长尾分布下收益最显著

### 4. 显存不平衡的量级（DP-only 视角）

| 场景 | Attn FLOPs 不平衡 | 激活显存不平衡 |
|------|----------------|--------------|
| 短序列均匀（$s<h$）| <2× | <1.1× |
| 混合长短（3h < s_max < 10h）| 5–20× | 1.0–1.2× |
| 极端长尾（s_max ≥ 100h）| 50–200× | 1.0–1.2× |
| CAD 解耦后 | 1.0× | 1.0× |

→ **显存其实很容易平衡**（FlashAttn 功劳），**计算不平衡才是真正的痛点**。这也正是 CAD 的切入点：先承认显存平衡容易（按 token 数分），再单独解决计算平衡（把 $\sum l^2$ 项独立调度）。

---

## 八、参考来源

| 论文 | 方案 | arXiv | 主要贡献 |
|------|------|-------|---------|
| **CAD / DistCA** | 核心注意力解耦 | [2510.18121](https://arxiv.org/abs/2510.18121) | 显式解耦 CA，DP 规模 8+ 的 1.35× 加速 |
| **ByteScale** | HDP + Balance Scheduler | [2502.21231](https://arxiv.org/abs/2502.21231) | 12000+ GPU 生产系统，DP/PP 统一调度 |
| **WLB-LLM** | Var-Len Packing + Outlier Delay | [2503.17924](https://arxiv.org/abs/2503.17924) | profiling 双项模型，1.23× 加速 |
| **LobRA** | 异构 FT 副本 | [2509.01193](https://arxiv.org/abs/2509.01193) | 多租户长度桶分发 |
| **AFD 理论** | Attn/FFN 解耦的闭合解 | [2601.21351](https://arxiv.org/abs/2601.21351) | 推理场景的理论基础 |
| **MegaScale-Infer** | 推理侧解耦 | [2504.02263](https://arxiv.org/abs/2504.02263) | ping-pong pipeline |
| **ChunkFlow** | Chunk 均衡 | [2503.02356](https://arxiv.org/abs/2503.02356) | 训练侧 chunk 粒度 |

## 九、相关笔记

- [[Attention-FFN计算量不平衡与解耦调研]]：综合调研（含 TP/PP/CP 讨论）
- [[DWDP - Distributed Weight Data Parallelism for LLM Inference]]：推理场景 DP + 异步预取
- [[ChunkFlow - Efficient Long Context Fine-tuning with Chunk Flow]]：chunk 视角的另一解
