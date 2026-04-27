---
tags: [实体, 概念, Attention]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-dp-only-attention-ffn-survey]
---

# Core-Attention

> 核心注意力（Core Attention, CA）：Transformer 中的 $\text{softmax}(QK^T)V$ 部分，**无可训练参数**、只有瞬时数据。是 [[CAD-DistCA]] 能够做训练侧解耦的关键性质。

## 两个关键性质

### 1. Stateless（无状态）

- 无可训练参数
- 只有瞬时 $Q, K, V$ 张量
- **平衡问题从"带状态的张量迁移"退化为"无状态计算任务调度"**

### 2. Composable（可组合）

- token-level 可任意切分/合并
- 现代 FlashAttention kernel 对变长 fused 输入保持高 MFU（shard ≥ 128 tokens 时）
- 不同文档的 shard 可融合进一次 kernel call

## 与其余层的 FLOPs 划分

对 token 长度 $l$：

$$
\text{FLOPs}_\text{CA}(l) = \alpha l^2, \quad \text{FLOPs}_\text{其余}(l) = \beta l
$$

变长序列下：
- $\sum l_i^2$（CA FLOPs）难以平衡
- $\sum l_i$（其余 FLOPs + Memory）容易平衡
- **CAD 的切入点**：把 $\sum l_i^2$ 项独立调度，$\sum l_i$ 项留在原地

## 相关页面

- [[CAD-DistCA]]
- [[FlashAttention]]
- [[Attention-FFN 计算量不平衡]]
- [[Attention-FFN 解耦]]
