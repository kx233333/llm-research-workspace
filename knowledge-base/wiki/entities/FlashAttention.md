---
tags: [实体, 概念, Attention, 显存]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-attention-ffn-imbalance-survey, 2026-04-22-dp-only-attention-ffn-survey]
---

# FlashAttention

> IO-aware 的 Attention kernel：不显式存储 $P = \text{softmax}(QK^T)$，只保留 LSE（log-sum-exp），backward 按需重算。**把 Attention 激活显存从 $O(s^2)$ 降至 $O(s)$**。

## 关键意义

### 对训练显存平衡的影响

| 组件 | 无 FlashAttn | 有 FlashAttn |
|------|------------|------------|
| Attention 激活 | $O(s^2)$ per layer | $O(s)$ per layer |
| FFN 激活 | $O(s)$ | $O(s)$ |
| 训练显存平衡 | 严重不均 | 基本均衡 |

→ **训练场景下 Attn/FFN 显存不平衡问题已被 FlashAttention 大幅缓解**

### 对 [[CAD-DistCA]] 的支撑

CAD 论文明确指出：

> activations saved for backward are therefore dominated by context-independent layers.

即有了 FlashAttn 后，显存支配项从 Attention 转为其余层，这是 CAD 能够独立平衡计算与显存的前提。

## 其他算法依赖

- [[BPipe]] 类方案在 FlashAttn 下收益大幅减弱
- [[ChunkFlow]] 依赖 FlashAttn 才能让 KV state 保留代价可控

## 相关页面

- [[Core-Attention]]
- [[CAD-DistCA]]
- [[Attention-FFN 计算量不平衡]]
- [[显存优化]]
