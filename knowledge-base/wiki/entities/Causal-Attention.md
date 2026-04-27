---
tags: [实体, 概念, Attention]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-chunkflow]
---

# Causal-Attention

> 带因果掩码的 Attention：每个位置只能看到自己及之前的 token，不能看到未来。Decoder-only LLM 的标配。

## 单向依赖性的系统意义

```
Forward:  token_k 的 Attention 只需读 token_0 … token_k 的 K, V
Backward: token_k 的 grad 只需后续 token_{k+1..n} 的 grad KV
```

**关键启发**：[[ChunkFlow]] 利用这个性质——forward 两次时第一次只保 KV state（前序依赖），backward 时也不需要重新计算前序的完整激活。

## 与其他 attention 模式的对比

| 模式 | 依赖方向 | 适用 |
|------|---------|------|
| Causal（本页）| 单向（左→右）| Decoder-only LLM |
| Full | 双向 | Encoder（BERT 类） |
| Prefix | 前缀双向，后缀单向 | 部分 instruction tuning |

## 相关页面

- [[ChunkFlow]]
- [[State-Aware 1F1B]]
- [[FlashAttention]]
