---
tags: [实体, 概念, 模型架构, 注意力机制]
created: 2026-04-29
updated: 2026-04-29
aliases: [混合注意力, Hybrid-Attention, 线性注意力混合]
---

# Hybrid Attention

## 定义

Hybrid Attention 是下一代大模型采用的混合注意力架构，将**线性复杂度的注意力层**（如 KDA、SWA、Lightning、GDN）与**少量全注意力层**（如 [[MLA]]、GQA）组合使用。

## 核心优势

线性注意力层产生**固定大小的 recurrent state**（与序列长度无关），只有少数全注意力层产生 O(seq_len) 的 [[KV-Cache]]。因此：

$$\text{KVCache}_{hybrid} = \frac{1}{R+1} \cdot \text{KVCache}_{dense} + \text{固定 state}$$

其中 $R$ 是线性层:全注意力层的比例。

## 代表模型

| 模型 | 参数量 | 比例 (线性:全注意力) | 线性层类型 | KV 缩减 |
|------|--------|-------------------|-----------|---------|
| Kimi Linear | 48B | 3:1 | KDA | ~4× |
| MiMo-V2-Flash | 309B | 5:1 | SWA | ~13× |
| Ring-2.5-1T | 1T | 7:1 | Lightning | ~36× |
| Qwen3.5-397B | 397B | 3:1 | GDN | ~4× |

## 系统影响

- **推理**：KVCache 缩减使 [[PrfaaS|跨数据中心 Prefill 卸载]] 成为可能
- **显存**：大幅降低 Decode 阶段的 KV-Cache 显存占用
- **长上下文**：线性层的固定 state 不随上下文增长

## 关联概念

- [[KV-Cache]]
- [[PrfaaS]]
- [[分离部署]]
- [[FlashAttention]]
