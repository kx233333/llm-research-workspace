---
tags: [实体, 概念, 训练, 调度]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-chunkflow]
---

# Chunk 粒度调度

> 把变长序列重组为近似等长 chunk，以 chunk 为调度单位的思想。[[ChunkFlow]] 的核心抽象。

## 核心价值

| 问题 | chunk 粒度下的解法 |
|------|------------------|
| 峰值显存由最长序列决定 | 峰值显存由 ChunkSize 决定 |
| 各 GPU 计算量差异大 | chunk 近似等长 → 计算均衡 |
| Pipeline bubble 高 | chunk 规则长度 → bubble 降低 |

## 与其他变长方案的对比

| 方案 | 粒度 | 思路 |
|------|------|------|
| [[WLB-LLM]] Var-Len | 文档 | 重组 packing 平衡 $W_a + W_l$ |
| [[ByteScale]] | 序列 | 允许各 rank 不同数 micro-batch |
| **[[ChunkFlow]]** | **Chunk (ChunkSize)** | **等长化，串行调度** |
| [[CAD-DistCA]] | Token-shard | 核心注意力分布式迁移 |

## 开放问题

- ChunkSize 能否变长？（docx 调研者的思考）
- 同 chunk 内混合长/短序列是否引入新的不均？

## 相关页面

- [[ChunkFlow]]
- [[State-Aware 1F1B]]
- [[长上下文训练]]
- [[负载均衡与变长序列]]
