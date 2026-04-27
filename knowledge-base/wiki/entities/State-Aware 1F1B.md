---
tags: [实体, 概念, 训练调度]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-chunkflow]
---

# State-Aware 1F1B

> [[ChunkFlow]] 的核心调度算法：将 chunk 化后的输入接入标准 1F1B pipeline，利用 causal attention 的单向依赖，让 pipeline bubble 从 57.14% 降到 47.8%（K=2）。

## 核心机制

### 参数 K 的语义

K = 最多同时保存 activation 的 chunk 数。

| 情况 | 行为 |
|------|------|
| N ≤ K | 正常顺序 forward → 逆序 backward |
| **N > K** | 前 (N-K) chunk **forward 两次**（第一次丢弃 activation 只保 KV state）|

显存峰值：**K × ChunkSize**（与最长序列无关）。

### 关键洞察

> Causal Attention 的单向依赖性：forward 只需前序 KV，backward 只需后续 KV 梯度。

### Bubble 率对比

| 配置 | Bubble 率 |
|------|---------|
| 标准 1F1B（变长）| 57.14% |
| State-Aware 1F1B K=1 | 54.1% |
| State-Aware 1F1B K=2 | **47.8%** |

## (ChunkSize, K) 权衡

固定 $\text{CS} \times K$ 不变：
- CS 过小 → chunk 碎 → GPU 效率低
- CS 过大 → chunk 数少 → bubble 增加
- **最优需 grid search**（7B 256K 场景：(CS=8K, K=4) 最优）

## 相关页面

- [[ChunkFlow]]
- [[Chunk 粒度调度]]
- [[Causal-Attention]]
- [[长上下文训练]]
- [[并行策略与同步开销]]
