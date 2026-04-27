---
tags: [实体, 概念, 通信, CP]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-long-context-supernode-survey]
---

# CP-a2a

> 上下文并行（Context Parallelism）中的 a2a 操作，典型代表是 DeepSpeed-Ulysses：从 sequence-parallel 布局切换到 head-parallel 布局时的 a2a 重分布，每 Attention 层两次。

## 与 [[MoE-a2a]] 的对比

| 维度 | MoE-a2a | CP-a2a |
|------|---------|--------|
| 触发 | token 路由到 expert | 张量从 seq-split 到 head-split |
| 频率 | 每 MoE 层 × 2 | 每 Attention 层 × 2 |
| 并行度约束 | expert 数 | head 数、节点数 |
| 不均衡来源 | 路由偏斜 | 长短序列打包 |

## 优化路径

### 路线 A：类 ChunkFlow 串行

[[ChunkFlow]] 通过 chunk 粒度调度，同一序列的多个 chunk 在同 GPU 串行执行，**用 P2P KV 传递替代 a2a**。是 docx 调研者对"CP 的 a2a 如何被优化掉"的关键猜想。

### 路线 B：Hybrid

[[USP]]：高带宽域内用 Ulysses（a2a），跨域用 Ring（p2p）混合。

### 路线 C：理论上的超节点完全消除

在 [[GB200-NVL72]] 这种单 NVLink 域上，ChunkFlow 思路甚至可以扩展到 72 GPU 范围——这是 **ChunkFlow + DWDP 三位一体** 研究的种子问题。<!-- confidence: INFERRED -->

## 相关页面

- [[MoE-a2a]]
- [[ChunkFlow]]
- [[USP]]
- [[Ulysses-CP]]
- [[Ring-Attention]]
- [[长上下文训练]]
