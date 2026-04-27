---
tags: [实体, 方案, 训练系统, 负载均衡, 长上下文]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-dp-only-attention-ffn-survey]
---

# ByteScale

> ByteDance + PKU 的超大规模长上下文训练系统（arXiv: 2502.21231）：Hybrid Data Parallelism (HDP) 统一 inter-data 与 intra-data 切分，Balance Scheduler 在 DP 层允许**各 rank 处理不同数量的 micro-batch**（打破梯度累积等量约束），**12000+ GPU / 2048K 上下文**。

## 关键信息

- **arXiv**：2502.21231
- **规模**：> 12000 GPU，256K–2048K 上下文
- **加速比**：up to **7.89×** vs SOTA

## 核心机制

### HDP（Hybrid DP）

- 合并 inter-data（DP）与 intra-data（CP）切分
- 动态 mesh，不再静态
- 按数据特征选择性 offload

### Balance Scheduler

两个核心洞察：

**Insight 1**：PP bubble 在长度相似的序列被分到同 pipeline 时更小

**Insight 2**：只有纯 DP 时，才只需要各 rank 在同一时间点计算量相近

具体策略：
1. 按 FLOPs 分桶（每桶 $\sum \text{FLOPs}$ 近似相等 → 长序列桶含更少样本）
2. 同一时刻各 rank 从同一桶抽取
3. **处理短序列的 rank 可被分配更多 micro-batch**

## 与其他方案的对比

| 对比 | 关系 |
|------|------|
| [[WLB-LLM]] | 同目标，WLB-LLM 保持 micro-batch 等量约束，ByteScale 打破 |
| [[ChunkFlow]] | ChunkFlow 用 chunk 均一化避免估算，ByteScale 直接做精确估算+调度 |
| [[CAD-DistCA]] | CAD 指出 ByteScale 类方案在高 DP 规模下 memory divergence 增大 |

## 相关页面

- [[负载均衡与变长序列]]
- [[Attention-FFN 计算量不平衡]]
- [[WLB-LLM]]
- [[CAD-DistCA]]
