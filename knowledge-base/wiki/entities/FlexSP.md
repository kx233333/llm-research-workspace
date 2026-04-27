---
tags: [实体, 方案, 训练系统, 序列并行]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-long-context-supernode-survey]
---

# FlexSP

> **Flex**ible **S**equence **P**arallelism（ASPLOS 2025，北大崔斌团队）：在变长训练中自适应形成多个**异构 SP 组**，让不同长度序列走不同并行度的分组，破解"统一 SP 分组无法兼顾长短序列"的难题。

## 关键信息

- **发表**：ASPLOS 2025
- **来源组织**：北京大学崔斌老师团队
- **所属主题**：[[负载均衡与变长序列]]

## 核心问题

- 长序列要求大显存 → 强制大 SP 度
- 短序列在大 SP 度下效率低下
- **统一 SP 分组导致通算不均衡**（长尾分布）

## 核心机制

两个模块：

1. **Sequence Blaster**：对序列做 bin-packing（动态规划）
2. **Parallelism Planner**：整数规划求解 SP 组配置 + 序列分配

数学形式化：给定 K 条样本和资源，求：
- SP 组数
- 每组 SP 度
- 每序列归属哪组

## 与其他方案的对比

| 对比 | 关系 |
|------|------|
| [[WLB-LLM]] | FlexSP 改 SP 度，WLB-LLM 改 packing 内容，正交 |
| [[LobRA]] | 都用异构 replica 配置，FlexSP 面向预训练，LobRA 面向多租户 FT |
| [[ChunkFlow]] | FlexSP 在 SP 层调整，ChunkFlow 在输入层调整 |

## 局限（docx 评论）

> 这里主要是 packing 层面的优化，揭示了异构长度下的适配思想，但是分组/分配的依据是直觉。

## 相关页面

- [[负载均衡与变长序列]]
- [[WLB-LLM]]
- [[USP]]
- [[LobRA]]
