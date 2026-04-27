---
tags: [实体, 方案, 多租户, LoRA, 训练系统]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-dp-only-attention-ffn-survey]
---

# LobRA

> **Lo**ra **B**atched **R**eplicas over heterogeneous **A**daptations（arXiv: 2509.01193）：多租户 LoRA fine-tuning 系统，用**异构 FT 副本**承接不同长度桶的序列，减少 GPU-seconds **45–60%**。

## 关键信息

- **arXiv**：2509.01193
- **组织**：PKU-DAIR（Hetu 框架）
- **GitHub**：https://github.com/PKU-DAIR/Hetu

## 核心思路

放弃"同构 DP"假设：
- 长序列 → 高 TP 副本（更多 GPU，避免 OOM）
- 短序列 → 低 TP 副本（更少 GPU，效率更高）

## 优化问题

$$
\min_{p_i, d_{i,j}} \max_i T(\{\lceil d_{i,j}/p_i \rceil\}; \mathcal{S}_i)
$$

约束：
- 所有序列被处理
- 未选中配置不分配
- GPU 总数不超

## 与其他方案的对比

- [[FlexSP]]：同用异构配置，FlexSP 面向预训练（SP 组异构），LobRA 面向多租户 FT（TP 度异构）
- [[WLB-LLM]]：都处理长尾，但 LobRA 走"资源异构"路径，WLB-LLM 走"数据重组"路径

## 相关页面

- [[负载均衡与变长序列]]
- [[FlexSP]]
