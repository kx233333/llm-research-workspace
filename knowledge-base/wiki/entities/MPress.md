---
tags: [实体, 方案, 训练系统, 显存优化]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-mpress, 2026-04-22-long-context-supernode-survey]
---

# MPress

> 单节点多 GPU 训练系统（HPCA 2023），用 D2D NVLink swap 把闲置 stage 显存"借"给活跃 stage，突破 GPU 显存墙。

## 关键信息

- **类型**：训练系统 / 显存优化方案
- **发表**：HPCA 2023，DOI: 10.1109/HPCA56546.2023.10071077
- **作者**：Quan Zhou, Haiquan Wang, Xiaoyan Yu, Cheng Li, Youhui Bai, Feng Yan, Yinlong Xu
- **目标模型**：BERT / GPT 亿级参数
- **硬件基线**：DGX-1 (8× V100) / DGX-2 (16× V100)
- **所属主题**：[[显存优化]]、[[并行策略与同步开销]]

## 核心机制

三级联合优化（流水线并行 + 选择性重计算 + [[D2D-NVLink-Swap]]）：

1. **算子间并行（Pipeline Parallelism）** 作为底座
2. **D2D NVLink swap**：NVLink ~600 GB/s vs PCIe ~16 GB/s，把 swap 目标从 CPU DRAM 换成同节点闲置 GPU
3. **选择性重计算**：对后层、重算便宜的激活按需重算
4. **Static plan + Runtime executor**：离线生成 memory compaction plan，运行时动态执行 swap-in/out/drop/recompute

## 实验结果

| 对比 | 加速比 |
|------|-------|
| vs ZeRO（相同显存约束）| **1.7× – 2.3×** |
| DGX-1 → DGX-2 扩展 | 近线性 |

## 局限与适用边界

- **单节点场景**（跨节点 NVLink 不可用会退化到 PCIe）
- **静态 plan 在动态路由（MoE）下会失效**（docx 调研者的评论）<!-- confidence: INFERRED -->
- **GPU 对称拓扑前提**（新机器如 NVL72 / DGX H100 无此问题）

## 不同素材中的观点

- [[2026-04-22-mpress]]：详细方案分析
- [[2026-04-22-long-context-supernode-survey]]：归类为 "PP stage Device Memory 不均衡" 的代表方案
- docx 调研者评论："联合优化的优越性" + "静态分析前提是相似的数据分布，现在做联合策略应该更注重 feedback 做动态策略"

## 相关页面

- [[D2D-NVLink-Swap]]
- [[NVLink]]
- [[算子间并行]]
- [[显存优化]]
- [[ChunkFlow]] — 都是显存优化方向，但 ChunkFlow 是时间换，MPress 是空间换
- [[DWDP]] — 都用 NVLink 跨 GPU 传输，但前者推理、后者训练
