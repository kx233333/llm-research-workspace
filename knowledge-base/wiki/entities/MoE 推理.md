---
tags: [实体, 概念, 推理]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-dwdp, 2026-04-22-attention-ffn-imbalance-survey]
---

# MoE 推理

> **M**ixture **o**f **E**xperts 推理的独特挑战：FFN 层稀疏激活，每 token 只路由到 top-K expert；但推理时每步 batch 的 expert 激活情况不可预测，导致负载不均衡。

## 独特问题

1. **Expert 路由偏斜**：热门 expert 收到更多 token
2. **a2a dispatch/combine 成本**：每层两次
3. **激活稀疏导致 GPU 利用率低**：大部分 expert 未被选中

## 代表方案

| 方向 | 方案 |
|------|------|
| 消灭 a2a | [[DWDP]] |
| 保留但隐藏 a2a | [[MegaScale-Infer]] |
| 理论最优比例 | [[AFD-Ratio 理论]] |

## 硬件需求趋势

- 大显存（expert 权重总和巨大）
- 高 P2P 带宽（expert 分散存储 + 预取）
- [[GB200-NVL72]] 是理想平台

## 相关页面

- [[DWDP]]
- [[MegaScale-Infer]]
- [[AFD-Ratio 理论]]
- [[MoE-a2a]]
- [[分离部署]]
