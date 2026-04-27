---
tags: [实体, 硬件, 显存]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-gpu-bandwidth-hierarchy]
---

# HBM3e

> 第 N 代高带宽内存（High Bandwidth Memory 3e）。B200 单卡 **8 TB/s**、192 GB，**相比 H100 的 HBM3（3.35 TB/s, 80GB）提升 2.4×**。

## 关键参数

- **带宽**：8 TB/s per GPU（B200）
- **容量**：192 GB per GPU（B200 in NVL72）/ 180 GB（DGX B200）
- **与 HBM3 对比**：带宽 2.4×，容量 2.25–2.4×
- **总线**：8192-bit（B200）

## 对算法的意义

- **长上下文推理的硬前提**：192 GB 单卡可以放下 1M token 的 KV cache
- **激活显存不再是瓶颈**（训练侧配合 [[FlashAttention]]）
- **减轻了 offload 的必要性**

## 相关页面

- [[GB200-NVL72]]
- [[DGX-B200]]
- [[FlashAttention]]
- [[KV-Cache]]
