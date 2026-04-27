---
tags: [实体, 硬件, 互联]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-gpu-bandwidth-hierarchy, 2026-04-22-long-context-supernode-survey]
---

# NVLink-C2C

> **C**hip-to-**C**hip NVLink：Grace CPU 与 B200 GPU（或 H100）之间的片间直连，**900 GB/s 双向**。是 [[GB200-NVL72]] 和 [[GH200-Superchip]] 的关键组件。

## 关键参数

- **双向带宽**：900 GB/s
- **相比 PCIe Gen5 x16**：**~7×**
- **相比 Grace LPDDR5X 带宽**（546 GB/s）：更高
- **延迟**：极低（片间级，而非 PCIe bus）

## 结构性差异

在传统 DGX 中，PCIe 是 CPU↔GPU 的唯一通道，128 GB/s 远低于 GPU 显存带宽（3.35 TB/s），使 CPU DRAM 对 GPU 几乎不可用。NVLink-C2C 以 900 GB/s 连接 Grace CPU 与 B200 GPU，打通了这一瓶颈。

## 使能的新模式

1. **CPU DRAM 作为 GPU 的有效 L3 存储**（[[SuperOffload]]）
2. **统一内存池**（NVL72 72 GPU × 192 GB HBM + 36 CPU × 480 GB LPDDR5X ≈ 30 TB）
3. **MoE 专家权重可以冷却到 CPU 侧而不显著降速**

## 相关页面

- [[Grace-CPU]]
- [[NVLink-5.0]]
- [[GB200-NVL72]]
- [[GH200-Superchip]]
- [[SuperOffload]]
- [[显存优化]]
