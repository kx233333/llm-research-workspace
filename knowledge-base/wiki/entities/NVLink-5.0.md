---
tags: [实体, 硬件, 互联]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-gpu-bandwidth-hierarchy, 2026-04-22-dwdp]
---

# NVLink-5.0

> NVIDIA 第五代 NVLink，Blackwell 架构。**1.8 TB/s per GPU 双向**，是 [[GB200-NVL72]] 和 [[DGX-B200]] 的核心互联。

## 关键参数

- **单 GPU 双向带宽**：1.8 TB/s
- **相比 NVLink 4.0**（H100）的提升：**2×**
- **在 NVL72 中域聚合带宽**：130 TB/s（72 GPU）
- **与 PCIe Gen5 对比**：~14×

## 对算法的关键意义

- 让 [[DWDP]] 的跨 GPU 异步权重预取成为可行
- 让 [[MegaScale-Infer]] 的 M2N 通信避开 IB 瓶颈
- 使得跨 8 卡的 a2a 不再是带宽瓶颈（但仍是同步瓶颈）

## 相关页面

- [[NVLink-4.0]]
- [[NVLink-C2C]]
- [[NVSwitch]]
- [[GB200-NVL72]]
- [[DGX-B200]]
