---
tags: [实体, 硬件]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-gpu-bandwidth-hierarchy]
---

# DGX-B200

> NVIDIA 8 卡 Blackwell 节点：升级到 NVLink 5.0 和 HBM3e，但仍然是 **传统 8 卡域**，只是每卡升级，不解决跨节点扩展问题。

## 关键规格

| 项目 | 数值 |
|------|------|
| GPU | 8× B200 SXM |
| GPU 显存 | 180 GB HBM3e × 8 = 1.44 TB |
| CPU | 2× Intel Xeon 8570 |
| NVSwitch 芯片 | 2 个 |
| NVLink | 5.0（1.8 TB/s per GPU） |
| NVLink 聚合 | 14.4 TB/s |
| HBM 带宽 | 8 TB/s per GPU |
| 节点间 NIC | ConnectX-7（400 Gbps） |
| 功耗 | ~14.3 kW（液冷） |

## 与 GB200 NVL72 的关系

- **同代 GPU**（B200 架构）
- 但 NVLink 域仍是 **8 卡**，不是 72 卡
- 单 NVLink 域内带宽 14.4 TB/s，**只有 NVL72 的 1/9**
- **DGX B200 不能用作 [[DWDP]] 的硬件**（跨节点带宽不够）

## 相关页面

- [[GB200-NVL72]]
- [[DGX-H100]]
- [[NVLink-5.0]]
- [[HBM3e]]
