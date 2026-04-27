---
tags: [实体, 硬件, 互联]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-gpu-bandwidth-hierarchy]
---

# ConnectX-SuperNIC

> NVIDIA 的高性能网卡系列，为节点间 scale-out 网络提供带宽。

## 代际对比

| 代次 | 单端口带宽 | 主机接口 | 配套 |
|------|----------|---------|------|
| ConnectX-7 | 400 Gbps | PCIe Gen5 x16 | DGX H100 / B200，Quantum-2 IB |
| **ConnectX-8** | **800 Gbps** | PCIe Gen6 × 48 lanes | **GB200 NVL72**，Quantum-X800 IB |

## 在 NVL72 中的角色

- 每 Grace Superchip 配 1 个 ConnectX-8
- NVL72 机柜内共 36 个 → 总 scale-out 带宽 ~3.6 TB/s
- 相比 DGX H100（8 × 50 GB/s = 400 GB/s）提升 ~9×

## 对算法的意义

- 跨 NVL72 机柜时 **仍然受 IB 约束**
- 但比传统 IB 带宽高 2×，降低了跨机柜训练的通信瓶颈

## 相关页面

- [[GB200-NVL72]]
- [[DGX-H100]]
- [[DGX-B200]]
