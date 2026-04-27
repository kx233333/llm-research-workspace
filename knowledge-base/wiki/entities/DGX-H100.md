---
tags: [实体, 硬件]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-gpu-bandwidth-hierarchy]
---

# DGX-H100

> NVIDIA 传统 8 卡训练节点（Hopper 架构）：NVLink 4.0 每 GPU 900 GB/s，4 个 NVSwitch，跨 8 卡边界需 IB。是大量训练系统的默认基线。

## 关键规格

| 项目 | 数值 |
|------|------|
| GPU | 8× H100 SXM5 |
| GPU 显存 | 80 GB HBM3 × 8 = 640 GB |
| CPU | 2× Intel Xeon 8480C |
| NVSwitch 芯片 | 4 个 |
| NVLink | 4.0（900 GB/s per GPU） |
| NVLink 聚合 | 7.2 TB/s |
| HBM 带宽 | 3.35 TB/s per GPU |
| 节点间 NIC | ConnectX-7（400 Gbps） |
| CPU-GPU 互联 | PCIe Gen5 x16（128 GB/s 双向） |

## 与 NVL72 的核心差异

| 项目 | DGX H100 | GB200 NVL72 |
|------|----------|------------|
| NVLink 域大小 | 8 GPU | **72 GPU** |
| 跨节点通信 | **必经 IB**（50 GB/s）| 域内 NVLink（1.8 TB/s）|
| 域内/域外带宽比 | 18× | **360×**（更锐利的层级）|
| CPU-GPU | PCIe Gen5 128 GB/s | **NVLink-C2C 900 GB/s** |

## 为什么 [[DWDP]] 等方案在 DGX H100 上失效

- 跨节点 P2P 只有 50 GB/s，在一层计算窗口内无法完成下一层 expert 权重预取（需约 160-320 GB/s/GPU）

## 相关页面

- [[GB200-NVL72]]
- [[DGX-B200]]
- [[NVLink-4.0]]
- [[超节点架构]]
