---
tags: [实体, 概念, 硬件]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-mpress, 2026-04-22-gpu-bandwidth-hierarchy]
---

# NVLink

> NVIDIA 的 GPU 间高带宽互联技术，贯穿多代硬件。本页是总览，具体代际见子页。

## 代际演进

| 代次 | per-GPU 带宽 | 典型平台 | 作为 NVLink 域 |
|------|------------|---------|-------------|
| NVLink 3.0 | ~600 GB/s | DGX A100 | 8 GPU |
| NVLink 4.0 | 900 GB/s | [[DGX-H100]] | 8 GPU |
| **[[NVLink-5.0]]** | **1.8 TB/s** | [[DGX-B200]]、[[GB200-NVL72]] | 8 / 72 GPU |

## 与 [[NVLink-C2C]] 的区别

- NVLink：GPU ↔ GPU
- NVLink-C2C：CPU ↔ GPU（chip-to-chip）

二者都是 NVLink 协议族，但接口和连接对象不同。

## 核心算法依赖

- [[DWDP]]：NVLink 5.0 + 单 NVLink 域 72 GPU
- [[MPress]]：任何版本 NVLink（D2D swap）
- [[MegaScale-Infer]]：NVLink 5.0

## 相关页面

- [[NVLink-5.0]]
- [[NVLink-C2C]]
- [[NVSwitch]]
- [[D2D-NVLink-Swap]]
