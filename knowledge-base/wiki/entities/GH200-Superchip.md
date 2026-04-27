---
tags: [实体, 硬件, 超节点]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-long-context-supernode-survey]
---

# GH200-Superchip

> NVIDIA Grace-Hopper Superchip：Grace CPU 与 H100 GPU 通过 NVLink-C2C 紧耦合（**900 GB/s**），是 [[SuperOffload]] 论文的硬件目标。

## 关键规格

- **GPU**：H100 SXM5
- **CPU**：Grace（72 ARM Neoverse V2 核）
- **CPU-GPU 互联**：[[NVLink-C2C]]（900 GB/s 双向）
- **FLOPS 比**：Hopper:Grace ≈ **330**（远高于 DGX-2 的 60 和 DGX-A100 的 135）

## 与 NVL72 的区别

| 项目 | GH200 Superchip | GB200 NVL72 |
|------|----------------|------------|
| GPU 数 | 1 | 72 |
| CPU 数 | 1 | 36 |
| CPU-GPU 互联 | NVLink-C2C | NVLink-C2C（同）|
| GPU 架构 | Hopper (H100) | Blackwell (B200) |
| 典型场景 | 单机 offload 优化 | 大规模训练/推理 |

## 为什么重要

- **首个把 CPU-GPU 视为 L3 存储层的硬件**
- SuperOffload 论文指出：过去 offload 基于 PCIe 松耦合的旧前提需要重写

## 相关页面

- [[NVLink-C2C]]
- [[Grace-CPU]]
- [[SuperOffload]]
- [[多层级内存管理]]
- [[GB200-NVL72]]
