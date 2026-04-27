---
tags: [实体, 硬件, CPU]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-gpu-bandwidth-hierarchy, 2026-04-22-long-context-supernode-survey]
---

# Grace-CPU

> NVIDIA 基于 ARM Neoverse V2 的服务器 CPU，**72 核**，LPDDR5X 内存 **546 GB/s**。是 [[GB200-NVL72]]、[[GH200-Superchip]] 的关键组件。

## 关键参数

- **核数**：72 ARM Neoverse V2
- **内存**：480 GB LPDDR5X
- **内存带宽**：546 GB/s per CPU
- **与 GPU 的连接**：[[NVLink-C2C]] 900 GB/s

## 与传统 x86 CPU 的对比

| 项目 | Grace | Xeon 8480C (DGX H100) |
|------|-------|--------------------|
| 内存带宽 | **546 GB/s** | ~307 GB/s |
| 与 GPU 互联 | NVLink-C2C 900 GB/s | PCIe Gen5 128 GB/s |
| FLOPS vs GPU（GH200）| 1:330 | — |

## 角色定位

在 [[SuperOffload]] 的分析中，Grace CPU 有三重角色：

1. **显存扩展**（被 GPU 以近显存速度访问）
2. **Optimizer Step 承载**（CPU-GPU overlap）
3. **辅助 cast / transform**（但需注意临时内存路径不友好的陷阱）

## 相关页面

- [[NVLink-C2C]]
- [[GH200-Superchip]]
- [[GB200-NVL72]]
- [[SuperOffload]]
