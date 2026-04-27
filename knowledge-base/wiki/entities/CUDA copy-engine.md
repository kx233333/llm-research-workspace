---
tags: [实体, 概念, 通信, 硬件]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-dwdp]
---

# CUDA copy-engine

> NVIDIA GPU 上独立于 SM 的 DMA 引擎，负责 HBM ↔ 其他 GPU HBM ↔ CPU DRAM 之间的数据搬运。是 [[DWDP]] 等异步预取方案的硬件基础。

## 关键特性

- **不占用 SM 资源**：计算和传输真正并行
- **单 GPU 通常有 2 个 copy engine**：一个用于 H2D、一个用于 D2H / D2D
- **可被 `cudaMemcpyAsync` / `cudaMemPrefetchAsync` 调度**

## 在 DWDP 中的角色

- 承担 [[异步 P2P 预取]] 的所有跨 GPU 传输
- 因为是独立引擎，不影响 MoE/Attention 的 SM 计算
- 但存在多对一竞争问题，需要 TDM 切片调度

## 相关页面

- [[异步 P2P 预取]]
- [[DWDP]]
- [[NVLink-5.0]]
