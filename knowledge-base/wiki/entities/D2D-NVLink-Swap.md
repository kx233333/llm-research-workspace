---
tags: [实体, 概念, NVLink, 显存优化]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-mpress]
---

# D2D-NVLink-Swap

> Device-to-Device Swap via NVLink：[[MPress]] 的核心原语。把显存压力 stage 的张量通过 NVLink 异步拷贝到同节点闲置 GPU，需要时再拉回。

## 对比 CPU Swap

| 对比维度 | CPU Swap（PCIe）| D2D Swap（NVLink）|
|----------|----------------|------------------|
| 带宽 | ~16 GB/s | ~600 GB/s（NVLink 3.0）/ 1.8 TB/s（NVLink 5.0）|
| 延迟 | 高 | 极低 |
| 显存来源 | CPU DRAM | 同节点闲置 GPU |

## 工程要点

- 活跃/闲置 GPU 识别由 static plan 离线决定
- 传输可与计算 overlap
- 需要 "DeviceMapping"（pipeline stage → GPU 映射）
- 需要 "DataStripping"（大张量切条带，多 NVLink 并行）

## 相关页面

- [[MPress]]
- [[NVLink]]
- [[NVLink-C2C]]
- [[显存优化]]
