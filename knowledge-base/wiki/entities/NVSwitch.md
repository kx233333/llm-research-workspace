---
tags: [实体, 硬件, 互联]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-gpu-bandwidth-hierarchy]
---

# NVSwitch

> NVIDIA 的 NVLink 交换芯片：把多个 GPU 的 NVLink 端口聚合成单一 fabric，决定 NVLink 域的规模。

## 不同平台的 NVSwitch 配置

| 平台 | NVSwitch 芯片数 | NVLink 域 GPU 数 |
|------|---------------|----------------|
| DGX H100 | 4 | 8 |
| DGX B200 | 2 | 8 |
| **GB200 NVL72** | **18（9 托盘 × 2）** | **72** |

## 结构性意义

- NVSwitch 数量 + 拓扑决定了**单一 NVLink 域的扩展上限**
- NVL72 的 18 个 NVSwitch 是把 72 GPU 组成单一 fabric 的硬件基础
- 这也是 NVL72 打破传统 8 卡节点边界的核心创新

## 相关页面

- [[GB200-NVL72]]
- [[NVLink-5.0]]
- [[NVLink-4.0]]
