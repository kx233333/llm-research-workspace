---
tags: [素材摘要, 硬件, 带宽, NVL72, 超节点]
created: 2026-04-22
updated: 2026-04-22
source_type: 本地笔记
source_path: raw/notes/GPU服务器带宽层级对比 - GB200 NVL72 vs DGX H100 vs DGX B200.md
---

# GPU 服务器带宽层级全景：GB200 NVL72 vs DGX H100 vs DGX B200

> 自下而上梳理五层带宽（HBM → CPU DRAM → CPU-GPU 互联 → GPU 间 NVLink → 节点间网络），对比三代平台，揭示算法设计的硬件边界。

## 基本信息

- **消化日期**：2026-04-22
- **文档性质**：技术调研表格 + 层级分析

## 核心观点（五层速览）

| 层级 | NVL72 | DGX H100 | DGX B200 | 跨层带宽悬崖 |
|------|-------|----------|----------|------------|
| L1 HBM | 8 TB/s | 3.35 TB/s | 8 TB/s | — |
| L2 CPU DRAM | 546 GB/s × 36 | 307 GB/s × 2 | 307 GB/s × 2 | ~10× |
| L3 CPU↔GPU | **900 GB/s NVLink-C2C** | 128 GB/s PCIe Gen5 | 128 GB/s PCIe Gen5 | **~7× 差异** |
| L4 GPU 间 | 1.8 TB/s, 130 TB/s 域 | 900 GB/s, 7.2 TB/s 域 | 1.8 TB/s, 14.4 TB/s 域 | **18× 域带宽差异** |
| L5 节点间 | 800 Gbps ConnectX-8 | 400 Gbps ConnectX-7 | 400 Gbps ConnectX-7 | 2× |

## 关键洞察

1. **NVL72 把 NVLink 从"节点内"扩展到"机柜级"**：72 GPU 单一 NVLink 域，消除了传统 8 卡边界外的 IB 带宽悬崖。
2. **NVLink-C2C 打通了"显存墙 ↔ 系统内存"之间的护城河**：900 GB/s 让 CPU DRAM 事实上可被 GPU 利用。
3. **每跨一层级，带宽下降 4-20×**：软件栈设计目标就是最大化高带宽层的利用率，避免频繁跌落。
4. **NVL72 硬件恰好是 DWDP、MegaScale-Infer 等论文的必要前提**。

## 关键概念

- [[GB200-NVL72]]
- [[DGX-H100]]
- [[DGX-B200]]
- [[NVLink-5.0]]
- [[NVLink-C2C]]
- [[NVSwitch]]
- [[HBM3e]]
- [[Grace-CPU]]
- [[ConnectX-SuperNIC]]

## 与其他素材的关联

- 是 [[DWDP]] 的硬件基础文档——DWDP 只能在 NVL72 这种 1.8 TB/s per GPU 的平台上成立。
- 与 [[长上下文&超节点调研]] 的附录带宽表完全对应，可视为其详细扩展。
- 为 [[SuperOffload]] 提供硬件上下文——GH200 Superchip 的 Hopper:Grace FLOPS 比 ~330 的观察与此文中 L3 层的 C2C 直连相呼应。

## 原文精彩摘录

> 带宽悬崖：每跨越一个层级，带宽通常下降 4–20×，软件栈设计（数据并行 / 张量并行 / 流水线并行 / MoE 路由）的核心目标就是最大化高带宽层的利用率。

> NVL72 的关键突破：通过 18 块 NVSwitch 芯片将 72 GPU 组成单一 NVLink 域。

## 相关页面

- [[GB200-NVL72]]
- [[超节点架构]]
- [[NVLink-5.0]]
- [[NVLink-C2C]]
