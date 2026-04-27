---
tags: [实体, 方案, 训练系统, Superchip, Offload]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-long-context-supernode-survey]
---

# SuperOffload

> **ASPLOS 2026**（UIUC SSAIL Lab）：针对 GH200 Superchip 重新设计 offload 策略。指出过去 offloading 基于 PCIe 松耦合旧前提，在 C2C 紧耦合场景下需重新审视。

## 关键信息

- **发表**：ASPLOS 2026
- **来源组织**：UIUC SSAIL Lab
- **硬件目标**：[[GH200-Superchip]]

## 核心观察

1. **Hopper:Grace FLOPS 比 ~330**，远大于 DGX-2（60.39）和 DGX-A100（135.65）
2. **通信/计算分布的经验需重写**：实验结果显示"在 GPU 侧 cast 再搬 FP32" 比"CPU 侧 cast 再搬 FP16"更快——Grace 侧临时内存路径不友好
3. **C2C 直连让 CPU DRAM 事实上成为 GPU 的 L3 存储**

## 核心机制（STV）

- Superchip-aware offloading
- Type casting 位置重设计
- Bucket repartitioning
- CPU-GPU overlap（optimizer step 与 backward 重叠）

## 实验结果

- 相比现有 offloading：**2.5× 吞吐提升**
- GH200 上达到 **55% MFU**

## 核心洞察（docx 评论）

> CPU-GPU 的高速连接本质上可以视为管理一个多层级内存吗？

这是 [[多层级内存管理]] 方向的种子问题。<!-- confidence: INFERRED -->

## 相关页面

- [[GH200-Superchip]]
- [[Grace-CPU]]
- [[NVLink-C2C]]
- [[显存优化]]
- [[超节点架构]]
