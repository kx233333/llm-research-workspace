---
title: "MPress: Democratizing Billion-Scale Model Training on Multi-GPU Servers via Memory-Saving Inter-Operator Parallelism"
authors:
  - Quan Zhou
  - Haiquan Wang
  - Xiaoyan Yu
  - Cheng Li
  - Youhui Bai
  - Feng Yan
  - Yinlong Xu
date: 2023-01-01
venue: "HPCA 2023"
doi: "10.1109/HPCA56546.2023.10071077"
url: "https://ieeexplore.ieee.org/document/10071077"
tags:
  - LLM训练
  - 分布式训练
  - Pipeline并行
  - 显存优化
  - NVLink
  - D2D交换
  - 重计算
  - BERT
  - GPT
status: 已读
---

# MPress：单节点多GPU亿级参数模型训练

## 核心问题

训练十亿参数规模的大模型（BERT、GPT 等）受制于 **GPU 显存墙**。现有方案各有明显局限：

| 方案 | 问题 |
|------|------|
| GPU→CPU Swap | PCIe 带宽低，传输开销大 |
| 激活重计算 | 额外计算量大，吞吐下降显著 |
| ZeRO（数据并行） | all-reduce 通信量高，仍有大量冗余显存 |

根本矛盾：**显存受限导致单机可训练的模型规模远低于亿级**，而扩容到多机引入更高通信成本。

---

## 核心思路：MPress

> **关键创新**：以**算子间并行（流水线并行）**为基础，叠加 **D2D NVLink 交换**，将流水线各阶段的闲置显存变为其他阶段的"虚拟显存扩展"，同时避免昂贵的 PCIe 传输。

### 整体架构

```
模型按层划分到各 GPU（流水线并行）
        ↓
各阶段同一时刻存在"活跃"与"闲置"GPU
        ↓
D2D 交换（NVLink）：
  ├── 活跃阶段显存不足时 → 将 tensor 推送到闲置阶段 GPU
  └── 需要使用时再拉回，全程走 NVLink 高带宽
        ↓
辅助手段：选择性重计算 + CPU 异构计算（必要时）
```

---

## 主要技术设计

### 1. 算子间并行（Pipeline Parallelism）

- 将模型的层按顺序分配到不同 GPU（每个 GPU 只持有部分层）
- 与 ZeRO 相比：显存冗余更低，cross-GPU 通信量更小
- 对 Transformer 类深层顺序结构（BERT、GPT）天然友好
- 可与已有系统（PipeDream、DAPPLE）无缝集成

### 2. D2D 交换（Device-to-Device Swap via NVLink）

**核心洞察**：流水线并行中各阶段并非同时全满——当前活跃阶段显存压力大，而其他阶段 GPU 处于相对空闲状态，其显存可临时借用。

| 对比维度 | CPU Swap（PCIe） | D2D Swap（NVLink） |
|----------|----------------|-------------------|
| 带宽 | ~16 GB/s | ~600 GB/s（NVLink 3.0） |
| 延迟 | 高 | 极低 |
| 显存来源 | CPU DRAM | 同节点其他 GPU |

操作流程：
1. 识别当前流水线中**闲置阶段**的 GPU 空余显存
2. 将活跃阶段的中间激活值通过 NVLink 异步拷贝过去
3. 反向传播需要时再拉回，整个过程与计算尽量重叠

### 3. 整体显存优化策略（三级联合）

MPress 动态联合调度以下三种机制，选取最优组合：

1. **流水线并行**：减少各 GPU 持有层数，降低权重冗余
2. **选择性重计算**：对显存占用大但重算便宜的激活值不保存，按需重算
3. **D2D NVLink 交换**：将确实需要保留但当前 GPU 放不下的 tensor 迁移到闲置 GPU

三者相互补充，覆盖不同显存压力程度的场景。

---

## 实验结果

**实验配置**：NVIDIA DGX-1（8× V100）和 DGX-2（16× V100）· BERT / GPT 亿级参数模型 · 对比 ZeRO-1/2/3、纯重计算、CPU Swap

### 训练吞吐（vs ZeRO）

| 场景 | MPress 相对 ZeRO 的加速比 |
|------|--------------------------|
| 相同显存约束下 | **1.7× – 2.3×** |

### 可训练模型规模

- 在 DGX-1/DGX-2 上支持训练的模型参数量**显著超过**纯重计算和 ZeRO 的上限
- D2D 交换有效拓宽了单节点可容纳的最大参数规模

### 扩展性

- DGX-1（8 GPU）→ DGX-2（16 GPU）呈**近线性扩展效率**
- GPU 利用率更高，流水线气泡对吞吐的影响被显存收益覆盖

---

## 关键结论

1. **D2D NVLink 交换是关键突破**：利用 NVLink 高带宽将闲置 GPU 显存变为"扩展池"，突破 PCIe 带宽瓶颈
2. **联合优化优于单一策略**：三级显存优化（流水线并行 + 重计算 + D2D 交换）组合效果远优于各自单独使用
3. **算子间并行天然适配显存优化**：与 ZeRO 的 all-reduce 相比，流水线并行的通信量更可控，是构建显存优化的更好基础
4. **无需跨节点**：整套方案在单节点多 GPU 服务器上即可支持亿级参数训练，降低了基础设施门槛

---

## 适用场景

- 单节点多 GPU 服务器上训练十亿参数规模模型
- 显存受限但 NVLink 互联可用的环境
- 需要在吞吐与显存之间取得平衡的场景（优于纯重计算或纯 CPU Swap）

---

## 相关工作

- [[HiPress]]（SOSP 2021）：同系列工作，解决数据并行通信墙问题
- [[PipeDream]] / [[DAPPLE]]：MPress 可集成增强的已有流水线并行框架
- [[ZeRO]] / [[DeepSpeed]]：主要对比基线
- [[ChunkFlow]]：同为流水线并行 + 显存优化方向，专注于长上下文 SFT 场景
