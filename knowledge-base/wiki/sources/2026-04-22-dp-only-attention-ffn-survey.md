---
tags: [素材摘要, 综述, DP, 负载均衡, 变长序列]
created: 2026-04-22
updated: 2026-04-22
source_type: 本地笔记
source_path: raw/notes/仅DP下的Attention-FFN计算量平衡调研.md
---

# 仅 DP 下的 Attention vs FFN 计算量平衡

> 假设只使用 DP（不考虑 TP/PP/CP），变长序列下 Attention 与 FFN 不对称时，主流方案如何估算负载？解耦二者能否更好平衡？显存分布差异多大？

## 基本信息

- **消化日期**：2026-04-22
- **覆盖方案**：Naïve DP、[[WLB-LLM]]、[[ByteScale]]、[[LobRA]]、[[CAD-DistCA]]
- **焦点**：**DP 的根本矛盾在于 $\sum l_i$ 和 $\sum l_i^2$ 是两个守恒量，无法同时平衡**

## 核心观点

1. **DP 的梯度 all-reduce 是硬性同步屏障**：任何一 rank 慢都会拖慢全局。
2. **两个守恒量的矛盾**：激活显存 $\propto \sum l_i$（FlashAttn），但 Attention FLOPs $\propto \sum l_i^2$。长尾分布下不可能同时平衡。
3. **临界长度 $s^* \approx 3h$**：$s > s^*$ 时 $s^2$ 项主导，必须专门处理。
4. **四类解决路径**：
   - 估算改进（[[WLB-LLM]] 的 $W_a(s) + W_l(s)$）
   - 破坏 micro-batch 等量约束（[[ByteScale]]）
   - 引入配置异构（[[LobRA]]）
   - **显式解耦核心注意力**（[[CAD-DistCA]]）
5. **CAD 是最彻底方案**：核心注意力无状态且 token-level composable → 可实现完美计算 + 显存平衡。
6. **显存其实很容易平衡，真正难的是计算**。

## 关键概念

- [[CAD-DistCA]]
- [[ByteScale]]
- [[WLB-LLM]]
- [[LobRA]]
- [[Core-Attention]]
- [[FlashAttention]]
- [[Attention-FFN 计算量不平衡]]
- [[负载均衡与变长序列]]

## 解耦 DP 规模判断（关键数据）

基于 CAD 论文实测：
- DP=4：bubble 19%（可以考虑解耦）
- DP=8：bubble **55%**（必须解耦）

## 与其他素材的关联

- 是 [[2026-04-22-attention-ffn-imbalance-survey]] 的 DP-only 子视角。
- 与 [[ChunkFlow]]、[[WLB-LLM]] 是同主题对比。
- 与 [[长上下文&超节点调研]] 中 "Data 分配 worker 不均衡" 章节衔接。

## 原文精彩摘录

> 显存按 token 数分配天然近乎平衡（FlashAttn 功劳），真正难的是计算平衡——这正是 CAD 的切入点：先承认显存平衡容易（按 token 数分），再单独解决计算平衡（把 $\sum l^2$ 项独立调度）。

## 相关页面

- [[Attention-FFN 计算量不平衡]]
- [[负载均衡与变长序列]]
- [[CAD-DistCA]]
- [[ByteScale]]
- [[WLB-LLM]]
