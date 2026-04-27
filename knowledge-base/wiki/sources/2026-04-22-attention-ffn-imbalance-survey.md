---
tags: [素材摘要, 综述, Attention, FFN, 解耦, 并行策略]
created: 2026-04-22
updated: 2026-04-22
source_type: 本地笔记
source_path: raw/notes/Attention-FFN计算量不平衡与解耦调研.md
---

# Attention vs FFN 计算量不平衡：估算模型、解耦方案与显存分析

> 系统性梳理 Attention（$s^2 h$）与 FFN（$sh^2$）在不同场景下的计算量估算方法，对比 5 种并行架构的平衡策略，总结是否值得显式解耦以及解耦后的显存分布。

## 基本信息

- **消化日期**：2026-04-22
- **覆盖方案**：传统 PP、[[ChunkFlow]]、[[WLB-LLM]]、[[MegaScale-Infer]]、[[AFD-Ratio 理论]]
- **覆盖场景**：训练（prefill-like）+ 推理（prefill + decode）

## 核心观点

1. **Attention 与 FFN 交叉点**：$s^* = 2h \approx 3h$（取决于 FFN 系数）。$h=4096$ 时 $s^* \approx 8–12K$，这是流水线按层均分无法平衡的根源。
2. **Decode 与 Prefill 的本质差异**：Decode Attention 是 memory-bound（KV cache），FFN 是 compute-bound，瓶颈完全不同 → decode 场景 FLOPs 估算会严重低估 Attention 延迟。
3. **解耦的核心收益不是 FLOPs 节省，而是硬件异构匹配**：高带宽 GPU 承接 Attention，高 TFLOPS GPU 承接 FFN。
4. **FlashAttention 把训练激活显存从 $O(s^2)$ 降至 $O(s)$**，使得训练侧 Attn/FFN 显存不平衡大幅缓解，推理侧 KV cache 仍是专属难题。
5. **AFD 理论（arXiv 2601.21351）给出最优比例 $r^*$ 的闭合解**，实测 $r^* \approx 9.3$（DeepSeek-V3，$B=256$）。

## 关键概念

- [[Attention-FFN 解耦]]
- [[AFD-Ratio 理论]]
- [[MegaScale-Infer]]
- [[WLB-LLM]]
- [[ChunkFlow]]
- [[FlashAttention]]
- [[KV-Cache]]
- [[负载均衡与变长序列]]
- [[Attention-FFN 计算量不平衡]]

## 与其他素材的关联

- 是 [[长上下文&超节点调研]] 中 "Data 分配引起 worker 不均衡" 章节的定量扩展。
- 与 [[2026-04-22-dp-only-attention-ffn-survey]] 是同一问题的两个视角：前者横跨 TP/PP/CP/DP，后者只考虑 DP。
- 与 [[ChunkFlow]]、[[WLB-LLM]]、[[MegaScale-Infer]] 均有专章讨论。

## 原文精彩摘录

> Attention 与 FFN 相等时 $s = 2h$。对 $h=4096$，短序列（$s \ll 8K$）FFN 主导，长序列（$s \gg 8K$）Attention 以 $s^2$ 爆炸式主导。

> 解耦的核心收益不是 FLOPs 节省，而是硬件异构匹配。

## 相关页面

- [[Attention-FFN 计算量不平衡]]
- [[Attention-FFN 解耦]]
- [[并行策略与同步开销]]
- [[MegaScale-Infer]]
- [[AFD-Ratio 理论]]
