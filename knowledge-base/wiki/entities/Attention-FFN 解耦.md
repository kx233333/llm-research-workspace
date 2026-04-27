---
tags: [实体, 概念, 解耦, Attention, FFN]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-attention-ffn-imbalance-survey, 2026-04-22-dp-only-attention-ffn-survey]
---

# Attention-FFN 解耦

> 把 Transformer 层内的 Attention 模块和 FFN 模块物理/逻辑上分开调度，各自用独立的并行策略和硬件。

## 动机

二者计算特性不同：

| 维度 | Attention | FFN |
|------|-----------|-----|
| Prefill FLOPs | $O(s^2 h)$ | $O(s h^2)$ |
| Decode 瓶颈 | **Memory-bound**（KV cache）| **Compute-bound**（GEMM）|
| 理想硬件 | 高 HBM 带宽 | 高 TFLOPS |
| 对序列长度敏感 | 高（$s^2$）| 低（$s$）|

## 三种解耦路线

### 路线 1：推理侧 Attention / FFN 解耦（MegaScale-Infer 类）

- [[MegaScale-Infer]]：ping-pong pipeline + 异构 GPU
- [[AFD-Ratio 理论]]：闭合解 $r^*$

### 路线 2：训练侧核心注意力解耦（CAD 类）

- [[CAD-DistCA]]：核心注意力 $\text{softmax}(QK^T)V$ 从其余层抽出，独立调度到 attention server

### 路线 3：MoE Expert 解耦（DWDP 类）

- [[DWDP]]：不是 Attn/FFN 解耦，但思路相似——把 expert 权重从 rank 剥离，变成分布式资源

## 解耦的代价

- 新增通信（activation transfer）
- 工程复杂度（调度器、profiler、fused kernel）
- 最优配置依赖 profiling

## 不同素材中的观点

- [[2026-04-22-attention-ffn-imbalance-survey]]：**Decode 阶段强烈建议解耦**（Attn memory-bound vs FFN compute-bound 差异大）；Prefill 阶段可选解耦
- [[2026-04-22-dp-only-attention-ffn-survey]]：DP 规模 ≥ 8 时必须解耦（CAD Figure 4b）

## 相关页面

- [[MegaScale-Infer]]
- [[CAD-DistCA]]
- [[DWDP]]
- [[AFD-Ratio 理论]]
- [[Attention-FFN 计算量不平衡]]
- [[Core-Attention]]
