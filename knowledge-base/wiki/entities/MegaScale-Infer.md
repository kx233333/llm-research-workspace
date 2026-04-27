---
tags: [实体, 方案, 推理系统, MoE, Attention-FFN 解耦]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-attention-ffn-imbalance-survey]
---

# MegaScale-Infer

> ByteDance 的 MoE 推理系统（arXiv: 2504.02263）：在每个 MoE 层内将 Attention 与 Expert（FFN）**物理分离**，独立配置并行度和硬件，Ping-Pong Pipeline Parallelism 隐藏通信。

## 关键信息

- **发表**：2025-04，arXiv: 2504.02263
- **来源组织**：ByteDance Seed + 北京大学
- **目标模型**：MoE 推理（decoding 主）

## 核心机制

### Ping-Pong Pipeline

将 batch 分 $m$ 个 micro-batch，在 attention 和 expert 节点间交替流转：

```
constraint 1: T_a ≈ T_e（通过配置 n_a 实现）
constraint 2: T_c < T_f
constraint 3: m × T_f ≥ 2 × (T_f + T_c)
```

最小 micro-batch 数：$m \geq 2(1 + T_c/T_f)$

### 延迟模型（profiling 拟合）

- $T_a = k_1 b_a + k_2$（attention 节点）
- $T_e = k_3 b_e + k_4$（expert 节点）

平衡：$n_a = \frac{k_1 E}{k_3 K}$（E = 专家数，K = top-K）

### M2N 通信库

替代 NCCL：消除 GPU-CPU 拷贝、group init 开销、GPU 同步。

### 异构部署

| GPU | 用途 | per-cost 带宽 | per-cost TFLOPS |
|-----|------|-------------|---------------|
| H20 | Attention | **2214 GB/s/$** | 80 |
| L40S | Expert | 800 | **335 TFLOPS/$** |

## 实验结果

- per-GPU throughput **1.9×** vs SOTA
- per-cost throughput **1.7×**

## 与其他方案的对比

| 对比 | 关系 |
|------|------|
| [[DWDP]] | 同是 MoE 推理 a2a 优化，MegaScale-Infer 保留 a2a + ping-pong 隐藏，DWDP 直接消灭 a2a |
| [[AFD-Ratio 理论]] | MegaScale-Infer 是 AFD 的工程落地版本，AFD 理论给出 $r^*$ 闭合解 |

## 相关页面

- [[Attention-FFN 解耦]]
- [[AFD-Ratio 理论]]
- [[MoE-a2a]]
- [[Ping-Pong Pipeline]]
- [[分离部署]]
- [[DWDP]]
