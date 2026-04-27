---
tags: [主题, Attention, FFN, 计算量]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-attention-ffn-imbalance-survey, 2026-04-22-dp-only-attention-ffn-survey, 2026-04-22-chunkflow, 2026-04-22-long-context-supernode-survey]
---

# Attention-FFN 计算量不平衡

> Transformer 层内 Attention 与 FFN 模块的计算量不对称（$s^2 h$ vs $sh^2$），是变长训练/推理的最根本矛盾。

## 核心公式

### Prefill FLOPs

$$
\text{FLOPs}_{\text{Attn}} \approx 4Bs h^2 + 4Bs^2 h
$$

$$
\text{FLOPs}_{\text{FFN}} \approx 8Bs h^2
$$

### 交叉点

$$
4s^2 h = 8s h^2 \implies s^* = 2h \approx 3h（含 SwiGLU 等系数）
$$

- $h=4096$ → $s^* \approx 8K–12K$
- $h=8192$（34B 级）→ $s^* \approx 16K–24K$

### 数值示例（$h=4096$，单层）

| $s$ | Attn FLOPs | FFN FLOPs | 比率 |
|-----|----------|----------|------|
| 512 | 0.004T | 0.134T | 0.03× FFN 主导 |
| 4K | 0.26T | 0.134T | 2.0× Attn 开始主导 |
| 32K | 16.7T | 0.134T | **125× Attn 压制** |

## Decode 阶段的本质翻转

| 特性 | Attention（Decode）| FFN（Decode）|
|------|------------------|-------------|
| 瓶颈 | **Memory-bound** | **Compute-bound** |
| 延迟 ∝ | KV cache 长度 $c$ | batch size |
| 与序列长度 | 线性 | 无关 |
| 理想硬件 | 高 HBM 带宽 | 高 TFLOPS |

→ FLOPs 估算在 decode 阶段**完全失效**，必须改用 memory bandwidth 模型。

## 解决路径（5 类）

| 类别 | 代表 | 处理方法 |
|------|------|---------|
| 估算改进 | [[WLB-LLM]] | 拟合 $W_a(s) + W_l(s)$ |
| 数据重组 | [[ChunkFlow]] | chunk 等长化 |
| 配置异构 | [[FlexSP]]、[[LobRA]] | 不同长度用不同并行度 |
| 破坏约束 | [[ByteScale]] | 打破 micro-batch 等量 |
| **显式解耦** | [[CAD-DistCA]]、[[MegaScale-Infer]]、[[AFD-Ratio 理论]] | 物理分离 Attn 和 FFN |

## 关键洞察

1. **训练 vs 推理的显存差异**：训练中 FlashAttn 让 Attn 激活从 $O(s^2)$ 降到 $O(s)$，**训练显存基本平衡**；推理中 KV Cache 是 Attention 独占的动态显存。
2. **解耦的真正收益不是 FLOPs 节省，而是硬件异构匹配**：高 BW GPU → Attention，高 TFLOPS GPU → FFN。
3. **DP 规模敏感**：CAD 实测 DP=8 时不均衡造成 bubble 55%，必须解耦。

## 素材汇总

| 素材 | 贡献角度 |
|------|--------|
| [[2026-04-22-attention-ffn-imbalance-survey]] | 综合调研（含 TP/PP/CP/DP） |
| [[2026-04-22-dp-only-attention-ffn-survey]] | DP-only 视角深化 |
| [[2026-04-22-chunkflow]] | 用 chunk 解决 $s^2$ |
| [[2026-04-22-long-context-supernode-survey]] | 问题陈述 |

## 相关页面

- [[Attention-FFN 解耦]]
- [[Core-Attention]]
- [[FlashAttention]]
- [[KV-Cache]]
- [[负载均衡与变长序列]]
