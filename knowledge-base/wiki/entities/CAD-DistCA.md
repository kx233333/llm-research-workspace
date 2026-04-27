---
tags: [实体, 方案, 训练系统, 长上下文, 解耦]
created: 2026-04-22
updated: 2026-04-25
sources: [2026-04-22-dp-only-attention-ffn-survey, 2026-04-25-distca-repo-analysis]
---

# CAD-DistCA

> **C**ore **A**ttention **D**isaggregation / DistCA（arXiv: 2510.18121）：在训练侧把核心注意力 $\text{softmax}(QK^T)V$ **从其余层解耦**，独立调度到 attention server 池。**DP 场景下最彻底的解耦训练系统**。

## 关键信息

- **arXiv**：2510.18121
- **作者**：Yonghao Zhuang, Junda Chen, ..., Ion Stoica, Eric Xing, Hao Zhang
- **硬件**：512 × H200，512K 上下文
- **加速**：**1.35× 端到端** vs WLB-LLM

## 核心洞察

### 双守恒量矛盾

对 packed batch 文档长度 $\{l_i\}$：

$$
\text{FLOPs} = \alpha \sum l_i^2 + \beta \sum l_i
$$

$$
\text{Memory} = \gamma \sum l_i
$$

要同时平衡计算和显存，必须 $\sum l_i = \sum l_j'$ **AND** $\sum l_i^2 = \sum l_j'^2$，**变长场景几乎无解**。

### 两个性质使得解耦可行

1. **Stateless**：核心注意力无可训练参数 → 可自由搬
2. **Composable**：token-level 可任意切分/合并，FlashAttn kernel 对变长 fused 输入保持高 MFU（shard ≥ 128 tokens 时）

### In-Place Attention Server

每个 GPU 周期性切换角色（context-indep layer ↔ attention server），避免专用 attention pool 的 memory 浪费（FFN 占大头）。

### Communication-Aware Greedy Scheduling

$$
E = \frac{\Delta F_{\max}}{V_{\text{comm}}}
$$

- 每个 deficit server 从 surplus 寻找最高 $E$ 的 shard 做迁移
- 容差参数 $\epsilon$ 调节 balance vs communication 的权衡

### Ping-Pong Execution

Ping/Pong 两个 nano-batch 交替执行，通信与计算重叠。

## DP 规模敏感性（关键数据）

| DP 规模 | Variable-length chunking 的 idle 比例 |
|--------|----------------------------------|
| DP=4 | 19% |
| DP=8 | **55%** |

## 实验结果

- 3D parallelism（无 PP）：1.07–1.20×（Pretrain），1.05–1.12×（ProLong）
- 4D parallelism（有 PP）：1.15–1.30×（8B），1.10–1.25×（34B）

## 与其他方案的对比

| 对比 | 关系 |
|------|------|
| [[WLB-LLM]] | 主要基线；CAD 指出 WLB-LLM 在 DP 规模扩大时 memory divergence 加剧 |
| [[ChunkFlow]] | 都处理 $s^2$ 项；ChunkFlow 选 chunk 串行，CAD 选 token-level 调度 |
| [[DWDP]] | 都引入异步/解耦；DWDP 面向推理 MoE，CAD 面向训练 attention |
| [[MegaScale-Infer]] | 推理版本的类似思想，CAD 是训练版 |

## 代码仓库

- **GitHub**：[hao-ai-lab/DistCA](https://github.com/hao-ai-lab/DistCA)
- **代码量**：Python ~2000 行 + CUDA/C++ ~1000 行 + Megatron 集成 ~1000 行
- **核心依赖**：Megatron-LM core_v0.12.1 + NVSHMEM 3.2.5 + FlashAttention 2.7.4
- **硬件要求**：H100/H200（CUDA sm_90a），节点间需 InfiniBand
- **复现指南**：详见 [[2026-04-25-distca-repo-analysis|DistCA 仓库深度剖析]]

## 相关页面

- [[Core-Attention]]
- [[Attention-FFN 解耦]]
- [[负载均衡与变长序列]]
- [[FlashAttention]]
- [[WLB-LLM]]
- [[ChunkFlow]]
- [[Dynamic CP]] — Megatron-LM 内置的替代方案
- [[Ping-Pong Pipeline]]
