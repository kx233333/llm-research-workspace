---
tags: [素材摘要, 推理系统, 长上下文, PD分离, 跨数据中心, 异构部署]
created: 2026-04-29
updated: 2026-04-29
source_type: 论文
source_path: https://arxiv.org/abs/2604.15039
paper_arxiv: "2604.15039"
paper_authors: [Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang]
paper_venue: arXiv 2026
---

# Prefill-as-a-Service: 跨数据中心的 KVCache 传输

> Hybrid Attention 模型将 KVCache 缩小 10-36×，使得跨数据中心的 Prefill-Decode 分离成为可能。PrfaaS 通过选择性卸载 + 带宽感知调度，在 1T 模型上实现 **54% 吞吐提升** 和 **64% P90 TTFT 降低**。

## 基本信息

- **来源类型**：arXiv 论文（2604.15039）
- **作者**：Ruoyu Qin, Weiran He 等（清华大学）
- **提交日期**：2026-04-16
- **篇幅**：16 页，5 图，6 表

---

## 核心问题

### 传统 PD 分离的瓶颈

Prefill-Decode 分离（如 [[分离部署|Splitwise/DistServe]]）要求 Prefill 节点和 Decode 节点在**同一 RDMA 网络**内，因为 Dense Attention 模型产生的 KVCache 传输量巨大：

| 模型 | 参数量 | 32K tokens 时的 KV 吞吐 |
|------|--------|----------------------|
| MiniMax-M2.5（Dense） | - | **~60 Gbps** |
| MiMo-V2-Flash（Hybrid） | 309B | **4.66 Gbps** |

Dense 模型的 60 Gbps KV 吞吐远超普通以太网（100 Gbps 链路实际可用 ~50-80 Gbps），导致：
- **必须同机房 RDMA 互联** → 硬件选型受限
- **无法异构部署** → 高算力 GPU（H200）和高带宽 GPU（H20）不能协同
- **弹性扩展受限** → 无法跨数据中心调度 burst 流量

### 关键洞察

**下一代 Hybrid Attention 模型**（线性注意力 + 少量全注意力）将 KVCache 缩小 **10-36×**，使得普通以太网传输成为可能：

```
Dense Model:    所有层都产生 O(seq_len) 的 KVCache → 总量巨大
Hybrid Model:   大部分层是线性注意力（固定大小 state）
                少数层是全注意力（O(seq_len) KVCache）
                → 总量缩小 10-36×
```

---

## Hybrid Attention 模型谱系

| 模型 | 参数量 | 线性层:全注意力层 | 线性层类型 | 全注意力类型 | KV 缩减 |
|------|--------|------------------|-----------|------------|---------|
| **Kimi Linear** | 48B | 3:1 | KDA | [[MLA]] | ~4× |
| **MiMo-V2-Flash** | 309B | 5:1 | SWA | GQA | ~13× |
| **Ring-2.5-1T** | 1T | 7:1 | Lightning | MLA | ~36× |
| **Qwen3.5-397B** | 397B | 3:1 | GDN | GQA | ~4× |

> **核心特性**：线性注意力层产生**固定大小的 recurrent state**（与序列长度无关），只有少数全注意力层产生 O(seq_len) 的 KVCache。

---

## PrfaaS 系统架构

### 部署拓扑

```
┌────────────────────────────────────────────────────────┐
│                    PrfaaS 集群                          │
│          (算力密集型，如 H200)                           │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│   │ Prefill  │  │ Prefill  │  │ Prefill  │            │
│   │ Instance │  │ Instance │  │ Instance │            │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│        └──────────────┼──────────────┘                  │
│                       │ KVCache (小体积)                 │
│                       ↓                                 │
│              ╔═══════════════╗                          │
│              ║ 100 Gbps 以太网 ║  ← 仅用 ~13 Gbps (13%) │
│              ╚═══════════════╝                          │
│                       │                                 │
├───────────────────────┼────────────────────────────────┤
│                    Local PD 集群                        │
│          (带宽密集型，如 H20)                            │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│   │  PD-P    │  │  PD-P    │  │  PD-D    │            │
│   │(短 Prefill│  │(短 Prefill│  │ (Decode) │            │
│   │ + Decode)│  │ + Decode)│  │          │            │
│   └──────────┘  └──────────┘  └──────────┘            │
└────────────────────────────────────────────────────────┘
```

### 选择性卸载（Selective Offloading）

不是所有请求都跨数据中心——**只卸载长序列的 Prefill**：

```
请求到达 → 增量 Prefill 长度 L
  │
  ├─ L > 阈值 t (长序列) → 发送到 PrfaaS 集群
  │                        → Prefill 完成后，KVCache 传回 Local PD
  │                        → Local PD 执行 Decode
  │
  └─ L ≤ 阈值 t (短序列) → Local PD 自行 Prefill + Decode
```

**为什么要选择性？** 短序列的 Prefill 很快，跨数据中心传输反而增加延迟。只有长序列才值得卸载。

### Hybrid Prefix Cache Pool

管理两种不同类型的 KVCache：

| 类型 | 特点 | 复用方式 | 管理粒度 |
|------|------|---------|---------|
| **线性 state** | 固定大小，与 seq_len 无关 | 精确匹配（request-level） | 整体复用 |
| **全注意力 KV** | O(seq_len)，序列相关 | 前缀匹配（block-level） | 部分复用 |

两种 cache 使用**统一的 block pool**，分别管理但共享物理内存。

---

## 数学模型

### KV 吞吐（单请求）

$$\Phi_{kv}(l) = \frac{S_{kv}(l)}{T_{prefill}(l)}$$

- $S_{kv}(l)$：序列长度 $l$ 的 KVCache 大小
- $T_{prefill}(l)$：Prefill 延迟

### 系统吞吐建模

$$\Lambda_{max} = \min\left(\frac{\Theta_{prfaas}}{p}, \frac{\Theta_{pd\text{-}p}}{1-p}, \Theta_{pd\text{-}d}\right)$$

其中：
- $\Theta_{prfaas} = \min\left(\frac{N_{prfaas}}{T_{prefill}(l_{long})}, \frac{B_{out}}{S_{kv}(l_{long})}\right)$（PrfaaS 吞吐 = min(算力, 出口带宽)）
- $p = P(L > t)$：路由到 PrfaaS 的请求比例
- 最优条件：三个阶段恰好平衡

### 最优化

通过 grid search 同时优化：
1. **路由阈值 $t$**：决定哪些请求卸载
2. **Prefill/Decode 比例 $N_p/N_d$**：Local PD 集群内的资源分配

---

## 双时间尺度调度

### 短期（秒级）

- 监控 PrfaaS 出口带宽利用率和队列深度
- **带宽受限时**：提高阈值 $t$，减少卸载
- **算力受限时**：考虑跨集群传输 prefix cache 以提高复用率

### 长期（分钟/小时级）

- 识别瓶颈阶段（通过队列深度）
- 重新分配 Local PD 的 Prefill/Decode 资源比例
- 相应调整 $t$

---

## 实验结果

### Case Study 配置

| 参数 | 值 |
|------|-----|
| **模型** | 1T 参数 Hybrid（Kimi Linear 架构，3:1 KDA:MLA） |
| **PrfaaS 集群** | 32× H200 |
| **Local PD 集群** | 64× H20 |
| **集群间网络** | 100 Gbps 以太网 |
| **请求分布** | 截断对数正态（μ=9.90, σ=1.00, [128, 128K]），均值 ~27K tokens |
| **输出长度** | 1024 tokens |
| **SLO** | 40 tokens/sec |

### 模型 KVCache Profiling

| 序列长度 | KVCache 大小 | Prefill 延迟 | KV 吞吐 |
|----------|-------------|-------------|---------|
| 1K | 190.8 MiB | 0.44s | 3.61 Gbps |
| 8K | 308.9 MiB | 0.72s | 3.59 Gbps |
| 32K | 701.3 MiB | 1.84s | 3.19 Gbps |
| 128K | 2.3 GiB | 7.40s | 2.62 Gbps |

> 对比 Dense 模型 32K 时 ~60 Gbps，Hybrid 模型仅 ~3 Gbps，**降低 ~20×**。

### 核心性能对比

| 指标 | PrfaaS-PD | 同构 PD | 朴素异构 |
|------|-----------|---------|---------|
| 路由阈值 $t$ | 19.4K | — | — |
| 最大吞吐 (req/s) | **3.24** | 2.11 | 2.45 |
| **vs 同构 PD** | **+54%** | baseline | +16% |
| P90 TTFT (s) | **3.51** | 9.73 | 3.51 |
| **vs 同构 PD** | **-64%** | baseline | -64% |
| 跨集群出口带宽 | **~13 Gbps** (13%) | — | — |

### 关键数据点

- **49.6%** 的请求被路由到 PrfaaS（长度 > 19.4K 的请求）
- 平均卸载长度 ~44K tokens
- 100 Gbps 链路仅用 **13%**，有巨大安全余量

---

## 核心贡献总结

1. **KVCache 缩减是必要但不充分条件**：Hybrid 模型缩小 KV 使跨数据中心传输可行，但没有选择性卸载和带宽感知调度就无法充分利用
2. **选择性卸载优于全量卸载**：只卸载长序列 Prefill，短序列本地处理，避免不必要的网络延迟
3. **双时间尺度调度**：短期适应 burst 流量，长期优化资源分配
4. **Hybrid Prefix Cache**：统一管理线性 state 和全注意力 KV 的不同复用模式

---

## 与其他素材的关联

- 与 [[分离部署]]：PrfaaS 是 PD 分离的跨数据中心进化版。传统 PD 分离（Splitwise/DistServe）要求 RDMA 互联，PrfaaS 利用 Hybrid 模型突破带宽限制。
- 与 [[KV-Cache]]：直接利用 Hybrid Attention 模型的 KVCache 缩减特性。MLA（[[MLA]]）、SWA、线性注意力等是使 PrfaaS 可行的前提。
- 与 [[CAD-DistCA]]：DistCA 在训练中解耦 Attention，PrfaaS 在推理中解耦 Prefill/Decode。都是"解耦"思想在不同阶段的体现。
- 与 [[MoE 推理]]：都面临异构硬件部署问题。MoE 推理的 Expert Parallelism 也需要跨节点通信优化。
- 与 [[长上下文训练]]：训练侧的长上下文问题（如 Dynamic CP）与推理侧的长上下文 Prefill 卸载是同一问题的两面。
- 与 [[MegaScale-Infer]]：都是推理系统级优化，但 PrfaaS 特别关注跨数据中心部署。

## 原文精彩摘录

> "KVCache-efficient model architectures are necessary but not sufficient — system-level selective offloading and bandwidth-aware scheduling are equally essential for viability."

> "MiMo-V2-Flash achieves 4.66 Gbps versus 59.93 Gbps for MiniMax-M2.5, a 13× reduction."

> "Aggregate PrfaaS egress: ~13 Gbps, consuming only 13% of 100 Gbps link."
