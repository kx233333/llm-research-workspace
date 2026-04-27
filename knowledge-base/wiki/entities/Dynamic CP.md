---
tags: [实体, 概念, 并行策略, Megatron-LM]
created: 2026-04-25
updated: 2026-04-25
aliases: [DCP, Dynamic Context Parallel, 动态上下文并行]
---

# Dynamic CP（动态上下文并行）

## 定义

Dynamic Context Parallel (DCP) 是 NVIDIA Megatron-LM 中的一种训练并行策略。与标准 [[CP-a2a|Context Parallel]] 对所有 micro-batch 使用**固定 CP 度**不同，DCP 根据每个 micro-batch 中样本的实际序列长度，**动态调整 CP group 的大小**。

## 核心动机

在 sequence packing 训练中，不同的 packed sample 序列长度差异很大：

```
标准 CP（CP=8）：
  Short sample (512 tokens) → 8 GPU 分摊 → 每 GPU 64 tokens → 通信开销 >> 计算
  Long sample (64K tokens)  → 8 GPU 分摊 → 每 GPU 8K tokens  → 通信合理

Dynamic CP：
  Short sample (512 tokens) → 1 GPU 独立处理 → 零通信开销
  Long sample (64K tokens)  → 8 GPU 分摊     → 通信合理
```

## 工作原理

### 1. Group Size 决策

```python
def dcp_gpus_needed(seq_len, max_seq_len_per_rank, min_cp_size=1):
    raw = max(1, 2 ** ceil(log2(seq_len / max_seq_len_per_rank)))
    return max(min_cp_size, raw)
```

- 根据序列长度与 `max_seq_len_per_rank` 的比值确定需要多少 GPU
- 向上取到 **2 的幂次**（匹配预创建的 process group）
- 不低于 `min_cp_size`

### 2. 负载均衡调度

使用贪心算法将子样本分配到 DPxCP 的所有 GPU 上：

1. **分桶**：按 CP size 将样本分成等工作量的桶
2. **贪心分配**：长序列优先，分配到负载最低的 GPU group
3. **空 GPU 填充**：递归扩大最小 group 的 CP size
4. **负载估算**：$W(s) = s^2 / cp\_size$（注意力复杂度的简化估算）

### 3. Dynamic Process Groups

在初始化阶段，为 DPxCP 的每个 2 的幂次大小预创建 process group：

```python
# 例：DPxCP = 8，min_cp_size = 1
# 预创建：size=1, size=2, size=4, size=8 的 groups
group_sizes = [2**i for i in range(int(log2(N))) if 2**i >= min_cp_size]
```

每个 micro-batch 根据其 `local_cp_size` 从预创建的 groups 中选取对应的 group。

## 版本演进

| 版本         | PR                                                       | 特点                                           |
| ---------- | -------------------------------------------------------- | -------------------------------------------- |
| Part 1     | [#1803](https://github.com/NVIDIA/Megatron-LM/pull/1803) | 独立调度器 + 独立训练循环，不支持 PP/VPP                    |
| **Part 2** | [#2000](https://github.com/NVIDIA/Megatron-LM/pull/2000) | 融入标准 pipeline scheduling，支持 PP/VPP/Mamba/MTP |

详见 → [[2026-04-25-megatron-dynamic-cp-pr2000|PR #2000 代码分析]]

## 使用方式

```bash
# 基本用法
--dynamic-context-parallel
--max-seqlen-per-dp-cp-rank 4096
# 自动设置 --sequence-packing-scheduler default_dynamic_cp

# 可选：设置最小 CP group size
--min-dynamic-context-parallel-size 2
```

## 约束条件

- CP group size 必须是 2 的幂次
- 不支持 CUDA Graphs
- 不支持 FSDP + PP
- 推荐使用 [[FlashAttention]]（cuDNN 有重编译开销）

## 关联概念

- [[CP-a2a]] —— 标准 Context Parallel
- [[USP]] —— Hybrid CP（Ulysses + Ring）
- [[负载均衡与变长序列]]
- [[长上下文训练]]
- [[CAD-DistCA]] —— 正交的 Attention 解耦方案
