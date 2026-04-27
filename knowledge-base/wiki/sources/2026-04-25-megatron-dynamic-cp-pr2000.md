---
tags: [素材摘要, 训练系统, 长上下文, 动态并行, 代码分析]
created: 2026-04-25
updated: 2026-04-25
source_type: GitHub PR
source_path: https://github.com/NVIDIA/Megatron-LM/pull/2000
pr_author: xiaoyao0115
pr_merged: 2026-04-07
pr_branch: dev
---

# Megatron-LM PR #2000：Dynamic CP Part 2 — 代码分析

> 将 Dynamic Context Parallel 从独立调度器重构为标准 pipeline scheduling 路径，实现端到端 DCP 支持（含 PP、VPP、Mamba、MTP），用户只需指定 `--sequence-packing-scheduler default_dynamic_cp`。

## 基本信息

- **来源类型**：GitHub PR（NVIDIA/Megatron-LM #2000）
- **作者**：xiaoyao0115
- **合入日期**：2026-04-07（dev 分支）
- **规模**：22 文件变更，+759 / -968 行（净减 209 行）
- **验证**：Qwen3-30B-A3B，32 GPU，收敛性对齐传统 CP 与 packed sequences

## 核心设计思想

### 问题：Part 1 的 DCP 走"旁路"

Part 1（PR #1803）实现了 DCP 的核心调度算法（`BalancedCPScheduler`），但它**绕过了标准的 pipeline scheduling**——通过 `DynamicCPDataLoaderWrapper` 包装 data iterator，再用独立的 `dynamic_context_parallel_forward_backward` 函数执行前向/反向。这意味着：

- 不支持 Pipeline Parallelism（PP）和 Virtual Pipeline Parallelism（VPP）
- 不支持 [[Mamba]] SSM 模型和 Multi-Token Prediction（MTP）
- 需要专用的 `DynamicCPMegatronPretrainingSampler` 采样器
- 维护两套独立的训练循环

### 解决方案：统一进标准路径

Part 2 将 DCP 调度逻辑**融入已有的 `DpBalancedScheduler` 继承体系**，让 DCP 走和普通 sequence packing 完全相同的 pipeline scheduling 路径。

```
Part 1 架构（已废弃）：
  DataLoader → DynamicCPDataLoaderWrapper → dynamic_context_parallel_forward_backward
                     ↑ 独立调度器                    ↑ 独立训练循环

Part 2 架构（当前）：
  DataLoader → wrap_data_iterator(scheduler=default_dynamic_cp)
             → 标准 forward_backward_no_pipelining / interleaved_pipelining
```

## 文件变更总览

### 🔴 删除的模块

| 模块 | 文件 | 说明 |
|------|------|------|
| `BalancedCPScheduler` 类 | `pipeline_parallel/dynamic_cp_schedule.py` | 660 行，整个文件删除 |
| `DynamicCPDataLoaderWrapper` 类 | `datasets/data_schedule.py` | 从 data_schedule.py 中移除 |
| `DynamicCPMegatronPretrainingSampler` 类 | `training/datasets/data_samplers.py` | 专用采样器不再需要 |
| `get_batch_on_this_dynamic_cp_rank` 函数 | `core/utils.py` | 51 行，独立的 batch 处理函数 |
| `dynamic_context_parallel_forward_backward` 函数 | `pipeline_parallel/dynamic_cp_schedule.py` | 独立训练循环 |

### 🟢 新增的模块

| 模块 | 文件 | 说明 |
|------|------|------|
| `DefaultDynamicCPScheduler` 类 | `datasets/data_schedule.py` | 继承 `DpBalancedScheduler`，57 行 |
| `next_hdp_group()` | `datasets/data_schedule_utils.py` | 独立函数版调度算法 |
| `align_sample_id_groups()` | `datasets/data_schedule_utils.py` | VPP micro-batch 对齐 |
| `dcp_gpus_needed()` | `datasets/data_schedule_utils.py` | GPU 需求计算 |
| `dcp_get_total_workload()` | `datasets/data_schedule_utils.py` | 负载估算 |
| `dcp_make_buckets_equal()` | `datasets/data_schedule_utils.py` | 等工作量分桶 |
| 测试配置 | `gpt3_mcore_te_tp2_pp1_cp4_dcp/` | 功能测试 YAML |

### 🟡 修改的模块

| 文件 | 关键变更 |
|------|---------|
| `model_parallel_config.py` | 新增 `min_dynamic_context_parallel_size` 参数；DCP 自动设置 `default_dynamic_cp` 调度器 |
| `parallel_state.py` | `create_dynamic_dp_cp_groups()` 支持 `min_cp_size` 过滤；`destroy_model_parallel()` 清理 DCP groups |
| `transformer_engine.py` | 修复 `TEDotProductAttention.cp_stream` 懒初始化 |
| `attention.py` | DCP forward 后恢复原始 `cp_group` |
| `mamba_mixer.py` / `mamba_context_parallel.py` | 添加 `set_context_parallel_group()`，支持动态 CP group 切换 |
| `multi_token_prediction.py` | MTP 层前向时临时切换 CP group |
| `transformer_config.py` | `supported_schedulers` 列表添加 `default_dynamic_cp` |
| `arguments.py` | DCP 参数校验；移除 PP 不兼容断言 |
| `training.py` | 删除 `DynamicCPDataLoaderWrapper` 包装；修复 eval 中 `num_microbatches` 变量名 |
| `pretrain_gpt.py` | `get_batch()` 传递 `dynamic_cp` 参数；简化 batch 处理分支 |

---

## 核心代码解析

### 1. `DefaultDynamicCPScheduler`：新的调度入口

```python
class DefaultDynamicCPScheduler(DpBalancedScheduler):
    def __init__(self, *args, min_cp_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_dynamic_cp = True          # 标记走 DCP 路径
        self.max_seq_len_per_rank = self.max_seqlen_per_dp_cp_rank
        self.total_hdp_gpus = self.dp_size * self.cp_size
        self.min_cp_size = min_cp_size

    def get_groups_and_subsamples(self, sample_id_seqlens):
        # 构造三个核心函数的闭包
        workload_fn = lambda seq_len, cp_size=None: dcp_get_total_workload(...)
        gpus_fn = lambda seq_len: dcp_gpus_needed(...)
        buckets_fn = lambda sample_seqlens, compute_est: dcp_make_buckets_equal(...)

        # 贪心循环：每轮构建一个 micro-batch
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)
        while sample_id_seqlens:
            mb, sample_id_seqlens, exec_times, sample_ids = next_hdp_group(...)
            groups.append(mb)
            sample_id_groups.append(sample_ids)

        # VPP 对齐
        if self.microbatch_group_size_per_vp_stage > 1:
            sample_id_groups = align_sample_id_groups(...)
        return sample_id_groups
```

**关键设计**：继承 `DpBalancedScheduler`，复用其 `run()` 方法中的完整数据流水线（Step 1~9），仅重写 `get_groups_and_subsamples()` 来替换调度策略。通过 `self.is_dynamic_cp = True` 在各处触发 DCP 特定逻辑。

### 2. `next_hdp_group()`：核心调度算法

这是从原 `BalancedCPScheduler.next_hdp_group()` 提取的独立函数版本，核心逻辑不变：

```
输入：(sample_id, seq_len) 列表（按序列长度降序排列）
输出：一个 micro-batch 的分配方案 + 剩余未分配的样本

算法：
1. 分桶：dcp_make_buckets_equal() 将样本按 CP size 分成 k 个等工作量桶
2. 贪心分配：
   a. 从桶中取出样本，计算需要的 GPU 数（2 的幂）
   b. 优先放入已有的同尺寸 group（负载最低的）
   c. 否则从空闲 GPU 中创建新 group
   d. 当负载差异 < delta(5%) 且 CP size 减小时，提前退出
3. 空 GPU 填充：递归扩大最小 group 的 CP size，直到所有 GPU 有工作
4. 返回分配方案和剩余样本
```

### 3. `local_cp_size` 传播路径

DCP 的核心特性是每个 micro-batch 的 CP group size 可能不同。`local_cp_size` 在数据管线中的传播：

```
build_packed_microbatches()
  → 从 sample_id_groups 推断每个 micro-batch 的 local_cp_size
  → _pack_sequences() 将其存入 sample dict

broadcast_to_pp_group()
  → 将 local_cp_size 广播到 PP group 的非首末 stage

create_data_iterator()
  → VPP metadata 中包含 local_cp_size

get_batch_on_this_rank_for_sequence_packing()
  → 从 batch 中读取 local_cp_size
  → 调用 parallel_state.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
  → 将 cp_group 写入 PackedSeqParams

forward() 各层
  → TEDotProductAttention / Mamba / MTP 从 packed_seq_params 读取 cp_group
  → 执行注意力计算
  → 恢复原始 cp_group
```

### 4. `min_dynamic_context_parallel_size` 参数

新增参数控制 DCP group 的最小尺寸：

```python
# parallel_state.py
def create_dynamic_dp_cp_groups(rank, ranks, pg_options, min_cp_size=1):
    group_sizes = [2**i for i in range(int(log2(len(ranks)))) if 2**i >= min_cp_size]
```

- 默认 `min_cp_size=1`：允许单 GPU 独立处理短序列（无 CP 通信）
- 设置更大值可减少 process group 数量，降低初始化开销

### 5. CP group 的 save/restore 模式

DCP 的每个 micro-batch 使用不同的 CP group，但模型层持有初始化时的 CP group 引用。PR 统一采用 save/restore 模式：

```python
# attention.py
output, bias = self.linear_proj(context_layer)
self.pg_collection.cp = _orig_cp_group  # 恢复原始 cp_group
return output, bias

# mamba_mixer.py
_orig_cp_group = self.cp.cp_group
if packed_seq_params is not None and packed_seq_params.cp_group is not None:
    self.cp.set_context_parallel_group(packed_seq_params.cp_group)
# ... forward 计算 ...
self.cp.set_context_parallel_group(_orig_cp_group)  # 恢复

# multi_token_prediction.py
_orig_cp_group = self.cp_group
if packed_seq_params is not None and packed_seq_params.cp_group is not None:
    self.cp_group = packed_seq_params.cp_group
# ... forward 计算 ...
self.cp_group = _orig_cp_group  # 恢复
```

### 6. TE 懒初始化修复

```python
# transformer_engine.py
if TEDotProductAttention.cp_stream is None:
    TEDotProductAttention.cp_stream = torch.cuda.Stream()
```

DCP 场景下 `set_context_parallel_group()` 可能在 `cp_stream` 初始化前被调用，这里确保 stream 已创建。

---

## 架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    Part 1 架构（已废弃）                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DataLoader                                                     │
│    ↓ DynamicCPMegatronPretrainingSampler（拉全局 batch）          │
│  DynamicCPDataLoaderWrapper                                     │
│    ├─ get_global_seqlens()         # AllGather 序列长度          │
│    ├─ BalancedCPScheduler.get_groups_and_subsamples()            │
│    └─ reroute_samples_to_hdp_ranks()  # All-to-All 重路由       │
│    ↓                                                            │
│  dynamic_context_parallel_forward_backward（独立循环）            │
│    ├─ 逐样本 forward → get_batch_on_this_dynamic_cp_rank()       │
│    └─ 逐样本 backward                                           │
│                                                                 │
│  ❌ 不支持 PP / VPP / Mamba / MTP                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Part 2 架构（当前）                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DataLoader                                                     │
│    ↓ MegatronPretrainingSampler（标准采样器）                      │
│  wrap_data_iterator(scheduler='default_dynamic_cp')             │
│    ├─ DefaultDynamicCPScheduler.run()                            │
│    │   ├─ get_batch_and_global_seqlens()                        │
│    │   ├─ get_groups_and_subsamples()  → next_hdp_group()       │
│    │   ├─ reroute_samples_to_dcp_ranks()                        │
│    │   ├─ build_packed_microbatches() + local_cp_size            │
│    │   └─ broadcast_to_pp_group()                               │
│    ↓                                                            │
│  标准 forward_backward_no_pipelining / interleaved_pipelining    │
│    ├─ get_batch_on_this_rank_for_sequence_packing(dynamic_cp=T) │
│    │   └─ 从 batch 读取 local_cp_size → 获取对应 cp_group       │
│    ├─ 各层 forward 临时切换 cp_group                              │
│    └─ 各层 forward 后恢复原始 cp_group                            │
│                                                                 │
│  ✅ 支持 PP / VPP / Mamba / MTP                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 关键概念

- [[Dynamic CP]] —— 动态上下文并行，本 PR 的核心功能
- [[负载均衡与变长序列]] —— DCP 要解决的根本问题
- [[长上下文训练]] —— 应用场景
- [[FlashAttention]] —— 推荐的 attention backend（cuDNN 有重编译开销）
- [[USP]] —— Hybrid CP 的另一种实现方式

## 已知限制

- DCP group size 必须是 **2 的幂次**
- 不支持 **CUDA Graphs**
- 不支持 **FSDP + PP** 组合
- cuDNN attention 有重编译开销，推荐使用 FlashAttention
- `trim_overload()` 函数目前被注释，不同 rank 可能有不同 micro-batch 数的场景尚未处理

## 与其他素材的关联

- 与 [[CAD-DistCA]]：DCP 在数据调度层面解决变长序列的负载不均，DistCA 在计算调度层面将 Attention 解耦到专用 Server。两者正交，可以共存。
- 与 [[ChunkFlow]]：都解决变长序列的负载均衡问题。ChunkFlow 用 chunk 时间串行，DCP 用动态 CP group 空间并行。ChunkFlow 的 state-aware scheduling 更适合 SFT 场景，DCP 更适合 pretraining。
- 与 [[WLB-LLM]]：WLB-LLM 用 $\sum s_i^2$ 估算负载，DCP 的 `dcp_get_total_workload()` 同样使用 $s^2 / cp\_size$ 估算。两者的估算公式本质一致。
- 与 [[USP]]：USP (Hybrid CP) 在 CP 内部组合 Ulysses + Ring，DCP 在 DP×CP 维度动态调整 CP group size。两者维度不同，理论上可以结合。

## 原文精彩摘录

> Dynamic CP forms variable-sized CP groups from the DPxCP ranks dynamically.

> DCP group sizes restricted to powers of 2.

> Convergence testing was performed on Qwen3-30B-A3B across 32 GPUs, demonstrating equivalent performance between traditional CP, packed sequences, and dynamic CP approaches.
