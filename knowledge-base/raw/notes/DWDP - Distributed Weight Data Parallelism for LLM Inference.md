---
title: "xiDWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72"
authors:
  - Wanqian Li
  - Jintao Peng
  - Zongfei Jing
  - Tianyu Zhang
  - Ze Long
  - Xianjie Qiao
  - Xiaoming Chen
  - Dongxu Yang
  - Kefeng Duan
  - June Yang
date: 2026-04-02
arxiv: "2604.01621"
url: https://arxiv.org/abs/2604.01621
github: https://github.com/NVIDIA/TensorRT-LLM
tags:
  - LLM推理
  - 分布式推理
  - MoE
  - 数据并行
  - 权重卸载
  - NVLink
  - DeepSeek
  - TensorRT-LLM
  - GB200
  - NVL72
status: 已读
---

# DWDP：去除层间同步的分布式权重数据并行推理

## 核心问题

多 GPU 推理的主流并行策略（张量并行、专家并行、流水线并行）均需要在**每一层边界进行跨 rank 同步**（all-reduce / all-to-all）。这使得端到端吞吐对**负载不均衡**极为敏感：

| 不均衡来源 | 具体表现 |
|-----------|---------|
| 请求级不均衡 | 不同 rank 分到的序列长度、KV-cache 命中率不同 |
| 权重级不均衡 | MoE 路由偏斜导致各 rank 激活专家数差异悬殊 |

实测（DeepSeek-R1, GB200, DEP 策略）：当各 rank 序列长度变异系数达 20%（生产负载常见水平）时，**同步等待开销占总迭代时间约 12%**。现有调度优化（cache-aware / load-aware scheduling）无法从根本上消除同步需求。

---

## 核心思路：DWDP

> **关键创新**：保持各 rank **完全数据并行**执行，将 MoE 专家权重分散存储于同组 peer GPU，推理时**按需异步拉取**（P2P NVLink），彻底消除层间集体通信同步。

### 与 DEP 的对比

```
DEP（数据并行 + 专家并行）：
  每层结束 → all-to-all 同步 → 等待最慢 rank → 继续下一层

DWDP：
  各 rank 独立执行 → 异步预取下一层缺失专家权重
  → 不等其他 rank → 层间无同步屏障
```

### 整体架构

- **Attention 权重**：每个 rank 完整复制，本地执行
- **MoE 专家权重**：在 DWDP group 内跨 GPU 分区存储，每 rank 只保留本地专家
- **执行时**：在执行第 $l$ 层 MoE + 第 $l+1$ 层 Attention 期间，异步预取第 $l+1$ 层缺失的远端专家（双缓冲）
- **通信原语**：使用 `cudaMemcpyAsync`（copy engine，不占 SM 资源），而非 NCCL all-gather

---

## 主要技术设计

### 1. 异步远端权重预取（双缓冲流水线）

```
Layer l:
  ├── [Compute] MoE block (layer l)
  ├── [Compute] Attention block (layer l+1)
  └── [Async P2P] 预取 layer l+1 的缺失专家权重
        ↓
Layer l+1:
  等待预取完成 → 执行 MoE → 释放预取缓冲区 → 继续下一层
```

- 利用 layer $l$ 的 MoE 计算窗口 + layer $l+1$ 的 Attention 计算窗口来**隐藏通信延迟**
- 预取越能被计算覆盖，DWDP 收益越大（序列越长、batch 越大时计算窗口越宽）

### 2. 优化一：消除分裂权重合并开销

**问题**：每层 MoE 的权重分散在"本地专家缓冲区"和"远端专家缓冲区"两处，标准 groupedGEMM kernel 要求权重连续 → 需要先做 D2D 拷贝合并，引入额外延迟（基线实现增加 34 μs / 迭代）。

**方案**：扩展 groupedGEMM kernel（基于 CuTeDSL），支持 **TensorList 多缓冲区直接输入**，kernel 内部按 token 路由选择本地/远端缓冲区，无需外部合并 → 消除 D2D 合并拷贝，TPS/GPU 额外提升约 **3%**。

### 3. 优化二：时分复用缓解多对一通信竞争

**问题**：异步预取时，多个目标 rank 可能同时向同一源 rank 拉取权重，源端 copy engine 串行化请求 → 产生 compute bubble（实测可见）。

**分析**：理论上对于 DWDP4，发生 C=2 竞争的概率高达 44%，随组规模增大而增加。

**方案**：将每次远端专家拉取切分为固定大小的**小切片（slice）**，按轮询（round-robin）顺序跨所有目标 rank 交错调度：

```python
# 伪代码：构建交错调度的 copy plan
for each parameter p:
    for offset in range(0, M, slice_size):
        for peer in round_robin(remote_peers):
            copy_plan.append((dst, src, chunk_size))
```

- copy engine 流水线可同时在途两个小切片，使源端在一个切片受阻时仍能服务其他目标
- 效果类似高性能网络中的虚拟通道隔离，防止单个阻塞请求拖慢所有并发传输
- 在计算窗口较小（MNT 小、ISL ratio 低）时收益最显著

---

## 实验结果

**实验配置**：GB200 NVL72 · DeepSeek-R1（NVFP4 MoE 量化，FP8 KV Cache）· TensorRT-LLM · 对比基线 DEP（数据并行 + 专家并行）

### Context-Only 性能（隔离上下文阶段）

| 测试维度 | TTFT 加速比 | TPS/GPU 加速比 |
|---------|-----------|--------------|
| ISL 1K（MNT=32768） | **1.27×** | **1.11×** |
| ISL 8K（MNT=32768） | 1.16× | 1.10× |
| ISL 32K（MNT=32768）| 1.11× | 1.09× |
| 高负载不均衡（ISL STD=4096）| 1.18× | 1.15× |

- 负载越不均衡，收益越大（验证核心动机）
- ISL 越短，TTFT 加速越显著（计算窗口相对较小，同步开销占比更高）

### 两项优化的消融（ISL=8K, ratio=0.5, MNT=16384）

| 方案 | TPS/GPU（归一化到 DEP） |
|-----|----------------------|
| DEP（基线） | 1.000 |
| DWDP + 仅消除合并 | 0.995（受通信竞争拖累）|
| DWDP + 消除合并 + 竞争缓解 | **1.081** |

在计算窗口较小的场景，竞争缓解优化至关重要。

### 端到端推理（分离部署，仅修改 Context Server）

| TPS/user 范围 | TPS/GPU 加速比 |
|-------------|--------------|
| 20–30 | **1.10×** |
| 40–50 | 1.08× |
| 60–70 | 1.10× |
| 80–90 | 1.06× |
| 170–180 | 0.97×（轻微回退）|

- 总体在 20–100 TPS/user 服务范围内提升端到端 **TPS/GPU 8.8%**
- DWDP 支持更细粒度的 context GPU 数量配置（单 rank 粒度）
- **代价**：TTFT 增加（context server GPU 数减少导致排队延迟上升，非计算效率下降）

---

## 关键结论

1. **消除层间同步是核心增益**：DEP 的 12% 同步开销在 DWDP 中降为零，这是最大的性能来源
2. **两项工程优化缺一不可**：消除分裂权重合并（+3% TPS/GPU）+ 时分复用缓解竞争（在计算窗口短时尤为关键），共同确保 DWDP 的净收益
3. **适用条件明确**：需要高带宽 GPU P2P 通信（NVLink）+ 足够大的计算窗口隐藏预取；解码阶段计算窗口太小，暂不适用
4. **支持弹性资源分配**：context server 可以以单 rank 粒度配置，而非受限于 TP/EP 对 GPU 数的倍数要求，有利于分离部署的资源优化

---

## 适用场景

- **MoE 大模型推理**（DeepSeek 系列、Qwen-MoE 等），专家权重占显存主体
- **分离部署（disaggregated serving）**中的 Context Server
- NVLink 高带宽互联可用的单节点多 GPU 系统（NVL72 等）
- 负载分布不均衡的生产推理场景

---

## 相关工作

- [[DEP]]（数据并行 + 专家并行）：主要对比基线，依赖 all-to-all 集体同步
- [[DeepSeek-R1]] / [[DeepSeek-V3]]：评估目标模型
- [[TensorRT-LLM]]：实现平台，DWDP 已提交 upstream PR #12136
- [[MPress]]：同为 NVLink D2D 传输优化，但场景是训练显存扩展而非推理同步消除
- [[DistServe]] / [[Splitwise]] / [[Mooncake]]：分离部署框架，DWDP 面向其 Context Server 侧优化
