---
tags: [综合分析, 研究路线图, CAD, NVL72, 负载均衡, 长文本]
created: 2026-04-23
updated: 2026-04-23
sources:
  - 2026-04-22-mpress
  - 2026-04-22-chunkflow
  - 2026-04-22-dwdp
  - 2026-04-22-gpu-bandwidth-hierarchy
  - 2026-04-22-long-context-supernode-survey
  - 2026-04-22-attention-ffn-imbalance-survey
  - 2026-04-22-dp-only-attention-ffn-survey
type: synthesis
---

# NVL72 时代长文本训练系统的研究方向与 Idea 汇总

> 基于知识库里 7 篇素材、52 个 wiki 页面，以及围绕 [[CAD-DistCA]]、[[DWDP]]、[[ChunkFlow]] 展开的深入讨论，梳理出 **6 个核心洞察 + 15 个具体 idea**，按可做性和影响力分级。

---

## 目录

- [0. 背景与问题领域](#0-背景与问题领域)
- [1. 核心洞察（六个根本观察）](#1-核心洞察六个根本观察)
- [2. 研究方向：论文级（Paper-Level）](#2-研究方向论文级paper-level)
- [3. 工程 Idea：可快速落地](#3-工程-idea可快速落地)
- [4. 跨方向的统一图景](#4-跨方向的统一图景)
- [5. 优先级与选型建议](#5-优先级与选型建议)

---

## 0. 背景与问题领域

### 0.1 问题空间

围绕 **LLM 分布式系统** 的两条交叉主线：

- **主线 A：变长长上下文下的通信/计算/显存负载均衡**
  - 代表：[[ChunkFlow]]、[[WLB-LLM]]、[[CAD-DistCA]]、[[ByteScale]]、[[FlexSP]]、[[LobRA]]
- **主线 B：超节点（NVL72 / GH200）带来的新硬件形态**
  - 代表：[[DWDP]]、[[SuperOffload]]、[[MegaScale-Infer]]

两条主线在 [[GB200-NVL72]] 上**汇合**——长上下文 + 超节点是目前最前沿也是最空白的研究区域。

### 0.2 关键数据点

| 观察 | 数值 | 来源 |
|------|------|------|
| DP=8 + 变长场景 | bubble 占 **55%** | [[CAD-DistCA]] |
| Attention vs FFN 交叉点 | $s^* \approx 2h$（$h$=4096 时 ~8K）| [[Attention-FFN 计算量不平衡]] |
| MoE a2a 同步开销（20% CV 负载偏斜下）| **12%** | [[DWDP]] |
| NVL72 vs DGX H100 NVLink 域大小 | 72 vs 8（**9×**）| [[GB200-NVL72]] |
| NVL72 vs 跨机柜 IB 带宽比 | 1800 : 100 GB/s（**18×**）| [[GB200-NVL72]] |

---

## 1. 核心洞察（六个根本观察）

这六个洞察是后续所有方向的出发点。它们不是方案，而是**看问题的角度**。

### 洞察 1：变长负载不均衡的根源是"两个守恒量无法同时平衡"

对 packed micro-batch 的文档长度 $\{l_i\}$：

$$
\text{FLOPs} = \alpha \sum l_i^2 + \beta \sum l_i, \quad \text{Memory} = \gamma \sum l_i
$$

要同时平衡计算和显存，需 $\sum l_i = \sum l_j'$ **AND** $\sum l_i^2 = \sum l_j'^2$——**长尾分布下几乎无解**。

这是 [[CAD-DistCA]] 的切入点，也是 [[ChunkFlow]]、[[WLB-LLM]] 等方案都在规避的核心矛盾。

### 洞察 2：a2a 是所有不均衡的"放大器"

训练对不均衡敏感的根因是**集合通信的屏障语义**——"等最慢的"。a2a 在 MoE / CP 场景下特别致命，原因：

1. **双向阻塞**：每 rank 发给每 rank 不等量数据，一端慢 = 所有 pair 等
2. **双重不均衡**：计算 + 通信不均衡都在 a2a 阶段暴露
3. **高频**：80 层模型 160 次 a2a / iter，远多于 all-reduce

消除 a2a 的三条路：**(a) 消灭**（[[DWDP]]、[[ChunkFlow]]）、**(b) 保留但隐藏**（[[MegaScale-Infer]]、DualPipe）、**(c) 减少源头不均衡**（aux-loss-free balancing）。

### 洞察 3：CAD 的 Scheduler 是"扁平"的——有三个未被感知的结构

CAD 的 scheduler 用 $E = \Delta F_{\max} / V_{\text{comm}}$ 做贪心，但：

- **不感知硬件拓扑**：NVL72 机柜内 vs 跨机柜带宽差 18×，scheduler 用同一个 $V_{\text{comm}}$
- **不感知数据结构**：同一序列的多 shard 共享 KV context，scheduler 当独立 task 处理
- **不感知模块种类**：只处理 attention，MoE 的 expert 不均衡无能为力

这三个"扁平"各自对应一个改进方向（后面的 idea 1、4、5）。

### 洞察 4：Causal Attention 的 shard 内在不均衡 $2n-1$

一个长度 $L$ 的序列切成 $n$ shard 时，最后 shard 的工作量是第一 shard 的 $(2n-1)$ 倍：

| $n$ | 最后/最前 比 |
|-----|-----------|
| 4 | 7× |
| 8 | 15× |
| 16 | 31× |

这是**结构性**不均衡（源于 causal mask），比跨文档的不均衡（2-5×）还严重。CAD 的全局 scheduler 错过了这个更锐利的矛盾。

### 洞察 5：变长不均衡在 Dense 和 MoE 里地位不同

| 模型类型 | 变长不均衡的角色 | 主要瓶颈 |
|---------|--------------|---------|
| Dense | **dominant 矛盾** | Attention $s^2$ 不均衡 |
| MoE | 次要矛盾 | Expert 路由偏斜 + a2a |

CAD / ChunkFlow 类方案主要针对 dense。MoE 场景下 attention 不均衡在整层占比降到 ~30%，主要瓶颈转移到 FFN / a2a。**现有方案几乎都偏科**。

### 洞察 6：NVL72 是"有硬件没软件"的状态

NVL72 的硬件创新（72 GPU 单 NVLink 域、1.8 TB/s per GPU、C2C 900 GB/s）已经就绪，但：

- CAD 假设通信均匀 → 不适配层次化拓扑
- ChunkFlow 假设单序列不跨 rank → 不利用大带宽
- DWDP 只做推理 forward → 训练 backward 空缺
- 没有方案统一利用 "72 GPU 单域 + Grace CPU DRAM + C2C" 的完整硬件栈

**这是大量研究机会的根源**。

---

## 2. 研究方向：论文级（Paper-Level）

这些方向需要论文级创新，规模较大，适合博士课题或团队协作。

### 方向 1：Topology-Aware CAD Scheduler（拓扑感知调度）

**对应洞察**：洞察 3、6

**空白**：[[CAD-DistCA]] 的 scheduler 对所有 GPU 用同一个 $V_{\text{comm}}$ 估算，在 NVL72 的 18× 带宽差异下会系统性地错误估算跨机柜搬运成本。

**技术路径**：

- **图分区**：METIS / KaHIP 把 GPU 组织成层次化加权图，边权反映带宽
- **Hierarchical Scheduler**：第一级跨机柜粗粒度，第二级机柜内沿用 CAD greedy
- **Bandwidth-weighted priority**：改为 $E = \Delta F_{\max} / (V_{\text{comm}} / B_{\text{path}})$，其中 $B_{\text{path}}$ 是该对 GPU 的实际带宽

**预期收益**：
- 跨机柜通信量减 70%
- 扩展性提升到 1000+ GPU 规模
- 端到端加速在 2+ NVL72 集群场景下 1.2-1.3×

**反方意见**：METIS 开销 vs scheduler 频率可能紧张；实际 NVL72 集群的 IB 拓扑不一定能简单建模。

**论文题目建议**：*"Topology-Aware Core Attention Disaggregation for NVL72-Scale Training"*

---

### 方向 2：Hierarchical CAD with Sequence Groups（层次化 CAD：组内 + 组间）

**对应洞察**：洞察 3、4

**空白**：同一序列的多 worker 天然共享 KV context 且存在结构性 $2n-1$ 不均衡。CAD 的全局 scheduler 把它们当独立 task，导致 KV 重复传输 + 错过最锐利的优化目标。

**技术路径**：

```
Level 1（组内）：同 sequence 的 worker 形成通信组
  - 共享 KV context（组内 all-gather 一次）
  - CAD 风格 token-level 调度解决 $2n-1$ 不均衡
  - 组尽量落在同一 NVL72 机柜内

Level 2（组间）：跨组粗粒度平衡
  - 只处理文档长度差异（2-5× 不均衡）
  - 粒度粗，通信少
```

**算法借用**：
- Recursive partitioning（算法经典）
- Hierarchical all-reduce 思想（NCCL）

**预期收益**：
- KV 传输量减 60-70%（组内局部化）
- 组内 $2n-1$ 不均衡消除
- 自然匹配 NVL72 的层次拓扑
- 端到端加速在长文档场景下 1.5-1.6×（vs CAD 1.35×）

**论文题目建议**：*"Hierarchical Core Attention Disaggregation: Exploiting Sequence-Group Locality in Long-Context Training"*

**和方向 1 的关系**：两者可以合并——方向 1 是"硬件拓扑层次"，方向 2 是"数据结构层次"。合并后的完整版：**两维度层次化 CAD**。

---

### 方向 3：CAD for Training Backward（反向通信的闭环）

**对应洞察**：洞察 2、3

**空白**：[[CAD-DistCA]] 只做了 forward。Backward 通信量翻倍（Q, K, V, grad_O 都要传），FlashAttention 的 backward kernel 窗口更小（recompute 前向），Ping-Pong 是否能完全隐藏通信**未被验证**。

**技术路径**：

- **Bi-directional scheduling**：forward 决策同时考虑 backward 最优
- **Activation stash on server**：forward shard A 搬到 server 5，stash activation；backward 自然去 server 5（通信量减半）
- **Min-Cost Max-Flow**：把 forward + backward 建模成有向图，最小化总通信代价
- **Rematerialization-aware**：recompute 的 FLOPs 也要纳入 scheduler 的平衡目标

**预期收益**：
- Backward 通信量减 50%
- 整个训练 step 的 bubble 全消除
- 端到端加速从 1.35× 拉到 1.6-1.8×

**反方意见**：backward 数据依赖紧，难度比 forward 高 2-3 倍；CAD 作者很可能自己在做。

---

### 方向 4：CAD + DWDP 融合（MoE 训练的完整 Disaggregation）

**对应洞察**：洞察 5、6

**空白**：
- CAD 只处理 attention，不管 MoE
- DWDP 只处理 MoE forward 推理，不管 training backward 和 attention

组合起来是**MoE 长上下文训练的终极方案**，但没人做过。

**架构草图**：

```
每个 MoE 层：
  [layernorm, QKV proj]              ← 本地
        ↓
  [All-to-all → CA Server pool]      ← CAD 负责（attention $s^2$ 均衡）
        ↓
  [post-CA, O proj]                  ← 本地
        ↓
  [Router + top-K]                   ← 本地
        ↓
  [Async P2P prefetch expert]        ← DWDP 负责（消除 a2a）
        ↓
  [Expert GEMM]                      ← 用预取权重本地算
        ↓
  [Combine，也可异步]
```

**技术挑战**：
- CAD 的 all-to-all 和 DWDP 的 P2P 都用 NVLink → **带宽仲裁**
- 两个 scheduler 需要协同，不能各自次优
- HBM fragmentation 叠加放大

**算法借用**：
- Work Stealing（Cilk / Rayon）统一任务队列
- Producer-Consumer with Backpressure（TCP congestion control 思想）
- 两级 scheduler 架构

**预期收益**：
- MoE 训练首次同时享受 CA + Expert disaggregation
- 填补 MoE 长上下文场景的空白
- 如果做成，是标志性系统工作

**论文题目建议**：*"NVL72-Native MoE Training: Unified Disaggregation of Core Attention and Expert Computation"*

---

### 方向 5：Memory-Tiered CAD（三层 KV 存储）

**对应洞察**：洞察 6

**空白**：CAD 的 in-place attention server 只用 HBM 做 KV buffer。但 NVL72 有完整的三层存储：
- HBM（8 TB/s, 192 GB per GPU）
- Peer GPU HBM（1.8 TB/s NVLink）
- Grace CPU DRAM（900 GB/s C2C + 17.3 TB 总容量）

**技术路径**：

```
Hot tier (HBM):       当前正在处理的 shard 的 KV
Warm tier (Peer HBM): 即将调度的 shard KV（预取中）
Cold tier (Grace DRAM): 历史 / 未激活文档的 KV
```

**算法借用**：
- Multi-Tier Buffer Pool（数据库经典）
- Stride Prefetcher（CPU 缓存经典）
- LFU-with-Aging 淘汰策略

**预期收益**：
- 单 "GPU" 可用 KV 容量从 192 GB → 672 GB（含本机 Grace）
- **解锁 1M+ 上下文训练**
- 跨 server 借 KV 成为可能

**自然连接 [[SuperOffload]] 的思想到训练场景**。

---

### 方向 6：Dynamic Plan with Feedback Control（动态反馈式联合优化）

**对应洞察**：洞察 6

**空白**：[[MPress]] 的 static plan 在 MoE 动态路由下失效；[[ByteScale]] 半动态；[[CAD-DistCA]] per-batch 动态但仅 attention。**没有全栈动态调度器**——同时根据 routing 偏斜、序列长度、通信延迟动态调整 swap / recompute / shard / offload。

**技术路径**：
1. Runtime telemetry：routing imbalance / stage utilization / memory / comm latency
2. 在线 plan 重评估（每 100 iter）
3. Low-overhead 切换机制
4. Bandit 或 RL 选择器积累经验

**预期收益**：
- 对 MoE routing / RL / mixed workload 场景收益特别大
- 打开"分布式训练即控制问题"的研究范式

**反方意见**：训练任何动态都影响复现性；RL 调度器的训练数据哪来。

---

## 3. 工程 Idea：可快速落地

这些 idea 不需要论文级创新，更适合作为系统增强 / 开源 contribution。

### Idea 7：静态内存分配器 + CUDA Graph（解 CAD 的 34B fragmentation）

**空白**：CAD 作者**自己点名**的 future work：34B 场景下 memory fragmentation 限制性能。

**算法借用**：
- Buddy Allocator / Slab Allocator（OS 内核经典）
- 按 2^k 分级预分配 slab
- CUDA Graph 减少 kernel launch overhead（FlashAttention 3 已验证）

**预期收益**：34B 4D 加速从 1.15× → 1.30-1.35×。

**为什么成熟**：几十年成熟算法，不需要理论创新。

---

### Idea 8：Adaptive $\epsilon$（Scheduler tolerance 自适应）

**空白**：CAD scheduler 的关键超参 $\epsilon$（balance tolerance）需要对每个 workload grid search，甜蜜点 0.10-0.15。

**算法借用**：
- Multi-Armed Bandit（UCB1 / Thompson Sampling）把 $\epsilon$ 离散成几个 arm
- 或 Hill Climbing 小步调整

**预期收益**：消除 grid search 成本，对 RL / SFT 非稳态 workload 收益明显。

---

### Idea 9：Online Profiler Calibration（在线校准 profiler）

**空白**：CAD 的 offline profiler 对 GPU 漂移 / 新架构 / 热状态敏感。

**算法借用**：
- 卡尔曼滤波 / EWMA 在线更新系数
- KNN + LRU cache 做无 prior 的新 pattern 处理

**预期收益**：不需要热身，scheduler 决策更准。

---

### Idea 10：Consistent-Hashing KV Locality（KV 亲和性）

**空白**：CAD scheduler 是 stateless 的——每 iter 重新分配，上次 server 上的 KV 被浪费。

**算法借用**：
- Consistent Hashing（Dynamo / Memcached 经典）+ Virtual Nodes
- Two-Choice Load Balancing（Power of Two Choices）
- 数据结构：红黑树维护 hash ring，O(log n) 查找

**预期收益**：
- RL / SFT 场景 KV 复用率 ↑ 60-80%
- 通信量减 20-30%

---

### Idea 11：Bandwidth-Proportional Work Stealing

**空白**：CAD 静态分配 shard；NVL72 下不同 server pair 带宽差 18×，有些 server 的"有效工作能力"更低。

**算法借用**：
- Work Stealing（Cilk / Rayon）
- 优先级队列按带宽排序
- 分布式决策，无需全局 scheduler

**预期收益**：对动态 workload 鲁棒性强，天然适配层次拓扑。

---

### Idea 12：Composability-Aware Packer（和 WLB-LLM 融合）

**空白**：CAD 被动接受 data loader 的 micro-batch。如果 packer 知道 CAD 存在，可以**提前把长文档拆分到多 rank**，让 scheduler 搬运量减少。

**算法借用**：
- First-Fit Decreasing / Best-Fit Decreasing
- Streaming Bin Packing
- 双索引（$\sum l$ 和 $\sum l^2$）类似 Redis skiplist

**预期收益**：CAD scheduler 搬运量减 30-50%，Ping-Pong 更易藏。

**亮点**：结合 [[WLB-LLM]] 的 data-side packing + CAD 的 compute-side scheduling。

---

### Idea 13：KV Compression on-the-fly（跨机柜场景）

**空白**：跨 NVL72 机柜时 IB 是瓶颈，CAD 传完整 Q/K/V 浪费。

**算法借用**：
- Low-Rank Decomposition（MLA 思想）
- FP8 / INT4 量化（only for KV transport）
- Fused decompression 在 kernel 内

**预期收益**：跨机柜通信量减 50-75%，允许更激进的 shard migration。

---

### Idea 14：Segment Tree for Load Tracking

**空白**：CAD scheduler 每次迁移 O(n) 扫描所有 server。72 server 还行，扩展到 1000+ 瓶颈。

**算法借用**：
- Segment Tree with Lazy Propagation（O(log n) 更新/查询）
- Fenwick Tree（常数更小）
- Pairing Heap 加速 top-k surplus/deficit

**预期收益**：复杂度 O(n²) → O(n log n)，支持 1000+ server。

---

### Idea 15：Grace CPU Scheduler Offload

**空白**：CAD scheduler 跑在 host CPU 上，每 iter 有 CPU-GPU 同步开销。NVL72 的 Grace CPU 有大量闲置算力。

**算法借用**：
- Double Buffering（iter N 跑 scheduler for iter N+1）
- Lock-Free Ring Buffer via C2C shared memory
- Atomic epoch counter

**预期收益**：
- 消除 sync overhead
- 利用 Hopper:Grace 330 的 FLOPS 比里被浪费的 Grace
- 为更激进的 scheduler 创新打开大门

---

### Idea 16：Dynamic ChunkSize + C2C KV Offload（ChunkFlow 升级版）

**空白**：
- [[ChunkFlow]] 的 chunksize 固定
- docx 调研者评论："chunksize 能否可变"
- ChunkFlow 自己 future work 提到 KV offload

**技术路径**：
- 按 chunk 负载动态调 ChunkSize
- KV state 跨 chunk 通过 C2C offload 到 Grace DRAM
- 1M+ 上下文训练 HBM 压力减 30-50%

**反方意见**：仅适用 NVL72 / GH200，DGX H100 PCIe 会瓶颈。

---

### Idea 17：Grace CPU 承担 MoE Routing

**空白**：MoE routing 是小批量离散决策，在 Hopper 上是算力浪费；Grace CPU 更适合。

**技术路径**：
- C2C 把 token representations 送到 CPU
- CPU 计算 top-K routing
- CPU 生成 dispatch plan 返回 GPU

**预期收益**：每层 5-15 μs，80 层下每 iter 省 400-1200 μs，约 3-5% 加速。

---

### Idea 18：SLO-Aware Priority Queue（推理版 CAD）

**空白**：CAD 是训练系统；如果扩展到 inference prefill，需考虑 SLO（TTFT）。

**算法借用**：
- EDF（Earliest Deadline First）
- Priority Queue + Admission Control
- Weighted Fair Queueing 做多租户

**预期收益**：
- 把 CAD 扩展到 inference
- 和 [[DWDP]] 组合做 NVL72 MoE inference 终极方案

---

## 4. 跨方向的统一图景

### 4.1 一个完整系统的蓝图

把上面 18 个 idea 整合起来，其实描绘的是一个 **"NVL72-Native Long-Context Training System"**：

```
┌─────────────────────────────────────────────────────┐
│  顶层：Adaptive Controller（idea 8）                 │
│    ↓ 实时调整参数                                    │
├─────────────────────────────────────────────────────┤
│  Scheduler 层：两维度层次化（方向 1 + 方向 2）        │
│    - 硬件层次：机柜内 vs 跨机柜                       │
│    - 数据层次：组内（同 sequence）vs 跨组             │
│    + Work Stealing（idea 11）做机柜内动态             │
│    + Consistent Hashing（idea 10）做 KV locality     │
│    + Segment Tree（idea 14）做 O(log n) 维护         │
│    ↓ 决策 shard 去哪                                 │
├─────────────────────────────────────────────────────┤
│  Memory 层：Multi-Tier Pool（方向 5 + idea 13, 16）   │
│    - HBM → Peer HBM → Grace CPU DRAM                │
│    - On-the-fly compression 做跨机柜                 │
│    - Static allocator + CUDA Graph（idea 7）         │
│    + Dynamic ChunkSize（idea 16）                    │
├─────────────────────────────────────────────────────┤
│  执行层：Backward-aware CAD（方向 3）+ DWDP（方向 4）  │
│    - 前后向统一优化                                  │
│    - CAD 管 attention                                │
│    - DWDP 管 expert                                  │
│    + Grace CPU 做 routing（idea 17）                 │
├─────────────────────────────────────────────────────┤
│  Runtime 加速：CPU Offload（idea 15）                │
│    - Grace 跑 scheduler                              │
│    - 双缓冲消除 sync                                 │
└─────────────────────────────────────────────────────┘
```

### 4.2 算法数据结构视角的统一观察

所有这些 idea 都在应用一个统一模式：

> **CAD 把一个"全局对称的不平衡问题"转化成了一个"分布式调度问题"。一旦进入调度问题的框架，所有经典的系统 / OS / 数据结构技巧都能用上：**
>
> - **Bin Packing** → idea 12, 方向 4（近似算法）
> - **Cache Management** → idea 10, 方向 5（LRU / LFU / Consistent Hashing）
> - **Memory Allocation** → idea 7, 方向 5（Buddy / Slab / Tiered）
> - **Control Theory** → idea 8, 方向 6（Bandit / EWMA / RL）
> - **Scheduling Theory** → 方向 3, idea 18（EDF / Work Stealing / Min-Cost-Flow）
> - **Graph Theory** → 方向 1, 方向 3（METIS / Min-Cost Max-Flow / Hierarchical）
> - **Concurrent Data Structures** → idea 11, 14, 15（Lock-free queues / Segment Tree）

CAD 的深层价值不是一个具体方案，而是**打开了一扇门**：让 Transformer 训练从"把 model 拼好"变成"把 compute 调度好"。

---

## 5. 优先级与选型建议

### 5.1 总表

| # | 名称 | 难度 | 影响 | 关键算法 | 推荐 |
|---|------|------|------|---------|------|
| **方向 1** | Topology-Aware Scheduler | 中 | 高 | METIS / Hierarchical | ⭐⭐⭐⭐⭐ |
| **方向 2** | Hierarchical with Sequence Groups | 中 | 高 | Recursive Partitioning | ⭐⭐⭐⭐⭐ |
| **方向 3** | CAD for Training Backward | 高 | 高 | Min-Cost Max-Flow | ⭐⭐⭐⭐ |
| **方向 4** | CAD + DWDP 融合 | 极高 | 极高 | Work Stealing | ⭐⭐⭐⭐ |
| **方向 5** | Memory-Tiered CAD | 中 | 高 | Multi-Tier Buffer Pool | ⭐⭐⭐⭐⭐ |
| **方向 6** | Dynamic Feedback Plan | 极高 | 中 | Bandit / RL | ⭐⭐⭐ |
| idea 7 | Static Allocator + CUDA Graph | 低 | 中 | Buddy / Slab | ⭐⭐⭐⭐ |
| idea 8 | Adaptive $\epsilon$ | 低 | 中 | Bandit | ⭐⭐⭐⭐⭐ |
| idea 9 | Online Profiler | 低 | 中 | EWMA / Kalman | ⭐⭐⭐⭐ |
| idea 10 | Consistent-Hashing KV | 低 | 中高 | Consistent Hashing | ⭐⭐⭐⭐⭐ |
| idea 11 | Work Stealing | 中 | 中高 | Work Stealing | ⭐⭐⭐⭐ |
| idea 12 | Composability Packer | 中 | 中高 | Bin Packing | ⭐⭐⭐⭐ |
| idea 13 | KV Compression | 中 | 中高 | Low-Rank / 量化 | ⭐⭐⭐⭐ |
| idea 14 | Segment Tree Tracker | 低 | 中 | Segment Tree | ⭐⭐⭐ |
| idea 15 | Grace CPU Scheduler | 中 | 中高 | Double Buffering | ⭐⭐⭐⭐ |
| idea 16 | Dynamic ChunkSize + C2C | 低 | 中高 | Multi-Tier Pool | ⭐⭐⭐⭐⭐ |
| idea 17 | Grace CPU Routing | 中 | 中 | CPU-GPU Overlap | ⭐⭐⭐ |
| idea 18 | SLO Priority Queue | 中 | 中 | EDF / WFQ | ⭐⭐⭐ |

### 5.2 推荐优先级

#### 🥇 第一优先（最容易出成果）

**方向 2：Hierarchical CAD with Sequence Groups**

- 空白最明确（CAD 原论文完全没处理 sequence 内部结构）
- 技术路径清晰（recursive partitioning）
- 收益可观（1.5× vs 1.35×）
- 可以单独成 paper，也可以和方向 1 合并成**两维度层次化 CAD**

---

#### 🥈 第二优先（最符合工程 ROI）

**idea 8 + idea 10 + idea 16 的组合**

- idea 8：Adaptive $\epsilon$（让 CAD 鲁棒）
- idea 10：Consistent-Hashing KV（让 CAD 利用历史状态）
- idea 16：Dynamic ChunkSize + C2C offload（让 ChunkFlow 升级到 1M+ 上下文）

三者都不需要论文级创新，但组合起来能**显著提升工业落地体验**。

---

#### 🥉 第三优先（最大野心）

**方向 4：CAD + DWDP 融合**

- 如果做成是**MoE 长上下文训练**的终极方案
- 填补业界真实空白（DeepSeek-V3 续训阶段痛点）
- 难度极高，需要系统团队
- 风险大但收益大

---

### 5.3 按研究场景的建议

| 场景 | 首选方向 |
|------|---------|
| Dense 长上下文训练（论文）| 方向 1 + 方向 2 合并 |
| MoE 长上下文训练（工业）| 方向 4 |
| 工程优化 CAD | idea 7 + 8 + 10 |
| 推理延伸 | idea 18 |
| 硬件利用极致 | 方向 5 + idea 16 + idea 17 |

---

## 6. 一个更宏大的论述

把所有观察和方向整合成一段话：

> **NVL72 把"a2a 是瓶颈"这个长期假设打破后，软件栈的每一层都需要重新审视——从数据打包到梯度同步到显存管理到硬件异构利用。现有论文每一个都只在自己的一亩三分地做了局部优化：CAD 解决了 dense 模型 attention 不均衡，DWDP 解决了 MoE 推理 a2a，ChunkFlow 解决了长上下文 SFT 显存，SuperOffload 解决了 CPU-GPU 紧耦合。但没有人把它们整合起来，也没有人为 NVL72 的层次化硬件 + 数据的结构性不均衡设计统一的 scheduler。**
>
> **这个整合的缺失，是 NVL72 时代分布式训练系统最大的研究空白。**

---

## 相关页面

### Sources

- [[2026-04-22-mpress]]
- [[2026-04-22-chunkflow]]
- [[2026-04-22-dwdp]]
- [[2026-04-22-gpu-bandwidth-hierarchy]]
- [[2026-04-22-long-context-supernode-survey]]
- [[2026-04-22-attention-ffn-imbalance-survey]]
- [[2026-04-22-dp-only-attention-ffn-survey]]

### 核心 Entities

- [[CAD-DistCA]]（主要基底）
- [[DWDP]]
- [[ChunkFlow]]
- [[MegaScale-Infer]]
- [[GB200-NVL72]]
- [[NVLink-5.0]]
- [[NVLink-C2C]]
- [[Grace-CPU]]
- [[Core-Attention]]
- [[MoE-a2a]]
- [[CP-a2a]]

### 主题

- [[负载均衡与变长序列]]
- [[并行策略与同步开销]]
- [[超节点架构]]
- [[Attention-FFN 计算量不平衡]]
- [[显存优化]]
