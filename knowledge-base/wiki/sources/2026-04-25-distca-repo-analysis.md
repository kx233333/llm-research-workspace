---
tags: [素材摘要, 训练系统, 长上下文, 代码分析, 复现指南, 实验分析]
created: 2026-04-25
updated: 2026-04-25
source_type: GitHub Repo + 论文
source_path: https://github.com/hao-ai-lab/DistCA
paper_arxiv: "2510.18121"
paper_authors: [Yonghao Zhuang, Junda Chen, Bo Pang, Yi Gu, Yibo Zhu, Yimin Jiang, Ion Stoica, Eric Xing, Hao Zhang]
---

# DistCA 仓库深度剖析：复现指南与性能分析方法

> 通过 Core Attention Disaggregation 将二次复杂度的注意力计算从线性层中解耦，调度到专用 Attention Server 上，消除长上下文训练中的 DP/PP straggler 问题。512 H200 GPU + 512K context 上实现 **1.35× 加速**。

---

## 一、仓库架构总览

```
DistCA/                              # Apache 2.0 | Python 90% + CUDA/C++ 3%
├── distca/                          # ★ 核心 Python 包 (~2000 行)
│   ├── planner/                     #   批次规划算法
│   │   ├── planner.py               #     核心 planner
│   │   ├── wlb_planner.py           #     WLB-LLM 对照 planner
│   │   ├── equal_flops.py           #     等 FLOPs 分配
│   │   └── buffer_size_adjust.py    #     NVSHMEM buffer 自适应
│   ├── runtime/                     #   运行时调度
│   │   ├── dispatch_fn.py           #     ★ CA task 分发核心
│   │   ├── compute_metadata.py      #     注意力元数据计算
│   │   ├── cuda_graph.py            #     CUDA Graph 捕获
│   │   ├── shard_info.py            #     token-level shard 信息
│   │   ├── attn_kernels/            #     FlashAttention kernel 调度
│   │   │   ├── dispatch.py          #       kernel 分发
│   │   │   └── ops.py               #       FA 操作封装
│   │   └── megatron/                #   ★ Megatron-LM 集成层 (~1000 行)
│   │       ├── base_transformer_layer.py  # 修改后的 Transformer 层
│   │       ├── forward_backward_func.py   # Pipeline schedule 修改
│   │       ├── model_patch.py             # 运行时 monkey-patch
│   │       ├── distca_rope.py             # RoPE 位置编码适配
│   │       ├── per_doc_cp_attn.py         # 每文档 CP 注意力
│   │       ├── packed_seq_params.py        # packed sequence 参数
│   │       ├── create_group.py            # process group 创建
│   │       └── ping_pong/                 # ★ Ping-Pong 通信重叠
│   ├── profiling/                   #   性能 profiling 工具
│   ├── simulator/                   #   负载模拟器
│   ├── timemodule/                  #   计时模块
│   ├── visuals/                     #   可视化工具
│   └── mem.py                       #   显存追踪
├── csrc/                            # ★ CUDA/C++ 扩展 (~1000 行)
│   └── core/
│       ├── fastalltoall.cu/.h       #     NVSHMEM All-to-All kernel
│       ├── memcpy.cu/.h             #     自定义 GPU memcpy
│       └── nvshmem_utils.h          #     NVSHMEM 初始化
├── baseline/
│   └── wlbllm_original/csrc/        #   WLB-LLM baseline 的 CUDA 代码
├── benchmarks/
│   ├── example-3d-parallel/run3d.sh #   3D 并行 benchmark
│   └── example-4d-parallel/run4d.sh #   4D 并行 benchmark
├── artifact_evaluation/             #   ACM 可复现性评估脚本 (Modal 云)
├── pretrain_llama.py                # ★ 主训练入口
├── pretrain_llama_megatron.py       #   Megatron 变体入口
├── pretrain_llama.sh                #   Slurm 启动脚本
├── Dockerfile                       #   Docker 镜像定义
├── env.template.sh                  #   环境变量模板
├── setup.py                         #   Python 包安装
└── requirements.txt                 #   依赖列表
```

---

## 二、硬件配置需求

### 论文实验参考配置

| 参数        | 值                                           |
| --------- | ------------------------------------------- |
| **GPU**   | NVIDIA H200 (140 GB HBM3e, 990 TFLOPS FP16) |
| **节点**    | DGX H200, 8 GPU/节点                          |
| **节点内互联** | NVLink / NVSwitch                           |
| **节点间互联** | InfiniBand, **50 GB/s**                     |
| **测试规模**  | 64 / 128 / 256 / 512 GPU                    |
| **最大配置**  | **512 H200 = 64 节点**                        |

### 最低复现配置

| 级别               | GPU                 | 用途                  |
| ---------------- | ------------------- | ------------------- |
| **Smoke test**   | 1× H100/H200        | 验证安装 + 单层 fwd/bwd   |
| **单卡 benchmark** | 1× H100/H200        | 单 GPU 吞吐测量          |
| **3D 并行**        | 8× H100/H200（1 节点）  | TP=8, 无 PP          |
| **4D 并行**        | 16× H100/H200（2 节点） | TP=8, PP=2, DP auto |
| **论文复现**         | 64~512× H200        | 完整对比实验              |

> ⚠️ **关键约束**：CUDA 扩展编译目标为 `sm_90a`（Hopper 架构），**必须使用 H100/H200**。A100 需要改为 `sm_80`，但 NVSHMEM IBGDA 特性可能不可用。

---

## 三、软件环境搭建

### 精确版本锁定

| 组件 | 版本 | 说明 |
|------|------|------|
| CUDA | **12.8** | 必须匹配 |
| Python | **3.12** (conda) | |
| PyTorch | **2.7.0** (cu128) | |
| NCCL | **2.27.6** | |
| **NVSHMEM** | **3.2.5** | ★ 核心通信库 |
| **OpenMPI** | **5.0.8** | NVSHMEM 依赖 |
| TransformerEngine | **v2.4** | 需源码编译 |
| Megatron-LM | **core_v0.12.1** | 需源码安装 |
| NVIDIA Apex | latest | 需源码编译 |
| FlashAttention | **2.7.4** | 预编译 wheel |
| transformers | ≥4.40, <4.46 | HuggingFace |

### 完整安装流程

```bash
# ======== Step 1: Conda 环境 ========
conda create -n distca python=3.12 -y && conda activate distca
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ======== Step 2: 环境变量 ========
cp env.template.sh env.sh
# 编辑 env.sh：设置 CUDA_DIR, NVSHMEM_PREFIX, OPENMPI_DIR 等路径
source env.sh

# ======== Step 3: DistCA 本体 ========
pip install -e .
pip install -r requirements.txt

# ======== Step 4: TransformerEngine v2.4 (源码编译 ~30min) ========
pip install pybind11
git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine && git checkout v2.4
git submodule update --init --recursive
NVTE_FRAMEWORK=pytorch MAX_JOBS=64 pip install --no-build-isolation -v '.[pytorch]'

# ======== Step 5: Megatron-LM core_v0.12.1 ========
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git checkout core_v0.12.1
git submodule update --init --recursive && pip install -e .

# ======== Step 6: NVIDIA Apex (源码编译) ========
git clone https://github.com/NVIDIA/apex.git && cd apex
git submodule update --init --recursive
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

# ======== Step 7: FlashAttention 2.7.4 (预编译 wheel) ========
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/\
v0.3.18/flash_attn-2.7.4+cu128torch2.7-cp312-cp312-linux_x86_64.whl
pip install ./flash_attn-*.whl

# ======== Step 8: DistCA CUDA 扩展 (Hopper sm_90a) ========
pip install ninja cmake
cd csrc && cmake -B build -S ./ -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90a
cmake --build build

# ======== Step 9 (可选): WLB-LLM baseline CUDA 扩展 ========
cd baseline/wlbllm_original/csrc
cmake -B build -S ./ -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90a && cmake --build build
```

### Docker 替代方案（推荐初次尝试）

```bash
# Smoke test（单 GPU，自动清理）
docker run --gpus all --rm --shm-size=2g \
  -v $(pwd):/workspace/DistCA \
  -e DISTCA_ROOT=/workspace/DistCA \
  <image> /workspace/DistCA/scripts/docker_install_and_build.sh --smoke

# Benchmark（单 GPU）
docker run --gpus all --rm --shm-size=2g \
  -v $(pwd):/workspace/DistCA \
  -e DISTCA_ROOT=/workspace/DistCA \
  <image> /workspace/DistCA/scripts/docker_install_and_build.sh --benchmark
```

### 关键环境变量

```bash
# ===== 必须设置 =====
export CUDA_DIR=/usr/local/cuda-12.8
export CUDA_HOME=$CUDA_DIR
export NVSHMEM_PREFIX=/usr/local/nvshmem      # NVSHMEM 安装路径
export OPENMPI_DIR=/usr/local/openmpi          # OpenMPI 安装路径
export NCCL_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/

# ===== 运行时调优 =====
export NVSHMEM_IB_ENABLE_IBGDA=true            # 启用 InfiniBand GPU-Direct Async
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1      # 允许非确定性 TE kernel

# ===== 实验控制 =====
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2     # NVSHMEM 对称堆大小 (GB)
export EXPERIMENT_ENABLE_CUDA_GRAPHS=0         # CUDA Graph（实验性）
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1         # benchmark 模式跳过 optimizer
export EXPERIMENT_FA2A_SPLIT_SENDRECV=1         # FA All-to-All 拆分 send/recv
export EXPERIMENT_ADD_SELECTIVE_CKPT=1           # 选择性激活 checkpointing
export EXPERIMENT_LOG_MEMORY_USAGE=0            # 显存日志
export ENABLE_NSYS=0                            # Nsight Systems profiling
```

---

## 四、实验复现方案

### 4.1 模型配置

| 参数 | LLaMA-3-8B | LLaMA-34B |
|------|-----------|-----------|
| 层数 | 32 | 48 |
| Hidden dim | 4096 | 8192 |
| Attention heads | 32 | 64 |
| KV heads (GQA) | 8 | 16 |
| Head size | 128 | 128 |

### 4.2 并行配置

**所有实验固定 TP=8**（一个 NVLink 域 = 一个节点内）

#### 3D 并行（TP + CP/CAD + DP）

| Model | SeqLen | Batch | GPU 数 | DP |
|-------|--------|-------|--------|-----|
| 8B | 128K | 8 / 16 / 32 | 64 / 128 / 256 | auto |
| 8B | 256K | 4 / 8 / 16 | 64 / 128 / 256 | auto |
| 8B | 512K | 2 / 4 / 8 | 64 / 128 / 256 | auto |
| 34B | 128K | 4 / 8 / 16 | 64 / 128 / 256 | auto |
| 34B | 256K | 2 / 4 / 8 | 64 / 128 / 256 | auto |
| 34B | 512K | 2 / 4 / 8 | 64 / 128 / 256 | auto |

#### 4D 并行（TP + PP + CP/CAD + DP）

| Model | SeqLen | Batch | GPU 数 | PP |
|-------|--------|-------|--------|-----|
| 8B | 128K~512K | 8~128 | 64~256 | grid search |
| 34B | 128K~384K | 8~128 | 128~512 | grid search |

### 4.3 启动命令

```bash
# === 3D 并行 benchmark ===
bash ./benchmarks/example-3d-parallel/run3d.sh

# === 4D 并行 benchmark ===
bash ./benchmarks/example-4d-parallel/run4d.sh

# === Slurm 集群提交 ===
sbatch pretrain_llama.sh
# 或交互式：
salloc -N 2 -G 16 -t 01:00:00 bash pretrain_llama.sh
```

### 4.4 关键 CLI 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--num-tokens` | int | 每 rank 序列长度 |
| `--num-batches` | int | batch 数 |
| `--cp-size` | int | Context Parallel 度 |
| `--tp-size` | int | Tensor Parallel 度 |
| `--pp-size` | int | Pipeline Parallel 度 |
| `--num-microbatch` | int | micro-batch 数 |
| `--num-layers` | int | 覆盖模型层数 |
| `--model-path` | str | HuggingFace 模型标识 |
| `--use-planner` | flag | 启用 batch planning 算法 |
| `--max-sample-id` | int | 训练迭代数 |
| `--sample-name` | str | 数据集名称 |
| `--elongate-factor` | int | 序列拉长因子 |
| `--filter-threshold` | int | 最小 token 数过滤阈值 |

---

## 五、性能分析方法论

### 5.1 吞吐量分析

#### A. 端到端吞吐测量

```bash
# benchmark 模式：跳过 optimizer step，只测 fwd+bwd
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1

# 论文方法：30 个 batch 取平均
--max-sample-id 30
```

**计算公式**：
$$\text{Throughput (tokens/s)} = \frac{\text{batch\_size} \times \text{seq\_len}}{\text{avg\_iteration\_time}}$$

$$\text{MFU} = \frac{\text{Actual TFLOPS}}{\text{Peak TFLOPS per GPU} \times \text{num\_GPUs}}$$

> 论文假设 context-independent layers 的 MFU 为 50%。

#### B. W&B 集成

```bash
python pretrain_llama.py --enable-wandb --wandb-project distca-training ...
# 自动记录：loss, duration, sample_id, validation metrics
```

#### C. 自带计时模块

`distca/timemodule/` 提供 per-layer 计时，可测量：
- Context-independent layers (FFN, LayerNorm 等) 耗时
- Core Attention 耗时
- 通信耗时（All-to-All dispatch/collect）

### 5.2 瓶颈分析

#### A. NVIDIA Nsight Systems Profiling

```bash
export ENABLE_NSYS=1
# 运行 benchmark 后，生成 .nsys-rep 文件
# 使用 nsys-ui 查看：
nsys-ui <output>.nsys-rep
```

**重点关注**：
1. **All-to-All 通信时间** vs **计算时间** — Ping-Pong 是否有效重叠
2. **Attention Server 负载均衡** — 各 GPU 的 CA kernel 执行时间差异
3. **Pipeline Bubble** — PP 维度的空闲时间
4. **DP Straggler** — 不同 DP rank 的 iteration 时间差异

#### B. Straggler 检测

论文的核心 motivation 指标：

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **DP Straggler Ratio** | $\frac{max(T_{dp})}{min(T_{dp})}$ | DP rank 间最大/最小迭代时间比 |
| **PP Bubble Rate** | $\frac{T_{idle}}{T_{total}}$ | 流水线空闲占比 |
| **GPU Idle %** | 空闲 GPU 时间 / 总时间 | 因注意力不均衡导致的 GPU 空闲率 |

论文数据参考：
- DP=4, 512K → **19% GPU 空闲**
- DP=8, 512K → **55% GPU 空闲**

#### C. 通信瓶颈分析模型

论文给出的解析通信模型（LLaMA-34B, InfiniBand 50 GB/s）：

$$T_{comm} = \frac{\text{Query bytes} + \text{KV bytes}}{\text{bandwidth}} = \frac{l \cdot h_q + (s+1) \cdot l \cdot h_{kv}/2}{50 \text{ GB/s}}$$

$$T_{compute} = \frac{1320 \cdot 2^{20} \text{ FLOPs/token}}{990 \text{ TFLOPS} \times 0.5 \text{ MFU}}$$

当 $T_{comm} < T_{compute}$ 时通信可完全隐藏。34B 模型最多可切 **31 个 shard** 而不增加端到端延迟。

### 5.3 通算比分析

#### A. Compute vs Communication Overlap 验证

论文的消融实验方法——**信号通信测试**：

```
原始运行：发送完整 Q/K/V tensor
信号测试：只发送 1 byte 信号（模拟零通信开销）

如果两者延迟接近 → 通信被完全重叠 ✓
如果差异显著     → 通信成为瓶颈 ✗
```

论文结果：
- 8B, ≥16 节点：接近一致 → 重叠成功
- 8B, 8 节点：例外 — 计算不足以隐藏通信

#### B. Ping-Pong vs Single Stream 对比

```
Single Stream：  [发送 Q/K/V] → [等待] → [计算 CA] → [发送结果] → [等待]
Ping-Pong：     [发送 nb0] → [计算 nb1] → [发送 nb1] → [计算 nb0] → ...
                 ↑ 通信与计算交替，重叠率接近 100%
```

论文数据：单流比 Ping-Pong 慢 **10-17%**。

#### C. All-to-All 延迟随规模的增长

| 节点数 | All-Gather 占比 (Per-Doc CP) | DistCA All-to-All |
|--------|---------------------------|-------------------|
| 2 | ~3% | 可忽略（重叠） |
| 8 | ~15% | 可忽略（重叠） |
| 16 | ~25% | 可忽略（重叠） |
| 32 | ~40% | 可忽略（重叠） |

### 5.4 显存分析

#### A. 内置显存追踪

```bash
export EXPERIMENT_LOG_MEMORY_USAGE=1
# distca/mem.py 会在关键点记录 GPU 显存使用
```

#### B. 关键显存指标

| 指标 | 测量方式 | 说明 |
|------|---------|------|
| **激活内存不均衡比** | $\frac{max(M_{act})}{min(M_{act})}$ | 跨 rank 的激活内存最大/最小比 |
| **KV Cache 冗余开销** | $(CP-1)/CP \times M_{kv}$ | CP 复制导致的额外 KV 内存 |
| **NVSHMEM 对称堆** | `EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB` | 通信 buffer 占用 |
| **Peak Memory** | `torch.cuda.max_memory_allocated()` | 峰值显存 |

论文数据参考：
- Variable-length chunking: 512K 时 **1.08×–1.17× 激活内存不均衡**
- KV 冗余：2 节点 **3%** → 16 节点 **30%**

#### C. 选择性激活 Checkpointing

```bash
export EXPERIMENT_ADD_SELECTIVE_CKPT=1
# 只对 Attention 层做 activation checkpointing
# 减少峰值显存，但增加重计算开销
```

#### D. PyTorch 内置工具

```python
# 在代码中插入
torch.cuda.memory_summary()
torch.cuda.memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.reset_peak_memory_stats()
```

### 5.5 Planner/Scheduler 分析

#### A. 负载均衡质量

Planner 的核心参数：
- **容忍因子 ε**：最优范围 **0.10 – 0.15**
- **最小 shard 长度**：**128 tokens**（FlashAttention block size）

```bash
python pretrain_llama.py --use-planner ...
# Planner 输出每个 GPU 的 CA task 分配方案
# 可通过 distca/visuals/ 可视化负载分布
```

#### B. 模拟器

```python
# distca/simulator/ 提供离线模拟
# 不需要实际 GPU 即可评估调度方案的负载均衡度
```

---

## 六、推荐的分析流程

### Phase 1: 环境验证（1× GPU）

```bash
# 1. Docker smoke test
bash scripts/run_docker_single_gpu_smoke.sh

# 2. 单卡 benchmark
bash scripts/run_docker_single_gpu_benchmark.sh

# 验证：loss 稳定下降，无 OOM
```

### Phase 2: 小规模 baseline 对齐（8~16× GPU）

```bash
# 3D: TP=8, CP=1, 单节点
bash benchmarks/example-3d-parallel/run3d.sh

# 4D: TP=8, PP=2, 双节点
bash benchmarks/example-4d-parallel/run4d.sh

# 同时运行 WLB-LLM baseline 做对比
# 切换 mode: distca → wlbllm
```

### Phase 3: 性能 Profiling

```bash
# 1. Nsight Systems trace
ENABLE_NSYS=1 bash benchmarks/example-3d-parallel/run3d.sh
# 分析：Ping-Pong 重叠率, All-to-All 延迟, kernel 执行时间

# 2. 显存分析
EXPERIMENT_LOG_MEMORY_USAGE=1 bash benchmarks/example-3d-parallel/run3d.sh

# 3. 通信消融
# 修改代码只发送 1 byte 信号，对比端到端延迟变化
```

### Phase 4: 规模扩展分析

```
64 GPU → 128 → 256 → 512

每个规模点测量：
- 端到端吞吐 (tokens/s)
- MFU
- DP straggler ratio
- PP bubble rate
- 峰值显存
- All-to-All 通信占比
```

### Phase 5: 灵敏度分析

```
固定规模，扫描参数：
- 序列长度：128K → 256K → 384K → 512K
- Batch size：逐步增大到 OOM
- ε 值：0.05, 0.10, 0.15, 0.20
- PP degree：1, 2, 4, 8
- NVSHMEM buffer：1, 2, 4, 8 GB
```

---

## 七、核心算法实现路径

### 数据流全貌

```
┌─────────────────────────────────────────────────────────────┐
│                     DistCA 训练一个 step                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Planner 阶段（CPU）                                      │
│     ├─ 读取当前 batch 的文档长度分布                          │
│     ├─ 估算每个文档的 CA 工作量：W = s² / cp_size            │
│     ├─ 贪心分配 CA tasks 到各 Attention Server                │
│     └─ 输出：shard_info（每 GPU 的 CA task 列表）             │
│                                                             │
│  2. Forward 阶段                                             │
│     对每个 Transformer Layer:                                │
│     ├─ [所有 GPU] 计算 Context-Independent 层 (FFN, LN...)   │
│     ├─ [所有 GPU] 计算 QKV projection                        │
│     ├─ [调度] All-to-All: 发送 Q/K/V shard → Attention Server│
│     ├─ [Attention Server] 执行 FlashAttention: softmax(QK^T)V│
│     │   └─ Ping-Pong: nano-batch 交替发送+计算               │
│     ├─ [调度] All-to-All: 返回 Attention 输出                │
│     └─ [所有 GPU] 计算 Output Projection                     │
│                                                             │
│  3. Backward 阶段                                            │
│     └─ 反向传播（同样的 disaggregation 模式）                 │
│                                                             │
│  4. Gradient Sync                                            │
│     └─ DP AllReduce                                          │
└─────────────────────────────────────────────────────────────┘
```

### Ping-Pong 执行模式

```
时间 →
GPU_A (Context-Indep):  [FFN_0] [FFN_1] [FFN_0] [FFN_1] ...
                          ↓ send Q/K/V    ↓ send Q/K/V
GPU_B (Attn Server):    [recv nb0] [CA_nb0] [recv nb1] [CA_nb1] ...
                                    ↑ 重叠          ↑ 重叠
                         Stream 0 ───┘    Stream 1 ──┘

nb = nano-batch, 交替使用两个 CUDA stream
```

### In-Place Attention Server

GPU 在两个角色间**时分复用**：

```
GPU_i 的时间线：
  [计算自己的 FFN] → [作为 Attn Server 处理其他 rank 的 CA] → [计算自己的 FFN] → ...
  
  优势：不需要专用 GPU 做 Attention Server
  实现：distca/runtime/megatron/ping_pong/
```

---

## 八、与其他方案的定量对比

| 方案 | 512K, 8B, 256 GPU | 512K, 34B, 512 GPU | 机制 |
|------|-------------------|---------------------|------|
| **WLB-LLM** | 1.00× (baseline) | 1.00× (baseline) | Workload-balanced packing |
| **Per-Doc CP** | <1.00× (all-gather 瓶颈) | <1.00× | 每文档独立 CP |
| **DistCA** | **1.20×** | **1.35×** | CA 解耦 + Ping-Pong |

---

## 九、潜在坑点与注意事项

### 安装相关
1. **NVSHMEM 安装复杂**：依赖 OpenMPI + InfiniBand 驱动，集群上可能需要管理员权限
2. **TransformerEngine 编译慢**：v2.4 源码编译约 30 分钟
3. **CUDA 架构必须匹配**：`sm_90a` 仅 Hopper，A100 用户需改为 `sm_80` 并可能丢失 IBGDA
4. **FlashAttention 版本敏感**：必须用 2.7.4 + cu128 + torch2.7 的精确组合

### 运行相关
5. **CPU 亲和性**：每进程至少 3 核（PyTorch + NVSHMEM + NCCL 各一个线程）
6. **NVSHMEM buffer 大小**：默认 2GB，长序列可能不够，需调 `EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB`
7. **InfiniBand IBGDA**：`NVSHMEM_IB_ENABLE_IBGDA=true` 要求驱动支持 GPU-Direct Async
8. **显存碎片化**：34B 4D 并行实验中观察到显存碎片问题

### 复现相关
9. **论文 Figure 3, 4, 6, 9-12 需要 8~64 节点**：Artifact evaluation 中明确说明这些结果只有作者能复现
10. **Grid search PP/DP**：论文对 baseline 做了 PP/DP 的 grid search 取最优，复现时需同样操作
11. **30 batch 平均**：每个配置点跑 30 个 batch 取平均，确保统计可靠性

---

## 十、关键概念链接

- [[CAD-DistCA]] — 本仓库对应的概念实体页
- [[Dynamic CP]] — Megatron-LM 内置的替代方案，在数据调度层面解决负载均衡
- [[FlashAttention]] — DistCA 使用的 Attention kernel
- [[长上下文训练]] — 应用场景
- [[负载均衡与变长序列]] — 核心问题
- [[Ping-Pong Pipeline]] — 通信重叠模式
- [[NVLink]] / [[NVSwitch]] — 节点内互联
- [[USP]] — Hybrid CP 的另一种方案
