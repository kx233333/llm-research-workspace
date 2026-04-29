---
tags: [素材摘要, 训练系统, 长上下文, 部署指南, 实验方案]
created: 2026-04-27
updated: 2026-04-28
source_type: 实验方案
source_path: https://github.com/hao-ai-lab/DistCA
---

# DistCA 部署实验方案 — "L20Y"(H800) × 32 GPU + 256K

---

## 〇、集群拓扑总览

```
┌─────────────────────────────────────────────────────────────────────┐
│  Taiji 集群 — 4 节点 × 8 GPU = 32 GPU                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Node 0 (launcher / host)          Node 1 (worker-0)               │
│  hostname: gpu-h800-sz-56c-        hostname: gpu-h800-sz-56c-       │
│            pbn5jhpc-0006                     cf7yedrc-0011           │
│  IP:       29.185.88.166           IP:       29.185.89.104           │
│  GPU:      L20Y(H800) × 8         GPU:      L20Y(H800) × 8         │
│  IB:       mlx5_bond_0~7 200Gb    IB:       mlx5_bond_0~7 200Gb    │
│                                                                     │
│  Node 2 (worker-1)                 Node 3 (worker-2)               │
│  hostname: gpu-h800-sz-56c-        hostname: gpu-h800-sz-56c-       │
│            tf5tbnsa-0000                     ijpt847f-0004           │
│  IP:       29.185.88.237           IP:       29.185.91.125           │
│  GPU:      L20Y(H800) × 8         GPU:      L20Y(H800) × 8         │
│  IB:       mlx5_bond_0~7 200Gb    IB:       mlx5_bond_0~7 200Gb    │
│                                                                     │
│  节点内: NV8 全互联 (NVSwitch)                                       │
│  节点间: RoCE 200Gb × 8 bond (reth0~15 → bond2~9)                   │
│  共享存储: CephFS /apdcephfs/yuanxiaokun-dev (2TB, 全节点可见)       │
│  本地盘:  overlay 14TB                                               │
│  SSH: 节点间免密 ✅                                                   │
│  Hostfile: /etc/taiji/hostfile                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Hostfile (`/etc/taiji/hostfile`)

```
29.185.88.166 slots=8   # Node 0 — launcher (host, 本机)
29.185.89.104 slots=8   # Node 1 — worker-0
29.185.88.237 slots=8   # Node 2 — worker-1
29.185.91.125 slots=8   # Node 3 — worker-2
```

### CPU / NUMA

| 项目 | 值 |
|------|-----|
| CPU | Intel Xeon Platinum 8480+ × 2 socket |
| 核数 | 56 core/socket × 2 HT = 224 logical |
| NUMA 0 | CPU 0-55, 112-167 → GPU 0-3 + NIC 0-3 |
| NUMA 1 | CPU 56-111, 168-223 → GPU 4-7 + NIC 4-7 |

### GPU-NIC 亲和性（`nvidia-smi topo -m`）

每个 GPU 都有一个 PIX 直连的 200Gb RDMA NIC：

| GPU | NIC (PIX) | Bond IP |
|-----|-----------|---------|
| GPU0 | mlx5_bond_0 | 200.21.1.86/30 |
| GPU1 | mlx5_bond_1 | 200.21.2.30/30 |
| GPU2 | mlx5_bond_2 | 200.21.2.230/30 |
| GPU3 | mlx5_bond_3 | 200.21.3.174/30 |
| GPU4 | mlx5_bond_4 | 200.21.4.118/30 |
| GPU5 | mlx5_bond_5 | bond6 |
| GPU6 | mlx5_bond_6 | bond7 |
| GPU7 | mlx5_bond_7 | bond8/bond9 |

---

## 一、硬件/软件审计

### 硬件确认

驱动报 "NVIDIA L20Y, compute_cap 9.0"，实测 NV8 全互联 + 80GB HBM + 132 SM，**确认为 H800 (Hopper 架构)**。

> ⚠️ 注意：原文档写 "compute_cap 8.9" 有误，实测 `torch.cuda.get_device_capability(0)` 返回 `(9, 0)`，说明是 sm_90 架构，csrc 编译应使用 `sm_90a`。

| 组件 | 实际值 | DistCA 要求 | 状态 | 备注 |
|------|--------|-------------|------|------|
| GPU | L20Y(H800) × 8/节点, 4 节点 = 32 | H100/H200 | ✅ | compute_cap 9.0, 132 SM |
| 显存 | 80 GB HBM (81089 MiB) | 80+ GB | ✅ | |
| NVLink | NV8 全互联 (NVSwitch) | NV8 | ✅ | 所有 GPU pair 均为 NV8 |
| 网络 | RoCE 200Gb × 8 (mlx5_bond) | IB/RoCE 200Gb | ✅ | 8 个 bond, 每 GPU 一个 PIX 直连 |
| CUDA | 12.9 (V12.9.86) | 12.8 | ✅ | 向后兼容，sm_90a 支持 |
| Driver | 570.172.08 | — | ✅ | |
| GDRCopy | 2.4 (/opt/gdrcopy) | 需要 | ✅ | IBGDA 依赖 |
| NVSHMEM | 3.4.5 (/usr/local/nvshmem) | 3.2.5 | ✅ | 更高版本向后兼容 |
| OpenMPI | 4.1.7 (/usr/local/openmpi) | 5.0.8 | ⚠️ | **版本偏低，但 NVSHMEM 3.4.5 已预装，可能兼容** |
| **PyTorch** | **2.10.0** | **2.7.0** | ⚠️ | **高 3 个大版本，API 可能不兼容** |
| **NCCL** | **2.27.3** | **2.27.6** | ✅ | 近似匹配 |
| FlashAttention | 2.8.3 + FA3 3.0.0b1 | 2.7.4 | ⚠️ | 版本更高，需测试兼容性 |
| TransformerEngine | 2.10.0 | v2.4 | ⚠️ | **高 6 个版本，API 变化大** |
| Megatron-LM | core 0.13.1 | core_v0.12.1 | ⚠️ | 高 1 个小版本 |
| Apex | **未安装** | 需要 | ❌ | **必须安装** |
| transformers | 4.57.1 | ≥4.40, <4.46 | ❌ | **版本过高，需降级** |
| Python | 3.13.11 | 3.12 | ⚠️ | 高 1 个版本 |

### 关键风险评估

```
风险等级:
  🟢 低风险 — 版本更高且向后兼容 (CUDA, NVSHMEM, NCCL)
  🟡 中风险 — 版本差异可能需要小改 (FlashAttention, Megatron-LM)
  🔴 高风险 — 版本差异大，可能导致 API 不兼容 (PyTorch, TE, transformers)
```

**核心策略：先用现有环境尝试编译运行（Strategy A），失败再创建 conda 降级环境（Strategy B）。**

---

## 二、Strategy A — 基于现有环境（推荐先试）

> **理由**：当前 torch-base 环境已有完整的 CUDA 12.9 + PyTorch 2.10 + TE 2.10 + Megatron 0.13 + FA 2.8 工具链。DistCA 的 Python 代码量仅 ~3000 行，csrc 主要依赖 NVSHMEM + CUDA，对 PyTorch 的依赖是 torch C++ extension 层面，高版本 torch 通常向后兼容。

### Step 0: 工作目录规划

```bash
# 所有操作在共享存储上，全节点可见
export DISTCA_ROOT=/apdcephfs/yuanxiaokun-dev/DistCA
export DISTCA_WORK=/apdcephfs/yuanxiaokun-dev/distca-work
mkdir -p $DISTCA_WORK

# HuggingFace 模型缓存（共享，避免每节点重复下载）
export HF_HOME=/apdcephfs/yuanxiaokun-dev/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME
```

### Step 1: Clone DistCA

```bash
cd /apdcephfs/yuanxiaokun-dev
git clone https://github.com/hao-ai-lab/DistCA.git
cd DistCA
```

### Step 2: 降级 transformers + 安装 Apex

```bash
# transformers 必须降级（DistCA 硬依赖 <4.46 的 API）
pip install 'transformers>=4.40,<4.46'

# 安装 DistCA 本体 + 依赖
pip install -e .
pip install -r requirements.txt

# 安装 Apex（当前环境缺失）
cd /apdcephfs/yuanxiaokun-dev
git clone https://github.com/NVIDIA/apex.git && cd apex
git submodule update --init --recursive
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS=32 pip install -v --no-build-isolation .
cd $DISTCA_ROOT
```

### Step 3: 环境变量配置

```bash
cat > /apdcephfs/yuanxiaokun-dev/DistCA/env.sh << 'ENVEOF'
#!/bin/bash
# ===== DistCA env.sh — 适配 Taiji H800 集群 =====

# -- CUDA (系统已安装 12.9)
export CUDA_DIR=/usr/local/cuda
export CUDA_HOME=$CUDA_DIR
export PATH="$CUDA_HOME/bin:$PATH"

# -- NCCL (从 PyTorch 包内获取)
# 注: torch-base 环境下 NCCL 在系统库中，不在 conda site-packages
export NCCL_HOME=$(python -c "import os,torch;p=os.path.dirname(torch.__file__);print(p)" 2>/dev/null)/lib
# 如上面不对，直接用系统路径：
# export NCCL_HOME=/opt/conda/envs/torch-base/lib

# -- NVSHMEM (系统预装 3.4.5)
export NVSHMEM_PREFIX=/usr/local/nvshmem
export NVSHMEM_DIR=$NVSHMEM_PREFIX
export NVSHMEM_INCLUDE=$NVSHMEM_PREFIX/include

# -- OpenMPI (系统预装 4.1.7)
export OPENMPI_DIR=/usr/local/openmpi
export OPENMPI_INCLUDE=$OPENMPI_DIR/include
export OPENMPI_LIBRARY_DIR=$OPENMPI_DIR/lib

# -- cuDNN (从 conda 环境获取)
export CUDNN_LIB=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__)+'/lib')" 2>/dev/null)
export CUDNN_INCLUDE_DIR=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__)+'/include')" 2>/dev/null)
export CUDNN_INCLUDE=$CUDNN_INCLUDE_DIR
export CUDNN_LIBRARY_PATH=$CUDNN_LIB

# -- GDRCopy (IBGDA 需要)
export GDRCOPY_HOME=/opt/gdrcopy

# -- LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_DIR}/lib64:${OPENMPI_DIR}/lib:${LD_LIBRARY_PATH}"
export PATH="${NVSHMEM_DIR}/bin:${OPENMPI_DIR}/bin:$PATH"

# ===== 运行时环境变量 =====

# NVSHMEM / IBGDA
export NVSHMEM_IB_ENABLE_IBGDA=true
export NVSHMEM_DISABLE_CUDA_VMM=1        # 避免 CUDA VMM 冲突
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# NCCL 优化（RoCE 环境）
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PIX            # GPU 直连 NIC，用 PIX 级别
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_SOCKET_IFNAME=bond1

# 实验控制
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_FA2A_SPLIT_SENDRECV=1
export EXPERIMENT_ADD_SELECTIVE_CKPT=1
export EXPERIMENT_LOG_MEMORY_USAGE=0
export ENABLE_NSYS=0
export EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND=1

# HuggingFace 缓存
export HF_HOME=/apdcephfs/yuanxiaokun-dev/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME
ENVEOF

chmod +x /apdcephfs/yuanxiaokun-dev/DistCA/env.sh
```

### Step 4: 编译 CUDA 扩展 (csrc)

```bash
cd $DISTCA_ROOT
source env.sh

# 编译 DistCA CUDA 扩展（sm_90a for Hopper/H800）
cd csrc
cmake -B build -S ./ -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=90a \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j$(nproc)
cd ..

# 验证编译产物
ls -la distca/runtime/attn_kernels/as_comm*.so
# 应看到 as_comm.cpython-*.so
```

> **如果编译失败**：最可能原因是 NVSHMEM 3.4.5 与 OpenMPI 4.1.7 的 API 不匹配。参见 Step 4.1 Fallback。

#### Step 4.1: 编译 WLB-LLM baseline CUDA 扩展（可选）

```bash
cd $DISTCA_ROOT/baseline/wlbllm_original/csrc
cmake -B build -S ./ -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90a
cmake --build build -j$(nproc)
cd $DISTCA_ROOT
```

### Step 5: 预下载模型权重

```bash
# 在 host 节点下载，CephFS 全节点共享
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ['HF_HOME'] = '/apdcephfs/yuanxiaokun-dev/hf-cache'

# DeepSeek-R1-Distill-Llama-8B（DistCA 默认测试模型）
print('Downloading DeepSeek-R1-Distill-Llama-8B...')
AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    torch_dtype='auto', trust_remote_code=True)
print('Done.')
"
```

> **注意**：如果集群无法直连 HuggingFace，需通过代理或手动传输模型到 `/apdcephfs/yuanxiaokun-dev/hf-cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/`。

---

## 三、Phase 1 — 单卡 Smoke Test

**目标**：验证 DistCA Python 包 + CUDA 扩展 + 模型加载都正常工作。

```bash
cd $DISTCA_ROOT && source env.sh

# 单卡极简测试：2 层, 1024 tokens, 无并行
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
python pretrain_llama.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --num-layers 2 \
  --num-tokens 1024 \
  --num-gpus-per-node 1 \
  --tp-size 1 --pp-size 1 --cp-size 1 \
  --max-sample-id 2
```

**预期结果**：
- 无 import error
- 模型加载成功（从 HF cache）
- 2 个 sample 的 forward + backward 完成
- 显存占用 < 10 GB

**常见问题排查**：

| 错误 | 原因 | 修复 |
|------|------|------|
| `ImportError: cannot import name 'xxx' from 'transformers'` | transformers 版本过高 | `pip install 'transformers>=4.40,<4.46'` |
| `ModuleNotFoundError: No module named 'apex'` | Apex 未安装 | 见 Step 2 安装 Apex |
| `as_comm*.so not found` | csrc 未编译或编译失败 | 见 Step 4 |
| `NVSHMEM init failed` | IBGDA 不支持 | 设 `NVSHMEM_IB_ENABLE_IBGDA=false` 重试 |
| `CUDA error: no kernel image is available` | sm 架构不匹配 | 确认用 `sm_90a` 编译 |

---

## 四、Phase 2 — 单节点 8 GPU (TP=8)

**目标**：验证 NVLink 域内的 Tensor Parallel + DistCA 调度。

```bash
cd $DISTCA_ROOT && source env.sh

# TP=8 单节点 — 逐步增加序列长度
for NUM_TOKENS in 16384 65536 131072; do
  echo "=== Testing NUM_TOKENS=$NUM_TOKENS ==="
  torchrun --nproc_per_node=8 --master_port=29500 \
    pretrain_llama.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --num-layers 4 \
    --num-tokens $NUM_TOKENS \
    --num-gpus-per-node 8 \
    --tp-size 8 --pp-size 1 --cp-size 1 \
    --max-sample-id 5 \
    --use-planner \
    --sample-name wlbllm
  echo "=== Done NUM_TOKENS=$NUM_TOKENS ==="
done
```

**预期**：
- 16384 tokens：应顺利完成
- 65536 tokens：开始看到 DistCA planner 输出调度方案
- 131072 tokens：可能需要 selective checkpointing (`EXPERIMENT_ADD_SELECTIVE_CKPT=1`)

**关键观察**：
- `nvidia-smi` 验证 8 卡均被使用
- 观察显存峰值（131072 tokens 可能接近 80GB 限制）
- 确认 NVLink 带宽被充分利用（`nvidia-smi nvlink -i 0 -sc`）

---

## 五、Phase 3 — 多节点 32 GPU

### 5.1 网络连通性验证

```bash
# 已验证 SSH 免密连接 ✅
# 验证所有节点的 GPU 和软件环境一致性
for ip in 29.185.89.104 29.185.88.237 29.185.91.125; do
  echo "=== Node $ip ==="
  ssh $ip "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1; \
           python -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')\"; \
           ls /apdcephfs/yuanxiaokun-dev/DistCA/distca/runtime/attn_kernels/as_comm*.so 2>/dev/null && echo 'csrc OK' || echo 'csrc MISSING'"
done
```

### 5.2 NCCL 通信基准测试

```bash
# 先跑 NCCL all-reduce 测试确认节点间带宽
# 如果有 nccl-tests:
mpirun -np 32 --hostfile /etc/taiji/hostfile \
  -mca btl_tcp_if_include bond1 \
  --map-by ppr:8:node \
  /path/to/all_reduce_perf -b 8 -e 1G -f 2 -g 1

# 或用 PyTorch 简单测试
torchrun --nnodes=4 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=29.185.88.166:29500 \
  -c "
import torch, torch.distributed as dist, os, time
dist.init_process_group('nccl')
rank = dist.get_rank()
device = torch.device(f'cuda:{rank % 8}')
x = torch.randn(1024, 1024, device=device)
dist.barrier()
t0 = time.time()
for _ in range(100):
    dist.all_reduce(x)
torch.cuda.synchronize()
t1 = time.time()
if rank == 0:
    print(f'AllReduce 100 iters: {t1-t0:.3f}s, avg: {(t1-t0)/100*1000:.2f}ms')
dist.destroy_process_group()
"
```

### 5.3 DistCA 3D 并行 (TP=8, CP=1, DP=4)

```bash
cd $DISTCA_ROOT && source env.sh

export MASTER_ADDR=29.185.88.166
export MASTER_PORT=29500

# --- 方案 A: 使用 torchrun (推荐) ---
# 在每个节点上分别运行（或用 pdsh/clush 批量执行）

# Node 0 (host):
torchrun --nnodes=4 --nproc_per_node=8 \
  --node_rank=0 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  pretrain_llama.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --num-layers 32 --num-tokens 65536 \
  --num-nodes 4 --num-gpus-per-node 8 \
  --tp-size 8 --pp-size 1 --cp-size 1 \
  --max-sample-id 5 --use-planner --sample-name wlbllm

# Node 1-3: SSH 到对应节点执行同样命令，只改 --node_rank=1/2/3

# --- 方案 B: 辅助脚本批量启动 ---
cat > /apdcephfs/yuanxiaokun-dev/DistCA/launch_multinode.sh << 'LAUNCHEOF'
#!/bin/bash
# Usage: bash launch_multinode.sh <script_args...>
set -e
export MASTER_ADDR=29.185.88.166
export MASTER_PORT=29500
NNODES=4
NPROC=8
HOSTLIST=(29.185.88.166 29.185.89.104 29.185.88.237 29.185.91.125)

for i in "${!HOSTLIST[@]}"; do
  HOST=${HOSTLIST[$i]}
  echo "Launching node $i on $HOST..."
  if [ "$HOST" == "$MASTER_ADDR" ]; then
    # 本机前台运行（最后一个启动也行，这里选前台以看日志）
    cd /apdcephfs/yuanxiaokun-dev/DistCA && source env.sh && \
    torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
      --node_rank=$i \
      --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      "$@" &
  else
    ssh $HOST "cd /apdcephfs/yuanxiaokun-dev/DistCA && source env.sh && \
    torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
      --node_rank=$i \
      --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      $*" &
  fi
done
wait
echo "All nodes finished."
LAUNCHEOF
chmod +x /apdcephfs/yuanxiaokun-dev/DistCA/launch_multinode.sh

# 使用启动脚本
bash launch_multinode.sh pretrain_llama.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --num-layers 32 --num-tokens 65536 \
  --num-nodes 4 --num-gpus-per-node 8 \
  --tp-size 8 --pp-size 1 --cp-size 1 \
  --max-sample-id 5 --use-planner --sample-name wlbllm
```

### 5.4 DistCA 4D 并行 (TP=8, PP=2, DP=2)

```bash
cd $DISTCA_ROOT && source env.sh

# 4D 并行：加入 Pipeline Parallelism
bash launch_multinode.sh pretrain_llama.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --num-layers 32 --num-tokens 131072 \
  --num-nodes 4 --num-gpus-per-node 8 \
  --tp-size 8 --pp-size 2 --num-microbatch 4 \
  --max-sample-id 10 --use-planner --sample-name wlbllm
```

### 5.5 序列长度递增测试

```bash
# 在 3D 配置下逐步增加序列长度
for NUM_TOKENS in 32768 65536 131072 262144; do
  echo "========== NUM_TOKENS=$NUM_TOKENS =========="
  export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$((NUM_TOKENS > 131072 ? 4 : 2))
  bash launch_multinode.sh pretrain_llama.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --num-layers 32 --num-tokens $NUM_TOKENS \
    --num-nodes 4 --num-gpus-per-node 8 \
    --tp-size 8 --pp-size 1 --cp-size 1 \
    --max-sample-id 5 --use-planner --sample-name wlbllm
done
```

> **注意**：262144 tokens (256K) 在 80GB × 32 GPU 下可能需要增大 NVSHMEM buffer 到 4GB 并启用 selective checkpointing。

---

## 六、完整实验矩阵

### 6.1 主实验（吞吐对比）

| # | 配置 | Mode | SeqLen | Layers | Batch | 目的 | 预计显存 |
|---|------|------|--------|--------|-------|------|----------|
| E1 | TP=8, PP=1, DP=4 | **distca** | 64K | 32 | 2 | 3D DistCA 吞吐 | ~50 GB/GPU |
| E2 | TP=8, PP=1, DP=4 | **wlbllm** | 64K | 32 | 2 | 3D WLB-LLM baseline | ~50 GB/GPU |
| E3 | TP=8, PP=1, DP=4 | **distca** | 128K | 32 | 1 | 3D 长序列 | ~65 GB/GPU |
| E4 | TP=8, PP=1, DP=4 | **wlbllm** | 128K | 32 | 1 | 3D baseline 长序列 | ~65 GB/GPU |
| E5 | TP=8, PP=2, DP=2 | **distca** | 64K | 32 | 4 (mb=4) | 4D DistCA 吞吐 | ~45 GB/GPU |
| E6 | TP=8, PP=2, DP=2 | **wlbllm** | 64K | 32 | 4 (mb=4) | 4D baseline | ~45 GB/GPU |
| E7 | TP=8, PP=2, DP=2 | **distca** | 128K | 32 | 2 (mb=2) | 4D 长序列 | ~60 GB/GPU |
| E8 | TP=8, PP=2, DP=2 | **wlbllm** | 128K | 32 | 2 (mb=2) | 4D baseline 长序列 | ~60 GB/GPU |

### 6.2 消融实验

| # | 基于 | 变更 | 目的 |
|---|------|------|------|
| A1 | E1 | `ENABLE_NSYS=1`, `MAX_SAMPLE_ID=3` | Nsight 时间线分析 Ping-Pong 重叠 |
| A2 | E1 | `EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=1` | 通信消融（信号通信） |
| A3 | E1 | `EXPERIMENT_D2_BALANCE_PING_PONG=1` | Ping-Pong 负载均衡消融 |
| A4 | E5 | `ENABLE_NSYS=1` | 4D 模式下的通信时间线 |

### 6.3 灵敏度分析

| 变量 | 扫描范围 | 基于配置 |
|------|---------|---------|
| 序列长度 | 32K → 64K → 128K → 256K | E1 (3D) |
| NVSHMEM Buffer | 1 / 2 / 4 GB | E3 (128K) |
| PP degree | 1 / 2 / 4 | TP=8, 128K |
| Batch size | 1 / 2 / 4 | E5 (4D) |
| 选择性 Checkpointing | on / off | E3 (显存压力测试) |

---

## 七、运行时调优指南

### 7.1 NCCL 环境变量（RoCE 环境）

```bash
# 基础配置
export NCCL_IB_DISABLE=0                   # 启用 IB/RoCE
export NCCL_NET_GDR_LEVEL=PIX              # 每个 GPU 有 PIX 直连 NIC
export NCCL_SOCKET_IFNAME=bond1            # 控制面走 bond1 (29.x 网段)
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7

# 调优
export NCCL_IB_GID_INDEX=3                 # RoCEv2
export NCCL_IB_TIMEOUT=23                  # 超时设大
export NCCL_IB_RETRY_CNT=7                 # 重试次数

# Debug（排查问题时开启）
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
```

### 7.2 NVSHMEM / IBGDA 配置

```bash
export NVSHMEM_IB_ENABLE_IBGDA=true        # GPU-Direct Async（H800 支持）
export NVSHMEM_DISABLE_CUDA_VMM=1          # 避免 CUDA VMM 冲突
export NVSHMEM_IB_GID_INDEX=3              # RoCEv2 GID index

# 如果 IBGDA 初始化失败，降级到非 IBGDA：
# export NVSHMEM_IB_ENABLE_IBGDA=false
# export NVSHMEM_REMOTE_TRANSPORT=ibrc
```

### 7.3 CPU 亲和性

```bash
# 每 GPU 进程绑定到对应 NUMA node
# GPU 0-3 → NUMA 0 (CPU 0-55, 112-167)
# GPU 4-7 → NUMA 1 (CPU 56-111, 168-223)
# DistCA 需要每进程至少 3 核 (PyTorch + NVSHMEM + NCCL)

# torchrun 自动处理 LOCAL_RANK → GPU 映射
# 如需手动：
# numactl --cpunodebind=0 --membind=0 python ... (for GPU 0-3)
# numactl --cpunodebind=1 --membind=1 python ... (for GPU 4-7)
```

### 7.4 显存优化

```bash
# 选择性激活 Checkpointing（CA 层重计算换显存）
export EXPERIMENT_ADD_SELECTIVE_CKPT=1

# NVSHMEM 对称堆大小（根据序列长度调整）
# 64K tokens → 2 GB
# 128K tokens → 2-4 GB
# 256K tokens → 4-8 GB
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2

# PyTorch 显存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 八、Strategy B — 降级 Conda 环境（Fallback）

> 仅在 Strategy A 编译失败或运行时出现不可调和的 API 兼容性问题时使用。

```bash
# 在共享存储上创建环境（全节点可见）
conda create -p /apdcephfs/yuanxiaokun-dev/conda-envs/distca python=3.12 -y
conda activate /apdcephfs/yuanxiaokun-dev/conda-envs/distca

# 精确版本锁定
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 注意：CUDA 12.9 系统 + cu128 wheel 通常兼容

# TransformerEngine v2.4（源码编译 ~30min）
pip install pybind11
git clone https://github.com/NVIDIA/TransformerEngine.git /apdcephfs/yuanxiaokun-dev/TransformerEngine
cd /apdcephfs/yuanxiaokun-dev/TransformerEngine && git checkout v2.4
git submodule update --init --recursive
NVTE_FRAMEWORK=pytorch MAX_JOBS=32 pip install --no-build-isolation -v '.[pytorch]'

# Megatron-LM core_v0.12.1
git clone https://github.com/NVIDIA/Megatron-LM.git /apdcephfs/yuanxiaokun-dev/Megatron-LM
cd /apdcephfs/yuanxiaokun-dev/Megatron-LM && git checkout core_v0.12.1
git submodule update --init --recursive && pip install -e .

# Apex
git clone https://github.com/NVIDIA/apex.git /apdcephfs/yuanxiaokun-dev/apex
cd /apdcephfs/yuanxiaokun-dev/apex && git submodule update --init --recursive
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS=32 pip install -v --no-build-isolation .

# FlashAttention 2.7.4（精确版本 wheel）
pip install flash-attn==2.7.4 --no-build-isolation

# transformers (精确版本)
pip install 'transformers>=4.40,<4.46'

# DistCA 本体
cd /apdcephfs/yuanxiaokun-dev/DistCA
pip install -e . && pip install -r requirements.txt

# 编译 csrc
source env.sh
cd csrc && cmake -B build -S ./ -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90a && cmake --build build
```

---

## 九、监控与 Profiling

### 9.1 实时 GPU 监控

```bash
# 在 host 节点监控所有 4 个节点
for ip in 29.185.88.166 29.185.89.104 29.185.88.237 29.185.91.125; do
  echo "=== $ip ==="; ssh $ip nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
done

# 持续监控（每 5 秒）
watch -n 5 'for ip in 29.185.88.166 29.185.89.104 29.185.88.237 29.185.91.125; do echo "=== $ip ==="; ssh $ip nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done'
```

### 9.2 Nsight Systems Profiling

```bash
export ENABLE_NSYS=1
export MAX_SAMPLE_ID=3   # 只需几个 iteration

# 运行实验 A1
bash launch_multinode.sh pretrain_llama.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --num-layers 32 --num-tokens 65536 \
  --num-nodes 4 --num-gpus-per-node 8 \
  --tp-size 8 --pp-size 1 --cp-size 1 \
  --max-sample-id 3 --use-planner --sample-name wlbllm

# 分析 .nsys-rep 文件
# 重点看：All-to-All 通信 vs FlashAttention 计算的重叠度
nsys stats <output>.nsys-rep
```

### 9.3 显存分析

```bash
export EXPERIMENT_LOG_MEMORY_USAGE=1
export EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP=1

# 运行后查看日志中的显存追踪输出
# 关注：激活内存不均衡比, 峰值显存, NVSHMEM 堆占用
```

---

## 十、预期结果与成功标准

### Phase 1 (单卡)
- [x] import 无错误
- [ ] 2 层 forward+backward 完成
- [ ] 无 OOM

### Phase 2 (8 GPU)
- [ ] TP=8 正确分片
- [ ] Planner 输出调度日志
- [ ] 16K tokens 完成
- [ ] 65K tokens 完成
- [ ] 131K tokens 完成（或 OOM 时启用 checkpointing 后完成）

### Phase 3 (32 GPU)
- [ ] 4 节点通信正常
- [ ] 3D 并行 (E1/E2) 完成，获得 distca vs wlbllm 吞吐对比
- [ ] 4D 并行 (E5/E6) 完成
- [ ] DistCA 相对 WLB-LLM 有可观测的加速（论文 4 节点预期 ~10-15%）

### 性能基准（参考）
- 8B, 64K, 32 GPU, 3D: 预期 **X tokens/s**（待实测填入）
- 8B, 128K, 32 GPU, 4D: 预期 **Y tokens/s**（待实测填入）
- DistCA 加速比: 预期 **1.10~1.20×**（4 节点规模，参考论文 Figure）

---

## 十一、故障排查速查表

| 症状 | 可能原因 | 排查/修复 |
|------|---------|---------|
| `NVSHMEM: bootstrap init failed` | OpenMPI 版本不兼容 | 尝试 `NVSHMEM_BOOTSTRAP=PMI2` 或安装 OpenMPI 5.0.8 |
| `NCCL timeout` | 节点间网络不通 | 检查 `NCCL_SOCKET_IFNAME=bond1`, 用 `nccl-tests` 验证 |
| `CUDA error: out of memory` | 显存不足 | 减少 `num-tokens`, 启用 `EXPERIMENT_ADD_SELECTIVE_CKPT=1`, 减少 `EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB` |
| `ImportError` (transformers) | transformers 版本不对 | `pip install 'transformers>=4.40,<4.46'` |
| `as_comm.so: undefined symbol` | NVSHMEM ABI 不匹配 | 重新编译 csrc, 确认 `NVSHMEM_PREFIX` 正确 |
| `IBGDA init failed` | GDRCopy 或驱动不支持 | `NVSHMEM_IB_ENABLE_IBGDA=false` 降级 |
| `torch.distributed init failed` | master 地址/端口错误 | 确认 `MASTER_ADDR=29.185.88.166`, 端口未被占用 |
| `hung at barrier` | 部分 rank 未启动 | 确认所有节点都执行了 torchrun |
| Apex 编译失败 | GCC/nvcc 版本 | 用 `pip install apex` (pure Python fallback) |
| 性能远低于预期 | NCCL 走了 TCP 而非 RDMA | 开 `NCCL_DEBUG=INFO` 查看传输层 |

---

## 关键概念

- [[CAD-DistCA]]
- [[Dynamic CP]]
- [[2026-04-25-distca-repo-analysis|DistCA 仓库深度剖析]]
- [[FlashAttention]]
- [[长上下文训练]]
- [[负载均衡与变长序列]]
