---
tags: [素材摘要, 训练系统, 长上下文, 部署指南, 实验方案]
created: 2026-04-27
updated: 2026-04-27
source_type: 实验方案
source_path: https://github.com/hao-ai-lab/DistCA
---

# DistCA 部署实验方案 — "L20Y"(H800) × 32 GPU + 256K

## 硬件确认

驱动报 "NVIDIA L20Y, compute_cap 8.9"，但实测 **FP16 725.6 TFLOPS + NV8 全互联**，确认为中国合规版 H800。

| 组件 | 值 | 状态 |
|------|-----|------|
| GPU | "L20Y"(实为 H800) × 8/节点，4 节点 | ✅ |
| FP16 | 725.6 TFLOPS（实测） | ✅ |
| 显存 | 80 GB HBM | ✅ |
| NVLink | NV8 全互联 | ✅ |
| IB | 200 Gbps × 8 (mlx5_bond) | ✅ |
| CUDA | 12.9 | ✅ |
| PyTorch | 2.10.0 | ⚠️ DistCA 要求 2.7.0 |
| NVSHMEM | 3.4.5 | ✅ |
| OpenMPI | 4.1.7 | ⚠️ 要求 5.0.8 |

## Phase 1: 单卡 Smoke Test

```bash
# 1. Clone
cd /workspace && git clone https://github.com/hao-ai-lab/DistCA.git && cd DistCA

# 2. 安装
pip install -e . && pip install -r requirements.txt

# 3. 环境变量
cp env.template.sh env.sh  # 编辑 CUDA_DIR, NVSHMEM_PREFIX, OPENMPI_DIR
source env.sh

# 4. 编译 CUDA 扩展（先试 sm_90a，失败改 sm_89）
cd csrc && cmake -B build -S ./ -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90a && cmake --build build && cd ..

# 5. 安装 TransformerEngine + Megatron-LM + Apex + FlashAttention（见完整步骤）

# 6. 单卡测试
export NVSHMEM_IB_ENABLE_IBGDA=true  # 先试 true
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
python pretrain_llama.py --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --num-layers 2 --num-tokens 1024 --num-gpus-per-node 1 \
  --tp-size 1 --pp-size 1 --cp-size 1 --max-sample-id 2
```

## Phase 2: 单节点 8 GPU (TP=8)

```bash
torchrun --nproc_per_node=8 pretrain_llama.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --num-layers 4 --num-tokens 16384 \
  --num-gpus-per-node 8 --tp-size 8 --pp-size 1 \
  --max-sample-id 5 --use-planner
# 然后逐步增加: 16384 → 65536 → 131072 → 262144
```

## Phase 3: 多节点 32 GPU (TP=8, PP=2, DP=2)

```bash
torchrun --nnodes=4 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=<master_ip>:29500 \
  pretrain_llama.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --num-layers 32 --num-tokens 262144 \
  --num-nodes 4 --num-gpus-per-node 8 \
  --tp-size 8 --pp-size 2 --num-microbatch 4 \
  --max-sample-id 10 --use-planner --sample-name wlbllm
```

## 实验矩阵

| # | 配置 | Mode | 目的 |
|---|------|------|------|
| 1 | TP=8, PP=1, DP=4 | distca | 3D 吞吐 |
| 2 | TP=8, PP=1, DP=4 | wlbllm | 3D baseline |
| 3 | TP=8, PP=2, DP=2 | distca | 4D 吞吐 |
| 4 | TP=8, PP=2, DP=2 | wlbllm | 4D baseline |
| 5 | #1 + ENABLE_NSYS=1 | distca | Ping-Pong profiling |
| 6 | #1 + signal-only | distca | 通信消融 |

## 关键概念

- [[CAD-DistCA]]
- [[Dynamic CP]]
- [[2026-04-25-distca-repo-analysis|DistCA 仓库深度剖析]]
