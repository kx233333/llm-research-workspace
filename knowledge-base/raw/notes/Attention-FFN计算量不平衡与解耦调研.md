---
title: "Attention vs FFN 计算量不平衡：估算模型、解耦方案与显存分析"
date: 2026-04-21
tags:
  - LLM推理
  - LLM训练
  - 负载均衡
  - Attention
  - FFN
  - 流水线并行
  - 解耦推理
  - FLOPs
  - 显存优化
status: 整理中
---

# Attention vs FFN 计算量不平衡：估算模型、解耦方案与显存分析

> 调研问题：多 GPU 并行（流水线/张量/序列并行）中，如何估算 Attention（$O(s^2 h)$）与 FFN（$O(s h^2)$）的计算量？各方案如何应对二者不平衡？将它们显式解耦是否更合适？显存如何分布？

---

## 一、基础：每层计算量公式

### 1.1 标准 Transformer 层 FLOPs（Prefill 阶段）

设 batch size $B$，序列长度 $s$，hidden dim $h$，FFN 中间维度 $d_{ffn} = 4h$：

```
组件              FLOPs（B × 序列）
────────────────────────────────────────────
QKV Projection    2B·s·3h·h = 6Bsh²
Attn Scores (Q·Kᵀ) 2B·s·s·h = 2Bs²h
Attn Weighted Sum  2B·s·s·h = 2Bs²h
Output Projection  2B·s·h·h = 2Bsh²
─────────────────────────────────────────
Attention 总计    8Bsh² + 4Bs²h
─────────────────────────────────────────
FFN up/gate/down   2 × 2B·s·h·4h = 16Bsh²
（SwiGLU 三矩阵）  ≈ 12Bsh²（含 gate，实际取决于架构）
─────────────────────────────────────────
FFN 总计          ~8–12 Bsh²
```

**简化记忆**（常见近似，去掉常数因子）：

$$
\text{FLOPs}_{\text{Attn}} \approx 4Bsh^2 + 4Bs^2h
\qquad
\text{FLOPs}_{\text{FFN}} \approx 8Bsh^2
$$

### 1.2 计算量交叉点

二者大小相等时：

$$
4Bs^2h = 8Bsh^2 \implies s = 2h
$$

| 序列长度 vs hidden dim | 主导模块 | 直觉 |
|----------------------|---------|------|
| $s \ll h$（短序列） | FFN 主导 | $Bsh^2 \gg Bs^2h$ |
| $s \approx 2h$ | 持平 | 对于 $h=4096$，$s\approx 8K$ 时交叉 |
| $s \gg 2h$（长序列）| Attention 主导 | $s^2h$ 项爆发，FFN 相对可忽略 |

**数值示例**（$h=4096$，每层，每 batch token）：

| 序列长度 $s$ | $4Bs^2h$ | $8Bsh^2$ | Attn/FFN 比 |
|------------|---------|---------|------------|
| 512 | 0.004T | 0.134T | **0.03×**（FFN 绝对主导）|
| 4K | 0.26T | 0.134T | **2.0×**（Attention 开始主导）|
| 32K | 16.7T | 0.134T | **125×**（Attention 完全压制）|

> 这是流水线并行中"一刀切"按层分配无法做到计算平衡的根本原因：**不同序列长度下同一层的 FLOPs 比值变化超过 3 个数量级**。

---

## 二、Decode 阶段的本质差异

Decode 时每步只生成 1 个新 token，但需访问全部历史 KV cache（长度 $c$）：

```
阶段          主导瓶颈        延迟正比于
─────────────────────────────────────────────
Attention     内存带宽受限   KV cache 读取量 ∝ B·c·h
FFN           计算受限       矩阵乘 GEMM    ∝ B·h²
```

这是更深层的不对称：

| 特性 | Attention（Decode） | FFN（Decode） |
|------|-------------------|--------------|
| 瓶颈 | **Memory-bound**（HBM 带宽）| **Compute-bound**（TFLOPS）|
| 延迟随序列增长 | 线性 $O(c)$ | 与序列无关 |
| 延迟随 batch 增长 | 线性（直到带宽饱和）| 线性（GEMM efficiency ↑）|
| 理想硬件 | 高 HBM 带宽（H20、L20）| 高 TFLOPS（H800、L40S）|
| KV Cache 存储 | **需要**，$O(B \cdot c \cdot h)$ | 无 |

---

## 三、现有方案中的计算量估算与均衡策略

### 3.1 传统流水线并行（忽视 Attn/FFN 差异）

**做法**：按 Transformer **层数**均匀分配到各流水线阶段，每阶段持有相同数量层。

**FLOPs 估算方式**：
- 假设各层计算量相同（实际仅在 $s \ll h$ 时成立）
- 实践中以每层参数量为代理（参数量均匀 ≈ 计算量均匀）

**问题**：
- 序列长度变化时各阶段实际 FLOPs 差异显著（Attention 层受 $s^2$ 影响，FFN 不受）
- 长上下文下前几层（含大量 Attention）与后几层实际 throughput 差异大
- 反向传播的 Attention 阶段需 2× forward compute，FFN 也是，但二者 ratio 相同，相对均衡性不变

---

### 3.2 ChunkFlow（序列切分 + 状态感知调度）

> [[ChunkFlow - Efficient Long Context Fine-tuning with Chunk Flow]]（arXiv:2503.02356）

**核心思路**：不按层分，而按固定大小的 Chunk 切分序列，每个 chunk 约 ChunkSize tokens。

**计算量估算方式**：
- 以 **ChunkSize 为单位**，每个 chunk 内 Attention FLOPs ≈ $4 \cdot \text{CS}^2 \cdot h$（ChunkSize = CS）
- 通过控制 chunk 大小使各步计算量近似恒定，而非控制层数
- 核心等式：$\text{CS} \times K$ = 显存上限内可保存的激活量

**对不平衡的处理**：
- chunk 粒度下，Attention 的 $s^2$ 不再依赖原始序列长度，而仅依赖 CS（可控）
- 长序列不会导致某些流水线阶段的 Attention 异常膨胀
- 剩余 KV state 用"两次 forward + KV 复用"换取线性显存（额外计算而非额外带宽）

**对 Attn/FFN 是否显式解耦**：❌ 未解耦，二者仍在同一阶段运行，只是通过 chunk 使得二者在每步的计算量都可预测。

---

### 3.3 WLB-LLM（4D 并行 + 变长文档均衡）

> arXiv:2503.17924，TLDR：变长文档打包 + per-document sharding，1.23× 加速

**核心思路**：在微 batch 层面识别 Attention 计算量（$\propto s^2$）的差异，通过 document packing 使各流水线阶段的 Attention 负载趋同。

**FLOPs 估算方式**：
- 对每个文档 $i$，其 Attention FLOPs ∝ $s_i^2 \cdot h$（可在调度前预计算）
- 构造 bin-packing 问题，目标是各 stage 的 $\sum s_i^2$ 相等（而非 $\sum s_i$ 相等）

**关键洞察**：Attention 导致的不平衡来自 $\sum s_i^2$，而非 $\sum s_i$（token 数）。按 token 数均衡会低估长文档的真实负载。

---

### 3.4 MegaScale-Infer（Attention-FFN 显式解耦）

> [[MegaScale-Infer]]（arXiv:2504.02263，ByteDance，25 citations）

**做法**：在每个 MoE 层内，将 Attention 节点与 Expert（FFN）节点**物理分离**，独立配置并行度和硬件。

**FLOPs/延迟估算模型**（Decode 阶段）：

```
T_a = k₁ · b_a + k₂        （attention 节点，b_a = micro-batch 大小）
T_e = k₃ · b_e + k₄        （expert 节点，b_e = dispatch 后的每专家 token 数）
```

其中 $k_i$ 通过 profiling + 线性回归获得（而非解析推导）。

**平衡条件**（constraint 1）：

$$
T_a \approx T_e
$$

→ 需要的 attention 节点数 $n_a = \frac{k_1 E}{k_3 K}$（E = 专家总数，K = Top-K 路由数）

**Ping-Pong Pipeline**：将 batch 分成 $m$ 个 micro-batch，在 attention 节点和 expert 节点之间交替流转，使通信与计算重叠。

最小 micro-batch 数：

$$
m \geq 2 \times \left(1 + \frac{T_c}{T_f}\right)
$$

**核心结论**：
- Decode 阶段 attention 是内存密集（适合 H20 类高带宽 GPU），FFN/Expert 是计算密集（适合 L40S 类高 TFLOPS GPU）
- 解耦后异构部署：H20（attention）+ L40S（expert），**per-cost throughput 提升 1.7×**

---

### 3.5 AFD 最优比例理论（解耦 + 解析建模）

> arXiv:2601.21351，TLDR：推导 Attention:FFN 实例最优比例 $r^*$ 的闭合解

**拓扑**：$r$ 个 Attention 实例 → 1 个 FFN 实例（rA-1F）

**延迟模型**（线性近似，由 roofline model 推导）：

| 组件 | 延迟公式 | 物理含义 |
|------|---------|---------|
| Attention | $t_A(T) = \alpha_A T + \beta_A$ | $T$ = KV cache 总 token 数，内存带宽受限 |
| FFN | $t_F(rB) = \alpha_F(rB) + \beta_F$ | $rB$ = 总 batch 大小，计算受限 |
| 通信 | $t_C(B) = \alpha_C B + \beta_C$ | activation transfer，带宽受限 |

**目标**：最大化 throughput per instance：

$$
\text{Throughput}_\text{per-inst}(B; r) = \frac{1}{r+1} \cdot \frac{rB}{\tau(B;r)}
$$

其中 $\tau = \max\{t_A, t_C, t_F\}$（最慢者决定步长）。

**三种工作区间**：

| 区间 | 瓶颈 | 最优 $r$ 策略 |
|------|------|-------------|
| Attention-bottleneck | $t_A \geq t_C, t_F$ | 增大 $r$（多加 Attention 实例）|
| Communication-bottleneck | $t_C \geq t_A, t_F$ | 增大 $r$，直到 $r_C$ |
| FFN-bottleneck | $t_F \geq t_A, t_C$ | 存在最优峰值 $r_{\text{peak}}$ |

**闭合解**（Theorem 4.4）：

$$
\boxed{r^* = \max\left\{
\underbrace{\frac{\alpha_A \bar{T} + \beta_A - \beta_F}{\alpha_F B}}_{\text{Attn-bottleneck 边界}},\;
\underbrace{\frac{\bar{t}_C - \beta_F}{\alpha_F B}}_{\text{Comm-bottleneck 边界}},\;
\underbrace{\sqrt{\frac{\beta_F}{\alpha_F B}}}_{\text{FFN-bottleneck 峰值}}
\right\}}
$$

其中 $\bar{T} = B(\mu_P + \mu_D)$ 是每个 Attention 实例的平均 KV cache 负载。

**实验验证**（DeepSeek-V3，Ascend 910C）：
- $B=256$, $\mu_D=500$, $\mu_P=100$ → 理论最优 $r^* \approx 9.3$，与仿真最优误差 <10%
- $r$ 较大时（$r>16$）仿真值低于理论（straggler effect，各 Attention 实例间负载不均）

**关键发现**：$r^*$ 随 batch size 增大和上下文增长而增大（需要更多 Attention 实例）：

| 工作点 | batch=128 | batch=512 |
|--------|----------|----------|
| 最优 $r^*$ | ~7.1 | ~10.3 |

---

## 四、Decoupling Attention 与 FFN 是否更合适？

### 4.1 理论层面：应该解耦

| 维度 | 耦合（传统按层分配） | 解耦（Attn/FFN 分离）|
|------|------------------|------------------|
| 计算量估算 | 以层数 or token 数为代理，忽视 $s^2$ vs $sh^2$ 差异 | 分别建模 $T_a = f(\text{KV cache})$，$T_e = f(\text{batch})$，更精确 |
| 硬件匹配 | 同一 GPU 需兼顾内存带宽 + TFLOPS | 异构部署：高 BW GPU → Attention，高 TFLOPS GPU → FFN |
| 扩展粒度 | 扩展单位是整个 transformer 层 | 可以独立扩展 $n_a$ 个 Attention 实例而不改变 FFN 数量 |
| 序列长度敏感性 | 长序列时 Attention 成本骤增，与 FFN 阶段形成气泡 | Attention 实例数可根据 $\bar{T}$ 动态调整 |
| MoE 场景 | 专家路由偏斜导致 FFN 负载不均，Attention 稳定 | 可单独给 FFN 做 load balancing，不影响 Attention |

### 4.2 实践层面：代价与限制

| 挑战 | 说明 |
|------|------|
| 通信开销 | Attention → FFN 需要传输 intermediate activation（$B \times h$ per step），要求 $T_c < T_f$ |
| 工程复杂度 | 需要新的通信库（如 MegaScale-Infer 的 M2N library，替代 NCCL，减少 CPU 拷贝和同步）|
| 适用场景限制 | Decode 阶段收益显著（内存 vs 计算 差异大）；Prefill 阶段两者都是 compute-bound，差异相对小 |
| 最优 $r$ 的估算难度 | $r^*$ 依赖 $\alpha_A, \alpha_F$（需 profiling），且随工作负载变化（context length 分布变则 $r^*$ 变）|

### 4.3 结论

**Decode 阶段强烈建议解耦**：
- Attention（memory-bound）与 FFN（compute-bound）的瓶颈完全不同
- 理论最优 $r^*$ 通常在 7–12 之间（视上下文长度和 batch size），远非 1:1
- MegaScale-Infer 实测 per-GPU throughput 提升 1.9×，per-cost throughput 提升 1.7×

**Prefill 阶段可选解耦**：
- 二者均为 compute-bound，但 Attention 随序列增长更快
- 长上下文（$s > 2h$）时解耦收益仍显著
- ChunkFlow / WLB-LLM 的思路（在不解耦的前提下通过精细调度减少气泡）是更轻量的替代方案

---

## 五、显存分布差异

### 5.1 参数显存（Model Weights）

每层参数量（典型 $h=4096$, $d_{ffn}=4h$）：

```
Attention 参数：
  W_Q, W_K, W_V, W_O = 4 × h² = 4 × 4096² ≈ 67 M 参数
  MHA with GQA（g=8）：≈ h²(1 + 2/g + 1) = h²(2 + 1/4) ≈ 37 M 参数

FFN 参数（SwiGLU）：
  W_gate, W_up, W_down = 3 × h × 4h = 12h² ≈ 201 M 参数

FFN / Attention 参数比 ≈ 3–5×
```

对于 **MoE 模型**（如 DeepSeek-V3，E=256 个专家，Top-K=8）：
- FFN 参数 × E = 极度膨胀，但每步只激活 K 个 → 参数量悬殊更大
- Attention 参数量不受 MoE 影响

### 5.2 激活显存（Activation Memory，Training/Prefill）

每层激活显存（保存用于反向传播）：

```
Attention 激活：
  - QKV: 3 × B × s × h（= 3Bsh float16 = 6Bsh bytes）
  - Attention matrix (Q·Kᵀ): B × num_heads × s × s（= Bh_s s² 字节，不用 FlashAttn 时）
  - FlashAttention: 不存 attention matrix，仅存 LSE（B × num_heads × s）

FFN 激活：
  - 中间激活: B × s × 4h × 2（gate + up）
  - 更大（绝对值），但与序列线性

无 FlashAttn 时 Attention 激活随 s² 增长，FFN 线性增长
```

**显存比较**（$B=1$, $s=4096$, $h=4096$, $n_h=32$）：

| 部分 | 大小（float16）| 备注 |
|------|--------------|------|
| Attn 激活（无 FlashAttn）| $\sim Bnh_s s^2 = 32 \times 4096^2$ bytes ≈ **512 MB** / 层 | $O(s^2)$，长序列杀手 |
| Attn 激活（FlashAttn）| $\sim 6Bsh \approx 200$ MB / 层 | $O(s)$，可接受 |
| FFN 激活 | $\sim 3 \times 2Bsh \approx 100$ MB / 层 | 恒定于 $s$（线性） |

→ **FlashAttention 是使 Attention 激活从 $O(s^2)$ 降至 $O(s)$ 的关键**，消除了这一显存不平衡。

### 5.3 推理（Decode）显存：KV Cache

**KV Cache 大小**（每层，每 token）：

$$
\text{KV Cache per layer} = 2 \times B \times s \times h_{kv} \times \text{bytes}
$$

其中 $h_{kv} = h / g$（GQA，g 为 GQA 分组数）。

总 KV Cache（所有层，batch B，最大上下文 $c_{max}$）：

$$
\text{KV Cache total} = 2 \times n_{\text{layer}} \times B \times c_{max} \times h / g \times 2 \text{ bytes (fp16)}
$$

**KV Cache vs 权重显存对比**（DeepSeek-R1 量级，70B dense，$h=8192$, $n_l=80$, $g=8$）：

| 场景 | 权重（fp8）| KV Cache |
|------|----------|---------|
| $B=1$, $c=4K$ | ~35 GB | 0.3 GB |
| $B=32$, $c=4K$ | ~35 GB | 10 GB |
| $B=32$, $c=32K$ | ~35 GB | 80 GB（**超出单卡**！）|

→ **KV Cache 完全属于 Attention 侧**，FFN 无 KV Cache。这使得 Attention 实例的显存需求随上下文长度线性增长，而 FFN 显存需求只与 batch size 和模型参数相关。

### 5.4 解耦后的显存分配建议

```
Attention 实例所需显存：
  ├── Attention 参数权重（约 1/4 总参数）
  ├── KV Cache（随上下文线性增长，B × c × h/g）
  └── 通信 buffer（activation transfer）
  
  → 需要 大显存 + 高带宽（H20, HBM3e + 96GB）

FFN/Expert 实例所需显存：
  ├── FFN 参数权重（约 3/4 总参数，MoE 时更大）
  └── 中间激活（B × h × 4h，较小）
  
  → 需要 高 TFLOPS + 成本效益（L40S, H800 fp8 compute）
```

MegaScale-Infer 实验配置（表 3 数据）：

| GPU | 用途 | 单价比 | 内存带宽 per-cost | TFLOPS per-cost |
|-----|------|--------|-----------------|-----------------|
| **H20** | Attention 节点 | 1.85× | **2214** GB/s/$ | 80 |
| **L40S** | Expert 节点 | 1.08× | 800 GB/s/$ | **335** TFLOPS/$ |
| H800 | 通用 | 5.28× | 650 GB/s/$ | 187 |

---

## 六、各方案对比汇总

```
方案              解耦Attn/FFN  计算量估算方法               平衡策略
────────────────────────────────────────────────────────────────────────────
传统PP            ❌            按层参数量均匀分配            静态，无感知
ChunkFlow         ❌            按 chunk token 数均匀化       控制 CS，使 Attn FLOPs 可预测
WLB-LLM           ❌            按 Σsᵢ²（而非 Σsᵢ）均衡     变长文档打包 + per-doc sharding
MegaScale-Infer   ✅            T_a = profiling 线性模型      n_a = k₁E/k₃K，Ping-Pong Pipeline
AFD r* 理论       ✅（纯理论）   解析：T_a ∝ KV tokens, T_f ∝ batch  r* 闭合公式，三区间分析
DWDP（context）   ❌（未解耦）   按 rank 数均匀 DP             消除 all-to-all 同步，不均衡无影响
```

---

## 七、关键洞察与思考

### 1. 估算精度取决于能否将 $s^2$ 项与 $sh^2$ 项分开对待

传统"按层参数量估算"在 $s \ll h$ 时精度尚可，但对长上下文（$s > h/2$）会显著低估 Attention 负载。**WLB-LLM 的 $\sum s_i^2$ 而非 $\sum s_i$ 的洞察**是最简单但最直接的修正。

### 2. Decode 阶段的 Attention 延迟与序列无关的"FLOPs直觉"完全失效

Decode 时 Attention 是内存访问瓶颈（KV cache 读取），与计算 FLOPs 无关。**用 FLOPs 估算 Decode 阶段 Attention 负载会严重低估其延迟**（特别在长上下文时），而 FFN 在 batch 足够大时才是 compute-bound。

### 3. 解耦的核心收益不是 FLOPs 节省，而是硬件异构匹配

解耦本身不减少 FLOPs，但允许：
- 针对性选择硬件（高 BW vs 高 TFLOPS）
- 独立扩展 Attention 实例数（应对长上下文）
- 消除跨模块的强同步依赖（类似 DWDP 的思路）

### 4. 显存不平衡在训练（有 FlashAttn）vs 推理中表现不同

- **训练**：FlashAttn 已将 Attention 激活显存从 $O(s^2)$ 降至 $O(s)$，与 FFN 同量级，显存不平衡问题已大幅缓解
- **推理**：KV Cache 是 Attention 专属的动态显存，完全没有 FFN 侧对应项，随上下文线性增长，是**最需要专项显存管理的部分**

---

## 八、参考来源

| 论文 | 方案 | 核心贡献 | arXiv |
|------|------|---------|-------|
| MegaScale-Infer（ByteDance）| 解耦推理 | Ping-pong pipeline + M2N通信 | [2504.02263](https://arxiv.org/abs/2504.02263) |
| AFD 最优比例理论 | 解耦理论 | 闭合解 $r^*$，三区间分析 | [2601.21351](https://arxiv.org/abs/2601.21351) |
| ChunkFlow（Alibaba/Qwen）| 训练均衡 | Chunk粒度均衡，显存与序列无关 | [2503.02356](https://arxiv.org/abs/2503.02356) |
| WLB-LLM | 训练均衡 | $\sum s_i^2$ 均衡，4D并行 | [2503.17924](https://arxiv.org/abs/2503.17924) |
| LoongTrain | 长序列训练 | 2D-Attention，头并行+上下文并行 | [2406.18485](https://arxiv.org/abs/2406.18485) |
| DWDP | MoE推理 | 消除层间同步，DP+异步P2P预取 | [2604.01621](https://arxiv.org/abs/2604.01621) |
| BPipe（re-eval）| 训练显存 | 激活显存均衡，FlashAttn后收益减弱 | [2401.02088](https://arxiv.org/abs/2401.02088) |
| Parallelization Strategies | 推理对比 | TP vs PP latency-throughput权衡 | [2603.05692](https://arxiv.org/abs/2603.05692) |
