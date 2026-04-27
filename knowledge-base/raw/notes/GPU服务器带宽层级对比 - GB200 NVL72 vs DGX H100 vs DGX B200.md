---
title: "GPU 服务器带宽层级全景：GB200 NVL72 超节点 vs DGX H100 vs DGX B200"
date: 2026-04-21
tags:
  - 硬件
  - GPU集群
  - GB200
  - NVL72
  - H100
  - B200
  - NVLink
  - 带宽
  - 系统架构
status: 整理中
---

# GPU 服务器带宽层级全景

> 从片内显存到机架间网络，自下而上覆盖五个带宽层级，对比 GB200 NVL72 超节点专用版与传统八卡服务器（DGX H100 / DGX B200）。

---

## 一、系统拓扑概览

| 项目 | **GB200 NVL72** | **DGX H100** | **DGX B200** |
|------|----------------|--------------|--------------|
| GPU | 72× B200 SXM | 8× H100 SXM5 | 8× B200 SXM |
| GPU 显存 | 192 GB HBM3e × 72 = 13.8 TB | 80 GB HBM3 × 8 = 640 GB | 180 GB HBM3e × 8 = 1.44 TB |
| CPU | 36× Grace（ARM Neoverse V2） | 2× Intel Xeon 8480C | 2× Intel Xeon 8570 |
| CPU 内存 | 480 GB LPDDR5X × 36 = 17.3 TB | DDR5 2 TB | DDR5 2–4 TB |
| NVSwitch 芯片 | **18 个**（9 托盘，每盘 2 芯片） | 4 个 | 2 个 |
| 形态 | **整机柜（单 NVLink 域）** | 标准 8U 机箱 | 标准 8U 机箱 |
| 功耗 | ~125 kW / 机柜（液冷） | ~10.2 kW（风冷） | ~14.3 kW（液冷） |

---

## 二、五层带宽层级（从快到慢）

### L1 — GPU 显存（HBM）

> 片内最快存储，AI 计算的直接数据源

| 规格 | **GB200 NVL72** | **DGX H100** | **DGX B200** |
|------|----------------|--------------|--------------|
| 显存类型 | HBM3e | HBM3 | HBM3e |
| **带宽（单 GPU）** | **8 TB/s** | **3.35 TB/s** | **8 TB/s** |
| 总带宽（全系统） | 576 TB/s | 26.8 TB/s | 64 TB/s |
| 容量（单 GPU） | 192 GB | 80 GB | 180 GB |
| NVL72 vs H100 | **2.4×** | — | — |

> **注**：HBM3e 通过更宽的总线（B200 为 8192-bit）和更高时钟实现倍增，是模型权重/激活值存取的根本吞吐上限。

---

### L2 — CPU 内存（DRAM）

> CPU 侧系统内存，传统架构中 GPU 通过 PCIe 访问速度极慢；NVL72 通过 C2C 直连使其可作为有效扩展内存池

| 规格 | **GB200 NVL72** | **DGX H100** | **DGX B200** |
|------|----------------|--------------|--------------|
| 内存类型 | LPDDR5X | DDR5-4800 | DDR5-5600 |
| **带宽（单 CPU）** | **546 GB/s** | ~307 GB/s | ~307 GB/s |
| 总带宽（全系统） | ~19.7 TB/s（36× CPU） | ~614 GB/s（2× CPU） | ~614 GB/s（2× CPU） |
| 容量（单 CPU） | 480 GB | — | — |
| 系统总内存 | **17.3 TB** | 2 TB | 2–4 TB |

> **NVL72 的独特优势**：Grace CPU 的 LPDDR5X 带宽达 546 GB/s，通过下方 C2C 接口可被 GPU 以近显存速度访问，形成 ~30 TB 统一内存池，是 MoE 大模型推理 / KV-cache 卸载的硬件基础。

---

### L3 — CPU ↔ GPU 互联

> 决定 CPU 内存能否被 GPU 高效利用的关键瓶颈层

| 规格 | **GB200 NVL72** | **DGX H100** | **DGX B200** |
|------|----------------|--------------|--------------|
| 互联类型 | **NVLink-C2C**（片间直连） | PCIe Gen5 x16 | PCIe Gen5 x16 |
| **双向带宽** | **900 GB/s** | **128 GB/s** | **128 GB/s** |
| 延迟 | 极低（同封装级） | 高（通过 PCIe RC） | 高（通过 PCIe RC） |
| NVL72 vs H100 | **~7×** | — | — |

> **结构性差异**：传统 DGX 中 PCIe 是 CPU↔GPU 的唯一通道，128 GB/s 远低于 GPU 显存带宽（3.35 TB/s），使 CPU DRAM 对 GPU 几乎不可用。NVL72 的 NVLink-C2C 以 900 GB/s 连接 Grace CPU 与 B200 GPU，打通了这一瓶颈。

---

### L4 — GPU 间互联（机箱/机柜内 NVLink Fabric）

> 决定多 GPU 集合通信（all-reduce / all-to-all）效率，直接影响训练和推理并行策略

| 规格 | **GB200 NVL72** | **DGX H100** | **DGX B200** |
|------|----------------|--------------|--------------|
| NVLink 版本 | **NVLink 5.0** | NVLink 4.0 | NVLink 5.0 |
| **带宽（单 GPU，双向）** | **1.8 TB/s** | **900 GB/s** | **1.8 TB/s** |
| NVSwitch 芯片数 | **18 个** | 4 个 | 2 个 |
| NVLink 域内 GPU 数 | **72** | 8 | 8 |
| **总 NVLink 带宽** | **130 TB/s** | **7.2 TB/s** | **14.4 TB/s** |
| vs PCIe Gen5 | **~14×** | ~7× | ~14× |
| NVL72 vs H100（单 GPU） | **2×** | — | — |
| NVL72 vs H100（全域）| **18×** | — | — |

> **NVL72 的关键突破**：通过 18 块 NVSwitch 芯片将 72 GPU 组成**单一 NVLink 域**，实现机柜内任意两卡之间均为全带宽直通，无需跨机柜通信即可运行千亿参数模型。DGX H100/B200 则需跨节点 InfiniBand 才能扩展至更多 GPU。

---

### L5 — 节点间 / 机柜间 Scale-out 网络

> 集群扩展的最终带宽上限

| 规格 | **GB200 NVL72** | **DGX H100** | **DGX B200** |
|------|----------------|--------------|--------------|
| NIC 型号 | **ConnectX-8 SuperNIC** | ConnectX-7 | ConnectX-7 |
| **单端口带宽** | **800 Gbps（100 GB/s）** | 400 Gbps（50 GB/s） | 400 Gbps（50 GB/s） |
| 主机接口 | PCIe Gen6 × 48 lanes | PCIe Gen5 × 16 lanes | PCIe Gen5 × 16 lanes |
| 端口数 / 节点 | 36（每 Grace Superchip 1 个） | 8 | 8 |
| **总 Scale-out 带宽** | **~3.6 TB/s**（36 × 100 GB/s） | ~400 GB/s（8 × 50 GB/s） | ~400 GB/s（8 × 50 GB/s） |
| NVL72 vs H100 | **~9×** | — | — |
| 交换机配套 | Quantum-X800 InfiniBand | Quantum-2 InfiniBand | Quantum-2 InfiniBand |

---

## 三、综合对比汇总表

```
带宽层级         GB200 NVL72          DGX H100          DGX B200
─────────────────────────────────────────────────────────────────────
L5 节点间         800 Gbps/port        400 Gbps/port     400 Gbps/port
                 (CX-8, PCIe Gen6)    (CX-7, PCIe Gen5) (CX-7, PCIe Gen5)

L4 GPU间         1.8 TB/s / GPU       900 GB/s / GPU    1.8 TB/s / GPU
  NVLink Fabric  130 TB/s (72 GPU域)  7.2 TB/s (8 GPU)  14.4 TB/s (8 GPU)
                 18× NVSwitch 芯片    4× NVSwitch 芯片   2× NVSwitch 芯片

L3 CPU↔GPU      NVLink-C2C           PCIe Gen5         PCIe Gen5
                 900 GB/s             128 GB/s          128 GB/s
                 (片间直连，低延迟)    (跨总线，高延迟)  (跨总线，高延迟)

L2 CPU内存       LPDDR5X              DDR5-4800         DDR5-5600
                 546 GB/s / CPU       ~307 GB/s / CPU   ~307 GB/s / CPU
                 (×36 CPU总计19.7TB/s)(2CPU合计~614GB/s) (2CPU合计~614GB/s)

L1 GPU显存       HBM3e                HBM3              HBM3e
                 8 TB/s / GPU         3.35 TB/s / GPU   8 TB/s / GPU
                 192 GB / GPU         80 GB / GPU       180 GB / GPU
```

---

## 四、关键洞察

### 1. NVL72 的本质：把 NVLink 从"节点内"扩展到"机柜级"

传统 DGX 的 NVLink 只覆盖箱内 8 卡（7.2–14.4 TB/s）；超出则须经 InfiniBand（~400 GB/s，带宽骤降 **18–36×**）。NVL72 将 NVLink 域扩展至整机柜 72 卡，**消除了 8 卡以上规模的 InfiniBand 瓶颈**，使 130 TB/s 的全互联带宽覆盖万亿参数模型的完整训练/推理过程。

### 2. C2C 打通了"显存墙"与"系统内存"之间的护城河

- 传统架构：CPU DRAM（~614 GB/s）→ PCIe（128 GB/s）→ GPU，瓶颈是 PCIe
- NVL72：CPU LPDDR5X（546 GB/s/CPU）→ NVLink-C2C（900 GB/s）→ B200 GPU
- 效果：72 GPU × 192 GB HBM + 36 CPU × 480 GB LPDDR5X = **~30 TB 统一内存池**，MoE 专家权重可在 GPU 显存不足时以极低代价卸载到 CPU 侧

### 3. 带宽悬崖：跨层级的量级差距

```
HBM3e显存  ████████████████████  8,000 GB/s  ←  计算最终受限于此
NVLink-C2C  ████████              900 GB/s
NVLink 5.0  █████████████████     1,800 GB/s（per GPU）
PCIe Gen5   █                     128 GB/s   ←  传统DGX的CPU↔GPU瓶颈
IB 400Gbps  ▌                     50 GB/s    ←  传统多节点扩展瓶颈
```

每跨越一个层级，带宽通常下降 **4–20×**，软件栈设计（数据并行/张量并行/流水线并行/MoE 路由）的核心目标就是最大化高带宽层的利用率，避免频繁跌入低带宽层。

### 4. DWDP 论文的硬件基础解读

[[DWDP - Distributed Weight Data Parallelism for LLM Inference]] 中的核心技巧——**MoE 专家权重异步 P2P 预取**——之所以可行，正是因为：
- NVLink 5.0 提供 **1.8 TB/s** per GPU 的 P2P 带宽，足以在一层 MoE 计算窗口内传完下一层的专家权重
- copy engine 独立于 SM，不与计算争资源
- 若在传统 DGX（PCIe 连接不同节点）上尝试同样策略，带宽下降 ~36×，预取根本无法在计算窗口内完成

---

## 五、参考来源

- [NVIDIA GB200 NVL72 官方产品页](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [NVIDIA Developer Blog CN — GB200 NVL72 互连架构深度解析](https://developer.nvidia.cn/blog/?p=9393)
- [NVIDIA DGX H100 官方规格](https://www.nvidia.com/en-us/data-center/dgx-h100/)
- [NVIDIA DGX H100 用户指南 — NVIDIA Docs](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)
- [NVIDIA DGX B200 官方规格](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [NVIDIA DGX B200 用户指南 — NVIDIA Docs](https://docs.nvda.net.cn/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)
- [ConnectX-8 SuperNIC 技术介绍 — 今日头条](https://www.toutiao.com/article/7440983922936218127/)
- [Intel Xeon Platinum 8480C 官方规格](https://www.intel.com/content/www/us/en/products/compare.html?productIds=231735,231730)
- [NVIDIA Blackwell Architecture Technical Overview](https://resources.nvidia.com/en-us-blackwell-architecture)
