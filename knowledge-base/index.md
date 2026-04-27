# 知识库索引

> 最后更新：2026-04-22

---

## 概览

- 主题：LLM 分布式系统调研
- 素材总数：7（6 md + 1 docx）
- Wiki 页面总数：52（7 sources + 40 entities + 5 topics）

---

## 主题页

> 研究主题、知识领域

- [[负载均衡与变长序列]] — 变长序列下的多 GPU 负载均衡
- [[并行策略与同步开销]] — DP/TP/PP/CP/EP 的同步语义与优化
- [[显存优化]] — GPU 显存墙的系统级应对
- [[超节点架构]] — NVL72 / GH200 等新硬件形态
- [[Attention-FFN 计算量不平衡]] — $s^2 h$ vs $sh^2$ 矛盾

---

## 素材摘要

> 每个消化过的素材都有一篇摘要

### 论文笔记

- [[2026-04-22-mpress]] — MPress (HPCA 2023)
- [[2026-04-22-chunkflow]] — ChunkFlow (ICML 2025)
- [[2026-04-22-dwdp]] — DWDP (NVIDIA, NVL72)
- [[2026-04-22-gpu-bandwidth-hierarchy]] — GPU 带宽层级对比

### 综合调研

- [[2026-04-22-long-context-supernode-survey]] — 长上下文 & 超节点调研（总 outline，docx 源）
- [[2026-04-22-attention-ffn-imbalance-survey]] — Attention-FFN 计算量不平衡与解耦调研
- [[2026-04-22-dp-only-attention-ffn-survey]] — 仅 DP 下的 Attention-FFN 平衡调研

---

## 实体页

> 人物、组织、概念、工具、方案等

### 方案 / 论文（12）

**训练系统**：
- [[MPress]]（HPCA 2023）
- [[ChunkFlow]]（ICML 2025）
- [[WLB-LLM]]（OSDI 2025）
- [[FlexSP]]（ASPLOS 2025）
- [[ByteScale]]（arXiv 2502.21231）
- [[CAD-DistCA]]（arXiv 2510.18121）
- [[LobRA]]（arXiv 2509.01193）
- [[SuperOffload]]（ASPLOS 2026）
- [[USP]]（Tencent Hybrid-SP）

**推理系统**：
- [[DWDP]]（arXiv 2604.01621）
- [[MegaScale-Infer]]（arXiv 2504.02263）
- [[AFD-Ratio 理论]]（arXiv 2601.21351）

### 硬件（9）

**平台**：
- [[GB200-NVL72]]
- [[GH200-Superchip]]
- [[DGX-H100]]
- [[DGX-B200]]

**互联**：
- [[NVLink]]（总览）
- [[NVLink-5.0]]
- [[NVLink-C2C]]
- [[NVSwitch]]
- [[ConnectX-SuperNIC]]

**存储 / CPU**：
- [[HBM3e]]
- [[Grace-CPU]]

### 技术概念（19）

**架构 / 并行**：
- [[Attention-FFN 解耦]]
- [[Core-Attention]]
- [[算子间并行]]
- [[Chunk 粒度调度]]
- [[State-Aware 1F1B]]
- [[Ping-Pong Pipeline]]

**通信**：
- [[MoE-a2a]]
- [[CP-a2a]]
- [[异步 P2P 预取]]
- [[D2D-NVLink-Swap]]
- [[CUDA copy-engine]]

**存储 / 数据**：
- [[FlashAttention]]
- [[KV-Cache]]
- [[Causal-Attention]]

**场景**：
- [[长上下文训练]]
- [[MoE 推理]]
- [[分离部署]]

---

## 对比分析

> 对比不同方案、工具、观点

（暂无，可通过 digest 工作流生成）

---

## 综合分析

> 跨素材的深度分析

- [[NVL72长文本训练研究路线图]] — **6 个核心洞察 + 18 个具体 idea**，基于 CAD 为基底的完整研究方向梳理（2026-04-23）

---

## 建议的下一步

本轮积累了丰富的素材，建议生成：

1. **对比表**：`digest 对比 ChunkFlow 和 DWDP 在 a2a 消除上的不同路径`
2. **深度报告**：`digest 超节点上长上下文预训练的系统设计空间`
3. **时间线**：`按时间排列 LLM 分布式训练系统的演进`
4. **知识图谱**：`画知识图谱`
