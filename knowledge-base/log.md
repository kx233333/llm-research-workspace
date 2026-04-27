# 操作日志

> 记录知识库的所有变更历史

---

## 2026-04-22 — 初始化

- **操作**：创建知识库
- **主题**：LLM 分布式系统调研
- **状态**：完成

---

## 2026-04-22 — batch-ingest | raw/notes/ 全量消化

- **操作**：批量消化 7 个本地素材
- **文件**：
  - 6 个 md 笔记（MPress / ChunkFlow / DWDP / GPU 带宽对比 / Attention-FFN 不平衡调研 / 仅 DP 下的调研）
  - 1 个 docx（长上下文&超节点调研，通过 textutil 提取为 txt 处理）
- **新增页面**：
  - `wiki/sources/` × 7
  - `wiki/entities/` × 40（12 方案 + 9 硬件 + 19 技术概念）
  - `wiki/topics/` × 5
  - 合计 **52** 个 wiki 页面
- **主题覆盖**：
  - [[负载均衡与变长序列]]
  - [[并行策略与同步开销]]
  - [[显存优化]]
  - [[超节点架构]]
  - [[Attention-FFN 计算量不平衡]]
- **核心概念网**：
  - 方案间相互引用（[[DWDP]] ↔ [[ChunkFlow]] ↔ [[CAD-DistCA]] ↔ [[MegaScale-Infer]]）
  - 硬件 → 方案依赖链（[[GB200-NVL72]] → [[DWDP]]、[[GH200-Superchip]] → [[SuperOffload]]）
  - 通信机制分类（[[MoE-a2a]]、[[CP-a2a]]、[[异步 P2P 预取]]、[[D2D-NVLink-Swap]]、[[Ping-Pong Pipeline]]）
- **跨素材洞察**（跳出单篇的综合）：
  - ChunkFlow 与 DWDP 的共同模式——用异步/串行替代 a2a
  - Decode 下 FLOPs 估算失效，必须用 memory bandwidth 模型
  - FlashAttention 解决训练显存，但推理 KV cache 仍是 Attention 独占难题
  - DP 规模 ≥ 8 时 Attention 解耦从"可选"变"必要"
- **状态**：完成

---

## 2026-04-23 — synthesis | NVL72 长文本训练研究路线图

- **操作**：基于 [[CAD-DistCA]] 为基底的多轮深度讨论，生成综合研究路线图
- **讨论脉络**：
  1. CAD method 深度解读 → pre-CA / post-CA / Ping-Pong 机制
  2. CAD 实验结果分析（1.35× 加速，55% bubble 消除）
  3. 以 CAD 为基底的多个扩展 idea
  4. CAD 是否考虑节点内外 → scheduler 扁平化盲点
  5. CAD 是否适用 MoE → 部分适用，attention 侧收益 ~14%
  6. Sequence-group 组内 $2n-1$ 不均衡（causal mask）
  7. 变长不均衡在 dense vs MoE 里的不同地位
- **产出**：[[NVL72长文本训练研究路线图]]
  - 6 个核心洞察
  - 6 个论文级研究方向
  - 12 个工程可落地 idea
  - 统一系统蓝图
  - 优先级选型建议
- **关键结论**：方向 1（Topology-Aware）+ 方向 2（Sequence-Group）合并为**两维度层次化 CAD** 是最推荐的起点
- **状态**：完成
