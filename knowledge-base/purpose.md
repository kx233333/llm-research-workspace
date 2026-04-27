# 研究目的与方向

## 核心目标
围绕 **LLM 分布式训练与推理系统** 积累一个持续演化的调研知识库。重点聚焦在：当模型规模（参数量）和上下文规模（序列长度）同时增大时，如何在多 GPU / 多节点系统中高效执行训练与推理，同时保持计算和显存负载的平衡。目标是建立一张关于"瓶颈 → 技术方案 → 硬件基础"的可查询知识网络。

## 关键问题
1. **负载估算与均衡**：变长序列下，Attention（$s^2 h$）和 FFN（$s h^2$）的计算量如何准确估算？TP/PP/DP/CP 各自在均衡策略上的取舍是什么？是否应当显式解耦 Attention 与 FFN？
2. **显存与带宽层级**：从 HBM → NVLink → C2C → InfiniBand，每一层的带宽如何决定可行的并行策略？长上下文训练/推理的显存瓶颈在哪？KV cache 应该放在哪一层存储？
3. **MoE 与稀疏激活**：MoE 模型的负载偏斜（专家路由不均）和层间同步代价如何处理？异步 P2P 预取、expert parallelism、disaggregated expert 等方案的适用边界是什么？
4. **超节点（NVL72 等）与传统 DGX 的差异**：NVL72 整机柜 NVLink 域扩展（72 GPU 全互联）相比传统 8 卡节点带来了哪些系统层面的新可能？哪些历史方案在这一硬件形态下被颠覆，哪些仍然适用？
5. **分离部署（Disaggregated Serving）**：Prefill/Decode 分离、Attention/FFN 分离的共性与差异？何时值得引入异构硬件？

## 研究范围
**涵盖：**
- LLM 训练系统：并行策略（DP / TP / PP / CP / EP）、显存优化（ZeRO / Offload / D2D swap / 重计算 / chunked）、负载均衡（WLB-LLM / ByteScale / ChunkFlow / CAD 等）
- LLM 推理系统：KV cache 管理、分离部署、MoE 调度、量化部署
- 硬件基础：NVLink / NVSwitch / HBM / C2C / ConnectX / InfiniBand 带宽层级
- 超节点架构：GB200 NVL72 / DGX H100 / DGX B200 的系统拓扑及其对算法设计的影响
- 长上下文技术：FlashAttention 族、Ring/Context Parallelism、序列切块

**不涵盖：**
- 模型算法本身（架构创新、预训练配方、RLHF 细节）
- 应用层（RAG、Agent、工具调用）
- 数据工程（数据清洗、tokenizer 设计）
- 边缘 / 移动端部署（除非与超节点设计形成有益对比）
