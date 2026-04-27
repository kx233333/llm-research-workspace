---
tags: [实体, 方案, 训练系统, 负载均衡]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-long-context-supernode-survey, 2026-04-22-attention-ffn-imbalance-survey, 2026-04-22-dp-only-attention-ffn-survey, 2026-04-22-chunkflow]
---

# WLB-LLM

> **W**orkload-**B**alanced 4D 并行训练系统（OSDI 2025，UC San Diego + Meta），对 PP 和 CP 层面的工作负载不均衡做联合优化，1.23× 平均加速。

## 关键信息

- **类型**：训练系统 / 负载均衡
- **发表**：OSDI 2025，arXiv: 2503.17924
- **所属主题**：[[负载均衡与变长序列]]、[[并行策略与同步开销]]

## 两个核心矛盾

| 矛盾 | 解决方法 |
|------|---------|
| PP 微批次间 workload 不均衡 | **Var-Length Packing + Outlier Document Delay** |
| CP 序列分片间不均衡 | **Fine-grained 逐文档分片 + 自适应 sharding 策略** |

## Var-Length Packing 公式

定义负载感知函数：

$$
W(s) = W_a(s) + W_l(s)
$$

其中：
- $W_a(s) \propto s^2$（Attention 部分）
- $W_l(s) \propto s$（GEMM + 通信 + element-wise）

通过离线 profiling 拟合。优化目标：

$$
\min \max_j \sum_i x_{ij}(W_a(d_i) + W_l(d_i))
$$

## Outlier Document Delay

- 多级等待队列保留极长文档
- 队列满（= micro-batch 数）才释放
- 保证每个 micro-batch 含恰好一个长文档 → Attention 主导项接近

## 实验结果

| 场景 | 加速比 |
|------|-------|
| 平均 | **1.23×** |

## 与其他方案的对比

| 对比 | 关系 |
|------|------|
| [[ChunkFlow]] | 都处理变长；ChunkFlow 批评 WLB-LLM 的 shuffle 依赖跨 batch，影响数据随机性。WLB-LLM 用 profiling + 启发式，ChunkFlow 用 chunk 均一化 |
| [[ByteScale]] | ByteScale 允许不同 DP rank 处理不同数量 micro-batch（打破梯度累积等量约束），WLB-LLM 保持等量约束 |
| [[FlexSP]] | FlexSP 侧重 SP 组异构，WLB-LLM 侧重 4D 并行内的 PP+CP 均衡 |
| [[CAD-DistCA]] | CAD 直接解耦 Attention，比 WLB-LLM 的估算路径更彻底——但 CAD 把此方案当主要对比基线 |

## 批评 / 局限（来自其他素材）

- [[ChunkFlow]]：shuffle 窗口大影响数据随机性和模型收敛
- [[CAD-DistCA]]：variable-length chunking 使某些 rank 显存翻倍（+8–17% overhead），且 DP 越大问题越严重

## 相关页面

- [[Attention-FFN 计算量不平衡]]
- [[负载均衡与变长序列]]
- [[ChunkFlow]]
- [[ByteScale]]
- [[FlexSP]]
- [[CAD-DistCA]]
