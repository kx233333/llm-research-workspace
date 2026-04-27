---
tags: [实体, 概念, 推理, 显存]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-attention-ffn-imbalance-survey, 2026-04-22-dp-only-attention-ffn-survey, 2026-04-22-dwdp]
---

# KV-Cache

> 推理 Decode 阶段保存的 Key/Value 张量，供后续 token 的 Attention 计算访问。**完全属于 Attention 侧，FFN 无对应项**。

## 关键性质

- **大小公式**：$2 \times n_\text{layer} \times B \times c \times h/g \times \text{bytes}$（GQA，$g$ 为分组数）
- **随上下文线性增长**：长序列场景的主要显存消耗
- **动态 / 请求级**：无法像权重那样静态分片
- **Memory-bound**：Decode Attention 主要瓶颈

## 典型规模（DeepSeek-R1 级别）

| 场景 | 权重（fp8）| KV Cache |
|------|----------|---------|
| $B=1, c=4K$ | ~35 GB | 0.3 GB |
| $B=32, c=4K$ | ~35 GB | 10 GB |
| $B=32, c=32K$ | ~35 GB | **80 GB（OOM）** |

## 对解耦策略的影响

KV cache 的存在使得：

1. [[Attention-FFN 解耦]] 时，Attention 实例需要大显存（高 BW + 大容量，如 H20）
2. FFN 实例可以显存小（只需装权重和中间激活）
3. **异构部署的直接动机**

## 未来方向

- KV compression（GQA, MLA, MQA）
- KV offloading（到 CPU DRAM 或 NVMe）
- KV paging（PagedAttention / vLLM）

## 相关页面

- [[Attention-FFN 解耦]]
- [[Core-Attention]]
- [[HBM3e]]
- [[NVLink-C2C]]（支撑 KV offload 到 CPU 侧）
