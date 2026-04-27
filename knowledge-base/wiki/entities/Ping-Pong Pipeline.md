---
tags: [实体, 概念, 推理, 通信调度]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-attention-ffn-imbalance-survey]
---

# Ping-Pong Pipeline

> [[MegaScale-Infer]] 提出的分离部署调度技术：把一个 batch 切成 $m$ 个 micro-batch，在 attention 节点和 expert 节点之间交替流转，使 forward 与通信完全 overlap。

## 平衡条件

```
constraint 1: T_a ≈ T_e（通过配置 n_a 实现）
constraint 2: T_c < T_f
constraint 3: m × T_f ≥ 2 × (T_f + T_c)
```

最小 micro-batch 数：

$$
m \geq 2 \left(1 + \frac{T_c}{T_f}\right)
$$

- $T_c / T_f < 1/2$（快通信）→ m ≥ 3
- $T_c / T_f$ 较大（慢通信）→ m ≥ 4

## 与其他 overlap 方案的对比

| 方案 | 适用场景 |
|------|---------|
| Ping-Pong Pipeline | [[MegaScale-Infer]] / 分离部署 |
| DualPipe | DeepSpeek-V3 训练 |
| CAD 的 Ping-Pong 执行 | [[CAD-DistCA]] 训练 |

## 相关页面

- [[MegaScale-Infer]]
- [[分离部署]]
- [[Attention-FFN 解耦]]
