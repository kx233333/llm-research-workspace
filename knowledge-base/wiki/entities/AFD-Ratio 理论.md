---
tags: [实体, 方案, 推理系统, 理论]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-attention-ffn-imbalance-survey]
---

# AFD-Ratio 理论

> 闭合解推导：给定工作负载，Attention 实例:FFN 实例的最优比例 $r^*$（arXiv: 2601.21351）。为 [[MegaScale-Infer]] 类 AFD 系统提供理论基础。

## 关键信息

- **arXiv**：2601.21351
- **拓扑**：$r$ 个 Attention 实例 → 1 个 FFN 实例（rA-1F）

## 延迟模型（三项线性拟合）

| 组件 | 公式 | 物理含义 |
|------|------|---------|
| Attention | $t_A(T) = \alpha_A T + \beta_A$ | $T$ = KV cache 总 token，内存带宽受限 |
| FFN | $t_F(rB) = \alpha_F rB + \beta_F$ | 计算受限，与 batch 线性 |
| 通信 | $t_C(B) = \alpha_C B + \beta_C$ | activation transfer |

## 三种工作区间与最优 $r$

| 区间 | 瓶颈 | 最优 $r$ 策略 |
|------|------|-------------|
| Attention-bottleneck | $t_A \geq t_C, t_F$ | 增大 $r$，取 $r_A$ 边界 |
| Communication-bottleneck | $t_C \geq t_A, t_F$ | 增大 $r$，取 $r_C$ 边界 |
| FFN-bottleneck | $t_F \geq t_A, t_C$ | $r_{\text{peak}} = \sqrt{\beta_F/(\alpha_F B)}$ |

## 闭合解

$$
r^* = \max\left\{
\frac{\alpha_A \bar{T} + \beta_A - \beta_F}{\alpha_F B},
\frac{\bar{t}_C - \beta_F}{\alpha_F B},
\sqrt{\frac{\beta_F}{\alpha_F B}}
\right\}
$$

## 实验验证（DeepSeek-V3 + Ascend 910C）

- $B=256, \mu_D=500, \mu_P=100$ → $r^* \approx 9.3$
- 与仿真最优误差 **<10%**
- **批量增大、上下文变长时 $r^*$ 增大**

## 实验数据

| 工作点 | 最优 $r^*$ |
|--------|-----------|
| batch=128 | ~7.1 |
| batch=512 | ~10.3 |

## 相关页面

- [[Attention-FFN 解耦]]
- [[MegaScale-Infer]]
- [[DWDP]]
- [[MoE 推理]]
