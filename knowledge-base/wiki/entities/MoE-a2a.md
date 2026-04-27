---
tags: [实体, 概念, 通信, MoE]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-dwdp, 2026-04-22-long-context-supernode-survey]
---

# MoE-a2a

> MoE 架构中的 token-to-expert 路由通信，每 MoE 层需要两次 a2a（dispatch + combine）。**MoE 对负载不均衡敏感的首要原因**。

## 结构

```
Forward:
  token → expert  (dispatch a2a)
  expert compute
  expert → token  (combine a2a)

Backward:
  grad_token → expert  (combine a2a 反向)
  expert grad compute
  grad_expert → token  (dispatch a2a 反向)
```

## 为什么特别敏感

### 三个性质叠加

1. **双向阻塞**：每 rank 发给每 rank 不等量数据（路由偏斜），一端慢 = 所有 pair 等
2. **双重不均衡耦合**：计算不均（热门 expert）+ 通信不均（目标 rank 接收更多）都暴露在 a2a 阶段
3. **高频**：80 层模型每 iter **160 次 a2a**（dispatch + combine），远多于 all-reduce

### 实测数据

- DEP 策略下 20% CV 负载不均 → 同步开销占总迭代 **12%**（[[DWDP]] 论文测得）

## 优化路径

### 路线 A：消灭 a2a

- [[DWDP]]：异步 P2P 预取替代（forward 端；backward 未解决）

### 路线 B：保留但隐藏

- [[MegaScale-Infer]]：Ping-pong pipeline + M2N 高性能通信库
- DeepSeek DualPipe：同时隐藏 forward 和 backward a2a

### 路线 C：从源头减少不均衡

- aux-loss-free balancing（DeepSeek-V3）
- 动态 expert duplication

## 相关页面

- [[DWDP]]
- [[MegaScale-Infer]]
- [[异步 P2P 预取]]
- [[CP-a2a]]
- [[MoE 推理]]
- [[并行策略与同步开销]]
