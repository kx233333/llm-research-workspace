---
tags: [实体, 概念, 通信]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-dwdp]
---

# 异步 P2P 预取

> [[DWDP]] 的核心通信原语：用 `cudaMemcpyAsync` 在 CUDA copy engine 上跨 GPU 预取下一层缺失的权重，与计算完全 overlap，**不占用 SM 资源**。

## 关键特性

### 为什么可行

1. **copy engine 独立于 SM**：拉权重不与计算争算力
2. **NVLink 5.0 带宽足够**：1.8 TB/s per GPU 在 NVL72 下可以在一层计算窗口内完成下一层权重传输
3. **双缓冲流水线**：对 layer $l$ 计算时预取 layer $l+1$ 的权重

### 与 a2a 的区别

| 对比 | 异步 P2P | a2a |
|------|---------|-----|
| 同步语义 | 异步，无屏障 | 屏障，等最慢 |
| 通信模式 | 点对点 | 全员对全员 |
| 对负载不均衡敏感度 | 低 | **高** |

## 工程挑战（DWDP 的两项优化）

### 1. 分裂权重合并

多缓冲区权重传统上需要 D2D 合并才能喂给 GEMM → 引入 34 μs/iter 延迟。**扩展 groupedGEMM 接受 TensorList** 消除合并。

### 2. 多对一通信竞争

多个目标 rank 同时向同一源 rank 拉权重 → 源端 copy engine 串行化。**时分复用（TDM）切片 + round-robin** 缓解竞争。

## 相关页面

- [[DWDP]]
- [[CUDA copy-engine]]
- [[NVLink-5.0]]
- [[GB200-NVL72]]
- [[MoE-a2a]]
