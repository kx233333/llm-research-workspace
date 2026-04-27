---
tags: [实体, 方案, 序列并行, 长上下文]
created: 2026-04-22
updated: 2026-04-22
sources: [2026-04-22-long-context-supernode-survey]
---

# USP

> **U**nified **S**equence **P**arallelism（Tencent）：把 DeepSpeed-Ulysses（a2a）和 Ring-Attention（p2p）组合为 **Hybrid-SP**，解决各自的并行度受限问题。

## 关键信息

- **来源组织**：腾讯（也是目前公司 Megatron 正在使用的 SP 方法）

## 核心问题

- **Ulysses** 并行度受 head number 和单节点卡数限制
- **Ring-Attention** 带宽利用低效
- 二者单独用都有硬约束

## 核心机制

两维混合：
- 节点内 / 高带宽域 → Ulysses（a2a）
- 跨节点 / 低带宽域 → Ring（p2p）

## 核心洞察（docx 思考）

> ulysses-degree 受 head 数约束以及高速互联域规模限制；tp-degree 受 hidden/FFN/head 的可切分性与 kernel 效率约束以及高速互联域规模限制，在较大参数和超节点中，这种限制可能一定程度会被释放（乘积会>8），如何设计单 layer 内部的张量布局变化？

引出的开放问题：
- 联合切分 head or 单独切
- A2A 通信是在 TP group 单独做，还是在全局做
- 分层动态布局 or 存在默认布局
- TP 和 EP 的生态位问题

## 相关页面

- [[Ulysses-CP]]
- [[Ring-Attention]]
- [[CP-a2a]]
- [[GB200-NVL72]]
- [[负载均衡与变长序列]]
