---
tags: [实体, 概念, 推理系统, 跨数据中心]
created: 2026-04-29
updated: 2026-04-29
aliases: [PrfaaS, Prefill-as-a-Service, 跨数据中心 Prefill]
---

# PrfaaS（Prefill-as-a-Service）

## 定义

PrfaaS 是一种跨数据中心的 LLM 推理架构。利用 Hybrid Attention 模型（线性注意力 + 少量全注意力）将 KVCache 缩小 10-36×，使得长序列 Prefill 可以卸载到远程算力密集型集群，通过普通以太网传回 KVCache，由本地集群执行 Decode。

## 核心机制

```
传统 PD 分离：Prefill → [RDMA 同机房] → Decode     ← 带宽要求高
PrfaaS：     长 Prefill → [100G 以太网跨机房] → Decode  ← Hybrid 模型 KV 小
             短 Prefill → 本地直接处理              ← 选择性卸载
```

## 关键数据

- 1T Hybrid 模型：100 Gbps 链路仅用 **13%**
- vs 同构 PD：**+54% 吞吐，-64% P90 TTFT**

## 前提条件

需要 Hybrid Attention 模型（如 Kimi Linear、MiMo-V2-Flash、Qwen3.5）才能实现足够的 KVCache 缩减。

详见 → [[2026-04-29-prefill-as-a-service|论文分析]]

## 关联概念

- [[分离部署]] — PD 分离的基础
- [[KV-Cache]] — KVCache 缩减是核心前提
- [[MegaScale-Infer]] — 推理系统优化
- [[长上下文训练]] — 训练侧的对应问题
