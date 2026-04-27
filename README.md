# LLM 分布式系统研究工作站

> 本项目是一个可移植的 Claude Code + Obsidian 工作站，包含 LLM 分布式系统的知识库、自动化 skill 和配置。
> 新机器上 `git clone --recursive` + `bash setup.sh` 即可使用。

## 项目结构

本目录同时是：
1. **Claude Code 项目根** — `claude` 命令在此目录下启动即自动加载 CLAUDE.md
2. **Obsidian Vault** — Obsidian 打开此目录即可浏览知识库
3. **Git 仓库** — 所有知识库内容版本管理

## 知识库

知识库位于 `knowledge-base/`，主题为 **LLM 分布式系统调研**，涵盖：
- 训练系统（长上下文、并行策略、负载均衡）
- 推理系统（MoE、分离部署）
- 硬件层级（NVLink、NVSwitch、InfiniBand、HBM）
- 超节点架构（GB200 NVL72、DGX）

### 文件结构
- `knowledge-base/.wiki-schema.md` — 知识库元数据和规则
- `knowledge-base/purpose.md` — 研究方向和范围
- `knowledge-base/wiki/entities/` — 概念实体页（50+）
- `knowledge-base/wiki/topics/` — 主题综合页
- `knowledge-base/wiki/sources/` — 素材分析页
- `knowledge-base/raw/notes/` — 原始研究笔记

### 关键知识路径
- 想了解某个概念 → 看 `entities/`
- 想了解某篇论文/PR → 看 `sources/`
- 想了解某个研究主题 → 看 `topics/`

## Skill

`skill-source/` 是 [llm-wiki-skill](https://github.com/sdyckjq-lab/llm-wiki-skill) 的 Git submodule，提供 10 个核心工作流：

| 工作流 | 触发方式 | 说明 |
|--------|---------|------|
| init | "初始化知识库" | 创建新的知识库结构 |
| ingest | URL/文件 + "消化" | 自动提取、分析、生成 wiki 页 |
| query | "关于 XX" | 从知识库合成回答 |
| lint | "健康检查" | 检查矛盾、断链、置信度 |
| digest | "深入研究 XX" | 跨素材综合分析 |
| graph | "显示图谱" | 生成交互式知识图谱 |
| crystallize | "保存会话" | 将对话结晶为综合页 |

## 日常使用

```bash
# 启动 Claude Code（在本目录下）
cd ~/llm-research-workspace
claude

# 打开 Obsidian
open -a Obsidian ~/llm-research-workspace
```

## 在新机器上部署

### 前置条件
- Claude Code CLI 已安装
- Git 已安装
- （可选）Obsidian 已安装

### 安装

```bash
git clone --recursive https://github.com/<your-username>/llm-research-workspace.git ~/llm-research-workspace
cd ~/llm-research-workspace
bash setup.sh
```

### 验证

```bash
cd ~/llm-research-workspace
claude
# 在 Claude 中输入：关于 Dynamic CP
# 应返回知识库中的内容
```
