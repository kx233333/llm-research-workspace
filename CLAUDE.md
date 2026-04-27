# CLAUDE.md — LLM 分布式系统研究工作站

## 这是什么

这是一个 LLM 分布式系统研究的知识库工作站。你（Claude）在这个目录下工作时，具备以下能力：

1. **知识库操作** — 通过 llm-wiki skill 管理结构化知识
2. **论文/代码分析** — 分析论文、GitHub PR、代码仓库，生成 Obsidian 格式的分析报告
3. **知识图谱** — 维护实体间的双向链接网络

## 知识库位置

- **知识库根目录**: `knowledge-base/`
- **Wiki 页面**: `knowledge-base/wiki/`
  - `entities/` — 概念实体（50+ 页）
  - `topics/` — 主题综合
  - `sources/` — 素材分析
  - `synthesis/` — 跨素材综合
  - `comparisons/` — 对比分析
- **原始素材**: `knowledge-base/raw/notes/`
- **知识库 schema**: `knowledge-base/.wiki-schema.md`
- **研究方向**: `knowledge-base/purpose.md`

## 写入知识库的规则

向知识库添加内容时，遵循以下规则：

1. **新建素材分析** → 放 `knowledge-base/wiki/sources/`，命名 `YYYY-MM-DD-标题.md`
2. **新建/更新实体** → 放 `knowledge-base/wiki/entities/`
3. **所有页面使用 YAML frontmatter**（tags, created, updated）
4. **使用 `[[双链]]`** 建立页面间关联
5. **中文输出**，技术术语保持英文

## Skill

llm-wiki skill 已通过 `setup.sh` 安装。可用的工作流：

- **消化素材**: 给我一个 URL 或文件，我会提取核心观点并更新知识库
- **查询**: "关于 XX" — 从知识库合成回答
- **健康检查**: 检查断链、矛盾、低置信度标注
- **知识图谱**: 生成交互式 HTML 图谱
- **保存会话**: 将有价值的对话结晶为综合页

## 项目配置

- **Skill 源码**: `skill-source/`（git submodule → sdyckjq-lab/llm-wiki-skill）
- **权限配置**: `.claude/settings.local.json`
- **Obsidian 配置**: `.obsidian/`（本目录即 Obsidian Vault root）
