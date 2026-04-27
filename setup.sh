#!/bin/bash
# setup.sh — 一键初始化 Claude Code + llm-wiki 工作站
# 用法：git clone --recursive <repo> && cd <repo> && bash setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SKILL_SOURCE="$REPO_ROOT/skill-source"
KB_PATH="$REPO_ROOT/knowledge-base"

echo "=== LLM Research Workspace Setup ==="
echo "Repo root: $REPO_ROOT"
echo ""

# 0. 检查 submodule 是否已拉取
if [ ! -f "$SKILL_SOURCE/install.sh" ]; then
    echo "[*] Initializing git submodules..."
    cd "$REPO_ROOT"
    git submodule update --init --recursive
fi

# 1. 安装 llm-wiki skill 到 ~/.claude/skills/
echo "[1/3] Installing llm-wiki skill..."
if [ -f "$SKILL_SOURCE/install.sh" ]; then
    bash "$SKILL_SOURCE/install.sh" --platform claude
    echo "  -> llm-wiki skill installed to ~/.claude/skills/llm-wiki"
else
    echo "  !! install.sh not found in skill-source/. Skipping skill install."
    echo "  Run: git submodule update --init --recursive"
fi

# 2. 写入知识库路径
echo "[2/3] Setting knowledge base path..."
echo "$KB_PATH" > ~/.llm-wiki-path
echo "  -> ~/.llm-wiki-path = $KB_PATH"

# 3. 验证
echo "[3/3] Verifying..."
ERRORS=0

if [ ! -f "$KB_PATH/.wiki-schema.md" ]; then
    echo "  !! knowledge-base/.wiki-schema.md not found"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -d "$HOME/.claude/skills/llm-wiki" ]; then
    echo "  !! ~/.claude/skills/llm-wiki not found"
    ERRORS=$((ERRORS + 1))
else
    echo "  -> Skill installed: $(ls ~/.claude/skills/llm-wiki/SKILL.md 2>/dev/null && echo 'OK' || echo 'MISSING')"
fi

ENTITY_COUNT=$(find "$KB_PATH/wiki/entities" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
SOURCE_COUNT=$(find "$KB_PATH/wiki/sources" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
echo "  -> Knowledge base: ${ENTITY_COUNT} entities, ${SOURCE_COUNT} sources"

echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "=== Setup complete! ==="
    echo ""
    echo "Usage:"
    echo "  cd $REPO_ROOT && claude    # Start Claude Code"
    echo "  open -a Obsidian $REPO_ROOT  # Open in Obsidian (macOS)"
else
    echo "=== Setup completed with $ERRORS warning(s) ==="
fi
