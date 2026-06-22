# Project Dev Context for GitHub Copilot

Canonical source: `.agents/skills/project-dev-context/`

## When to use

- reference project development docs
- reference project context
- refresh project context
- update project context
- write project context
- register a new project context topic
- 参考项目开发文档
- 参考项目上下文
- 刷新项目上下文
- 更新项目上下文
- 写项目上下文

## Start here

1. Use this adapter when the request asks to reference, refresh, write, or register project development context.
2. Then follow `.agents/skills/project-dev-context/SKILL.md`.
3. Resolve topics through `agent_context/MAP.yaml`.

## Update contract

Keep this file thin. If canonical routing behavior changes, update the
canonical skill first, then only adjust this adapter if Copilot needs a
different local entry hint.

