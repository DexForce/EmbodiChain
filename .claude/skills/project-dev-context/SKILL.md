---
name: project-dev-context
description: Claude adapter for locating EmbodiChain code, configs, defaults, entry points, and change sites or for reading and maintaining registered project context.
---

# Project Dev Context - Claude Adapter

Canonical source: `.agents/skills/project-dev-context/`

## When to use

- locate a file, config, default, entry point, registration path, or change site
- explain a code or configuration resolution chain
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
- 文件在哪里
- 配置或默认值在哪里
- 应该修改哪个文件

## Start here

1. Use this adapter for codebase navigation or when the request asks to
   reference, refresh, write, or register project development context.
2. Then follow `.agents/skills/project-dev-context/SKILL.md`.
3. Resolve topics through `agent_context/MAP.yaml`.

## Update contract

Keep this file thin. If canonical routing behavior changes, update the
canonical skill first, then only adjust this adapter if Claude needs a
different local entry hint.
