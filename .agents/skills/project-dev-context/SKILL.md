---
name: project-dev-context
description: Use when a request asks to reference, refresh, write, or register project development context so the agent resolves the topic through agent_context/MAP.yaml and reads or updates the mapped Markdown context files.
---

# Project Dev Context

Use this skill when:
- the request says `reference project development docs`
- the request says `reference project context`
- the request says `refresh project context`
- the request says `update project context`
- the request says `write project context`
- the request says `参考项目开发文档`
- the request says `参考项目上下文`
- the request says `刷新项目上下文`
- the request says `更新项目上下文`
- the request says `写项目上下文`
- the request names a known project topic such as `env-framework`, `manager-functor`, `ik-solvers`, or `atomic-actions`

## Start here

- Read `agent_context/MAP.yaml` first
- Read `references/context-system.md` for routing rules
- Read `agents/openai.yaml` for the canonical agent metadata
- Read `agent_context/conventions/*.md` when creating or updating context files

## Workflow

1. Resolve the topic through `agent_context/MAP.yaml`
2. Match in this order: exact `id`, then `aliases`, then `keywords`
3. Choose the operation mode:
   - **read**: load only the matched Markdown files under `agent_context/`
   - **refresh existing topic**: re-read `source_of_truth` and rewrite the mapped topic Markdown so it matches current implementation
   - **add new topic**: write a new topic Markdown file and register it in `agent_context/MAP.yaml`
4. Load `agent_context/conventions/*.md` if you add or update context files
5. Do not re-read `docs/source/` unless the user explicitly asks for Sphinx documentation

This skill routes context. It does not replace the underlying source-of-truth files listed in each topic entry.

## Explicit refresh mode

Use explicit refresh mode when the request is phrased like:

- `refresh <topic> context`
- `update <topic> context`
- `根据当前实现刷新 <topic> 上下文`
- `重写 <topic> 项目上下文`

In refresh mode:

1. Resolve the topic in `agent_context/MAP.yaml`
2. Re-read the files listed in `source_of_truth`
3. Rewrite the mapped topic Markdown from current implementation, not stale notes
4. Update `aliases`, `keywords`, `paths`, `related_topics` if needed

## Update contract

If code behavior changes a routed topic, update all relevant pieces in the same change:
- the matching file under `agent_context/topics/...`
- `agent_context/MAP.yaml` if topic metadata changed
- `AGENTS.md` if routing guidance changed
- `.agents/skills/project-dev-context/references/context-system.md` if routing behavior changed
- `.claude/skills/project-dev-context/SKILL.md` if Claude adapter wording changed
- `.github/copilot/project-dev-context.md` if Copilot adapter wording changed

## Source-of-truth

This skill does not store the project knowledge itself. The canonical project context lives in:
- `agent_context/MAP.yaml`
- `agent_context/topics/**/*.md`
- `agent_context/conventions/*.md`

## Map schema

`agent_context/MAP.yaml` topic entry fields:

- `id` — stable kebab-case identifier
- `title` — human-readable title
- `aliases` — alternate names for matching
- `keywords` — search terms for fuzzy matching
- `paths` — Markdown files under `agent_context/` to load
- `source_of_truth` — source code files that define the behavior
- `related_topics` — other topic ids for cross-reference
- `status` — `active` or `deprecated`
