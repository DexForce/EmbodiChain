---
name: project-dev-context
description: >
  Route EmbodiChain development-context and codebase-navigation requests through
  agent_context/MAP.yaml. Use when asked to locate files, configs, defaults,
  entry points, registration paths, or change sites; explain a code or
  configuration resolution chain; reference project context; refresh or write
  context; register a context topic; or work with a named topic such as
  simulation-system, env-framework, rl-learning, manager-functor, ik-solvers,
  or atomic-actions. Chinese triggers include 文件在哪里、配置或默认值在哪里、
  入口或注册逻辑在哪里、应该修改哪个文件、参考项目上下文、刷新项目上下文。
---

# Project Development Context and Codebase Navigation

## Start here

- Read `agent_context/MAP.yaml` first
- Read `references/context-system.md` for routing rules
- Read `agents/openai.yaml` for the canonical agent metadata
- Read `agent_context/conventions/*.md` when creating or updating context files

## Select the operation

- **navigate**: locate a file, symbol, config, default, entry point,
  registration path, or recommended change site.
- **read**: load the matched agent context without changing it.
- **refresh**: rebuild an existing topic from its current
  `source_of_truth`.
- **add**: create and register a new topic from current source code.

An explicit refresh or add request determines the mode. If the user asks to
implement work covered by a specialized skill such as `add-robot`,
`add-solver`, or `add-functor`, let that skill own the implementation and
use this skill only for orientation or mapped context.

## Route the request

1. Resolve the topic through `agent_context/MAP.yaml`.
2. Match in this order: exact `id`, then `aliases`, then `keywords`.
3. For a matched read request, load only the Markdown files in `paths`.
4. For a matched navigation request, load the mapped topic and verify the
   relevant path or behavior against the current `source_of_truth`.
5. For an unmatched navigation request:
   - list candidate files with `rg --files`;
   - search symbols, flags, config keys, registries, and imports with `rg -n`;
   - inspect the closest package `__init__.py`, config loader, registry,
     test, and entry point as relevant;
   - inspect `pyproject.toml` and `embodichain/__main__.py` for CLI or
     package-discovery questions.
6. Do not add a topic merely because navigation did not match. Propose or add
   one only when the user requests it or the missing area is recurring.
7. Do not read `docs/source/` unless the user explicitly asks for Sphinx
   documentation.

Never treat topic Markdown as a substitute for current code. For navigation,
report only paths and behavior verified in the working tree.

## Navigation answer contract

Include the parts relevant to the request:

- the entry point or owning location;
- the call, registration, or configuration-resolution path;
- the file or symbol to change;
- the focused tests or documentation affected by that change.

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
5. Load and follow `agent_context/conventions/*.md`

## Add mode

1. Choose a stable kebab-case topic id.
2. Read the current source files that define the topic.
3. Write one focused Markdown file under
   `agent_context/topics/<topic-id>/`.
4. Register the topic in `agent_context/MAP.yaml`.
5. Load and follow `agent_context/conventions/*.md`.

## Update contract

If code behavior changes a routed topic, update all relevant pieces in the same change:
- the matching file under `agent_context/topics/...`
- `agent_context/MAP.yaml` if topic metadata changed
- `AGENTS.md` if routing guidance changed
- `.agents/skills/project-dev-context/references/context-system.md` if routing behavior changed
- `.claude/skills/project-dev-context/SKILL.md` if Claude adapter wording changed
- `.github/copilot/project-dev-context.md` if Copilot adapter wording changed

## Source-of-truth

This skill stores the routing procedure, not project facts. Canonical project
context lives in:
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
