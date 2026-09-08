---
name: project-dev-context
description: >
  Route EmbodiChain project-context and codebase-navigation requests through
  agent_context/MAP.yaml. Use to locate files, configs, defaults, entry points,
  registration paths or change sites; explain resolution chains; read, refresh
  or add project context. Chinese triggers include 文件在哪里、配置或默认值在哪里、
  入口或注册逻辑在哪里、应该修改哪个文件、参考项目上下文、刷新项目上下文。
---

# Project context and navigation

## Select and route

Read `agent_context/MAP.yaml` first. Select **navigate** (current code facts),
**read** (context), **refresh** (update a topic), or **add** (register a topic).
Specialized skills such as `/add-robot` own implementation; use this skill for
orientation and context maintenance.

Use the deterministic helper when matching is unclear or needs verification:

```bash
python .agents/skills/project-dev-context/scripts/context.py route 'QUERY'
```

Match a complete topic `id` first, including an explicitly named id within a
request; otherwise match aliases, then keywords. ASCII tokens have boundaries,
so `sim` does not match `simready`; Chinese phrases support substring matching.
Within the first matching alias/keyword tier, prefer the longest matching
phrase, then additional distinct matches. Equal best matches remain candidates;
never silently choose the first topic in MAP.

For ambiguity, use the request's subsystem/owning path to narrow candidates,
or inspect the specific symbol in their source files. If that still leaves
different answers, ask one concise clarifying question. For example,
`SceneManifest` belongs to both visualization and Task Program semantics;
`visualization SceneManifest` identifies the visualization topic.

## Read progressively

1. Load only the selected topic's `paths` (the overview).
2. Follow only the overview's detail links relevant to the question.
   `related_topics` is navigation metadata, not an automatic load list.
3. For navigation, inspect the relevant symbols in `source_of_truth`, expanding
   to callees/tests as needed. A listed directory is a search scope, not a
   request to read every file. Current code takes precedence over context prose.
4. If nothing matches, use `rg --files` and `rg -n` for symbols, flags, config
   keys, registries and imports. CLI ownership starts in `embodichain/cli/` and
   `pyproject.toml`; `embodichain/__main__.py` is an entry wrapper.
5. Do not read `docs/source/` unless the user asks for Sphinx documentation.

Do not load writing conventions, this skill's reference schema or agent UI
metadata for ordinary reads/navigation. Answer with the owning entry point,
resolution path, recommended change site and focused validation when relevant.

## Maintain context

For refresh/add, read `references/context-system.md` and the conventions in
`defaults.write_contexts`. Use current implementation as evidence, not old notes.
Keep one owner for each contract and link from other topics. Preserve stable
topic ids and overview paths; linked detail pages carry optional depth.

Find topics potentially affected by a code change:

```bash
python .agents/skills/project-dev-context/scripts/context.py affected --base origin/main
python .agents/skills/project-dev-context/scripts/context.py affected embodichain/lab/gym/envs/base_env.py
```

Review affected topics in the same change; this is an impact hint, not proof
that prose is stale or that unchanged paths are fresh. Update relevant prose,
source pointers, watch scopes and routing terms. Add a topic for a requested or
recurring missing domain, not automatically for every unmatched lookup.

After editing:

```bash
python .agents/skills/project-dev-context/scripts/context.py check
python -m pytest -q -c /dev/null --noconftest tests/test_agent_context_map.py tests/test_agent_context_tools.py
```

Validate representative natural-language routes before and after routing changes,
including ambiguous and unmatched requests. Thin adapters only point here;
they need changes only when their local entry hints change.
