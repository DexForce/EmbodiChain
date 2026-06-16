# EmbodiChain Agent Context System

EmbodiChain keeps agent-facing context in `agent_context/`, indexed by
`agent_context/MAP.yaml`. Agent skills are stored under `.agents/skills/`, and
tool-specific adapters under `.claude/skills/` and `.github/copilot/` should
stay thin.

## Routing Rules

1. Read `agent_context/MAP.yaml` first.
2. Resolve the requested topic by exact `id`, then `aliases`, then `keywords`.
3. Load only the matched Markdown files listed in the topic `paths`.
4. Do not read `docs/source/` unless the user explicitly asks for Sphinx
   documentation.

## Update Rules

When behavior covered by a context topic changes, update the topic Markdown and
`agent_context/MAP.yaml` metadata in the same change. If routing behavior itself
changes, update:

- `.agents/skills/project-dev-context/SKILL.md`
- `.agents/skills/project-dev-context/references/context-system.md`
- `AGENTS.md`
- `.claude/skills/project-dev-context.md`
- `.github/copilot/project-dev-context.md`

