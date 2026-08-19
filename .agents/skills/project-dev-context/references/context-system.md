# EmbodiChain Agent Context System

EmbodiChain keeps agent-facing context in `agent_context/`, indexed by
`agent_context/MAP.yaml`. Agent skills are stored under `.agents/skills/`.
Claude Code project adapters use `.claude/skills/<skill>/SKILL.md`, and
GitHub Copilot adapters under `.github/copilot/` should stay thin.

## Operation Modes

- `navigate`: locate current files, symbols, configs, defaults, entry points,
  registration paths, and recommended change sites.
- `read`: load an existing topic without changing it.
- `refresh`: rebuild an existing topic from its current source of truth.
- `add`: create and register a new topic from current source code.

## Routing Rules

1. Read `agent_context/MAP.yaml` first.
2. Resolve the requested topic by exact `id`, then `aliases`, then `keywords`.
3. For read requests, load only the matched Markdown files listed in `paths`.
4. For navigation requests, verify the relevant mapped facts against the
   current `source_of_truth`.
5. If navigation does not match a topic, search the working tree with
   `rg --files` and `rg -n`. Inspect package exports, config loaders,
   registries, tests, `pyproject.toml`, and `embodichain/__main__.py` as
   relevant.
6. Do not create a topic for a one-off unmatched lookup unless the user asks
   for it.
7. Do not read `docs/source/` unless the user explicitly asks for Sphinx
   documentation.

Navigation answers should identify the owning entry point, the call or config
resolution path, the recommended change site, and the focused validation
surface when those details are relevant.

## Update Rules

When behavior covered by a context topic changes, update the topic Markdown and
`agent_context/MAP.yaml` metadata in the same change. If routing behavior itself
changes, update:

- `.agents/skills/project-dev-context/SKILL.md`
- `.agents/skills/project-dev-context/references/context-system.md`
- `AGENTS.md`
- `.claude/skills/project-dev-context/SKILL.md`
- `.github/copilot/project-dev-context.md`
