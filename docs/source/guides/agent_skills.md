# Agent Skills

EmbodiChain stores canonical in-repo agent skills under `.agents/skills/`.
Each skill directory contains a `SKILL.md` file with the source-of-truth
workflow for that task.

Tool-specific files are adapters only:

- `.claude/skills/<skill>/SKILL.md` points Claude Code project skills to the canonical skill.
- `.github/copilot/*.md` points GitHub Copilot guidance to the canonical skill.

Keep adapters thin. When a workflow changes, update the matching
`.agents/skills/<skill>/SKILL.md` first, then adjust adapters only when a tool
needs a different local entry hint.

## Available Skills

| Skill | Canonical path | Purpose |
| --- | --- | --- |
| `/add-atomic-action` | `.agents/skills/add-atomic-action/SKILL.md` | Scaffold simulation atomic actions. |
| `/add-task-env` | `.agents/skills/add-task-env/SKILL.md` | Scaffold `EmbodiedEnv` task environments. |
| `/add-functor` | `.agents/skills/add-functor/SKILL.md` | Add observation, reward, event, action, dataset, or randomization functors. |
| `/add-test` | `.agents/skills/add-test/SKILL.md` | Write tests following project conventions. |
| `/pre-commit-check` | `.agents/skills/pre-commit-check/SKILL.md` | Run local CI-style checks before committing. |
| `/pr` | `.agents/skills/pr/SKILL.md` | Draft or create pull requests. |
| `/benchmark` | `.agents/skills/benchmark/SKILL.md` | Write benchmark scripts for EmbodiChain modules. |
| Project context routing | `.agents/skills/project-dev-context/SKILL.md` | Resolve project context through `agent_context/MAP.yaml`. |

## Project Context Routing

Requests such as `reference project development docs` or `reference project
context` should use the `project-dev-context` skill. The routing flow is:

1. Read `agent_context/MAP.yaml`.
2. Resolve the topic by exact `id`, then `aliases`, then `keywords`.
3. Load only the matched Markdown files under `agent_context/`.
4. Do not read `docs/source/` unless the user explicitly asks for the Sphinx
   documentation.
