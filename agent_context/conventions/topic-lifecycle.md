# Agent Context Topic Lifecycle

Use these rules when a request changes project context, adds a new topic, or updates an existing topic.

## When code behavior changes

If a change affects an existing routed topic, update the matching files under `agent_context/topics/...` in the same change.

Typical triggers:

- entry-point changes
- lifecycle/state-machine changes
- config field changes
- manager or functor signature changes
- new solver, sensor, or robot added

## How to refresh an existing topic

1. Identify the topic in `agent_context/MAP.yaml`
2. Re-read the current source-of-truth files listed in that topic entry
3. Rewrite the topic markdown so it reflects the current implementation, not old intent
4. Keep the file concise and operational:
   - entry points
   - invariants
   - common failure modes

Explicit refresh requests such as `refresh <topic> context` or
`根据当前实现重写 <topic> 上下文` should follow this exact flow.

## How to create a new topic from current code

1. Pick a stable kebab-case topic id
2. Write one Markdown file under `agent_context/topics/<topic-id>/`
3. Summarize the current behavior from source files, not from stale notes
4. Add a topic entry to `agent_context/MAP.yaml` with:
   - `id`
   - `title`
   - `aliases`
   - `keywords`
   - `paths`
   - `source_of_truth`
   - `related_topics`
   - `status`
5. If the routing rule or recommended usage changed, update:
   - `AGENTS.md`
   - `.agents/skills/project-dev-context/`
   - `.claude/skills/project-dev-context/SKILL.md`
   - `.github/copilot/project-dev-context.md`

## Rule

Do not let topic markdown drift from the code. If the routed context becomes stale, treat that as part of the same maintenance task.
