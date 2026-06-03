# Agent Context Naming

Directory and index naming rules:

- Topic directories use kebab-case ids, for example:
  - `env-framework`
  - `manager-functor`
  - `ik-solvers`
- `agent_context/MAP.yaml` is the only topic registry for agents.
- Each topic entry must have:
  - `id`
  - `title`
  - `aliases`
  - `keywords`
  - `paths`
  - `source_of_truth`
  - `related_topics`
  - `status`
- Sphinx documentation lives under `docs/source/` and is the human-facing reference.
  Agent context files under `agent_context/topics/` summarize operational knowledge for agents.
