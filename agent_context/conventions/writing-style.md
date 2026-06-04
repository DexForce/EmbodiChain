# Agent Context Writing Style

Use these rules when adding or updating files under `agent_context/`:

1. Keep each context file topic-focused. One file should answer one class of request.
2. Put actionable engineering facts first:
   - entry points
   - source-of-truth files
   - invariants
   - common failure modes
3. Prefer short sections over narrative prose.
4. Record behavior, ownership, and workflow constraints; avoid long background history.
5. If a topic depends on existing Sphinx documentation, reference its path under `docs/source/` instead of copying the full document.
6. If the context changes the working route for an agent, update `agent_context/MAP.yaml` in the same change.
