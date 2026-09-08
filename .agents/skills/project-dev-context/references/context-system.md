# Context registry schema and maintenance

Read this for context maintenance. The routing procedure lives only in
`../SKILL.md`; the executable checker/router is `../scripts/context.py`.

`agent_context/MAP.yaml` uses schema `version: 1`:

| Field | Contract |
|---|---|
| `defaults.write_contexts` | Convention Markdown paths relative to `agent_context/`, read for writing only |
| `topics[].id` | Unique stable kebab-case id |
| `title` | Nonempty human-readable title |
| `aliases`, `keywords` | Lists of matching phrases; intentional ambiguity is allowed |
| `paths` | Default overview Markdown paths relative to `agent_context/` |
| `source_of_truth` | Prioritized repository-relative implementation files or narrow search directories |
| `watch_paths` (optional) | Repository-relative files/directories for change impact, without increasing read load |
| `related_topics` | Registered topic ids; never auto-load them |
| `status` | `active` or `deprecated` |
| `replaced_by` | Required active topic id when deprecated |

Paths must exist and stay inside their base, including resolved symlinks.
`paths` and `write_contexts` point to Markdown files. Detail pages under
`agent_context/topics/` must be reachable by local Markdown links from an
overview. Keep code paths in backticks and document links as Markdown links.

The checker validates schema, relations, paths, local Markdown links and orphan
topic pages. It does not prove source facts, frames, tensor dimensions, examples
or freshness. Review those against the owning implementation and focused tests.

`affected --base REF` compares the merge base with the checkout, including
staged, unstaged and untracked files. Deleted/renamed source paths can identify
their former topics. Run this before changing metadata to avoid losing old
watch coverage. Impact uses the union of `source_of_truth` and optional `watch_paths`;
context edits identify their owning topic as well. MAP, conventions or routing
skill/adapter changes conservatively select all active topics for review.

Add/delete/move topics atomically with their MAP entries and incoming links.
Deprecation keeps a redirect until callers migrate. No timestamp field is used
as proof of freshness; maintain relevant source changes and context together.
