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

`affected --base REF --explain` compares the merge base with the checkout,
including staged, unstaged and untracked files. Impact uses both baseline and
current `source_of_truth`/`watch_paths`, so deleted or renamed paths still identify
their former owners after metadata is updated. Context edits identify their
owning topic. Reasons distinguish source, watch, context and registry changes;
they identify review candidates, not required prose edits. Omit `--explain` to
retain the topic-ID-only output.

MAP changes compare parsed entries by stable id: additions, removals and updates
select those entries; comments, formatting and topic-list order do not. Global
MAP settings, conventions and routing skill/adapter changes select all active
topics. Removed entries remain in the report so incoming links can be repaired.
Without `--base` (or when the baseline has no MAP), a MAP path match conservatively
selects all active topics; `--explain` reports the missing baseline. An unreadable
or malformed baseline is an error, not an empty impact report.

`stats [--base REF]` reports total context Markdown files/lines/whitespace words
and per-overview word counts, including checkout additions/deletions. With a
base, deltas compare against the merge base. These are not model token counts.
Overviews above 1,200 words, or growing by more than 25% and at least 100 words,
are flagged for review without failing the command. Use the flags to inspect
reading cost and duplication, not to remove necessary invariants or enforce
fixed prose lengths. Check total content as well as overview size after splits.

Add/delete/move topics atomically with their MAP entries and incoming links.
Deprecation keeps a redirect until callers migrate. No timestamp field is used
as proof of freshness; maintain relevant source changes and context together.
