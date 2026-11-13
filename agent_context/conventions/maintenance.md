# Maintaining project context

`MAP.yaml` is the sole topic registry. The canonical
[project-context skill](../../.agents/skills/project-dev-context/SKILL.md) owns
routing, progressive reads and validation commands; its
[schema reference](../../.agents/skills/project-dev-context/references/context-system.md)
owns registry fields and checker semantics. Read these conventions only when
writing or maintaining context.

## Keep one owner

Put operational facts first: entry points, resolution and lifecycle, essential
contracts, likely failures, change sites and focused validation. Give each
cross-cutting contract one owning page; consumers keep a short boundary warning
and link there. Skills own procedures and AGENTS owns repository-wide rules.
Do not duplicate either in topic prose.

An overview is the default read, not a complete API guide. Keep enough to locate
the implementation and choose the next relevant detail. Link specialized
contracts on demand; `related_topics` and source directories are search scopes,
not automatic read lists. Exact defaults, field inventories, local algorithms,
example parameters and one-off results belong in source, tests or user guides.
Retain non-obvious frame, unit, timing, ownership and failure boundaries that
change an implementation decision. Distinguish defaults from examples and
optional dependencies from unconditional requirements.

## Names and references

Use stable kebab-case topic IDs and `topics/<id>/<id>.md` overviews. Detail
filenames describe the contract. Keep implementation paths repository-relative
in backticks and document links relative to their containing Markdown file.
All paths must exist and remain within the repository after resolving symlinks.
Every detail page must be reachable from an overview.

Aliases and keywords should identify concrete domains, including useful Chinese
phrases and distinct symbols. Avoid broad terms that swamp specific matches.
Keep prioritized implementation owners in `source_of_truth`; use `watch_paths`
only for additional change coverage. A directory already covers its descendants.

## Review source changes

Review affected context when entry points, loaders, lifecycle, public config,
serialization or resource boundaries change. Impact matches identify candidates;
they prove neither stale prose nor fresh unchanged paths. Verify facts against
current implementation and supported callers, then choose:

| Outcome | When | Action |
|---|---|---|
| No update | Existing guidance remains accurate and sufficient | State the reason in the task or PR |
| Revise | An existing fact or boundary changes | Replace its owning passage and repair consumer links |
| Add | A recurring missing domain or new durable constraint changes navigation | Extend the owner, or register a new topic and its links atomically |

Preserve useful constraints while removing duplicated explanations and expired
migration notes. A matched path or new symbol alone does not require an edit.
Do not append a PR history, add topics automatically, or use timestamps as proof
of freshness. Deprecation needs an active `replaced_by` target until links migrate.

Use the skill's structural and behavioral checks after edits. Statistics are
advisory whitespace-word counts, not token counts or length gates. Compare both
default-read cost and total content: splitting a page should remove repetition,
not merely move it. Validate representative navigation and ambiguous routes;
do not lock exact headings, prose lengths or formatting in tests.
