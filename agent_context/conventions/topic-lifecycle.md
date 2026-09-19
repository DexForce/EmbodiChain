# Context lifecycle

When a change affects an entry point, loader, lifecycle, public configuration,
serialization contract or resource boundary, review the affected context in the
same PR. `affected --base REF --explain` identifies candidates and why they
matched; neither a path match nor a behavior change alone requires a prose edit.

Choose one outcome and briefly record it in the PR or task result:

| Outcome | When | Action |
|---|---|---|
| No update | Existing guidance remains accurate and sufficient | State why; internal optimizations, fixes restoring an existing contract, and tutorial tuning usually belong in code/tests |
| Revise | An existing fact or boundary becomes inaccurate | Replace the owning passage and remove superseded claims; adjust consumers' boundary summaries only as needed |
| Add | A new durable constraint changes future navigation, implementation or validation decisions | Extend the owning detail; add a topic only for a recurring/requested domain |

To refresh, resolve the topic, verify relevant `source_of_truth` symbols and
tests, update the owning overview/details, and repair metadata or incoming links.
Preserve useful constraints rather than replacing facts with background prose.
Review the containing section, not just the appended paragraph: consolidate
duplicate explanations and remove obsolete exceptions, migration notes whose
compatibility path is gone, and implementation inventories recoverable from code.
Keep repair history and one-off tuning results in the PR rather than context.

To add a recurring/requested domain, choose a stable id, verify current code,
write its overview and only necessary details, and register routing phrases,
source pointers, watch scopes and related topics in MAP.

Run `context.py check` and the focused context tests after maintenance. Use
`context.py stats --base REF` to compare default reading size and total content;
splitting a page should also remove repetition, not merely relocate it. Size and
growth flags request editorial review, not mandatory cuts or a CI failure. Routing
changes also need representative read/navigation exercises, including Chinese,
explicit ids, ambiguous symbols and unmatched requests. Validate real behavior
and broken-reference cases; do not enforce exact headings or prose lengths.

Use `status: deprecated` with `replaced_by` for a redirect, or remove the entry
and repair all links together. Update the canonical skill when routing rules or
workflow change; alias/keyword edits alone do not require skill prose changes.
Thin adapters do not carry duplicate routing rules.
