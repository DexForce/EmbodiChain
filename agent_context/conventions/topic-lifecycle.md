# Context lifecycle

When code changes an entry point, loader, lifecycle, config field, signature,
serialization contract or resource boundary, review its affected context in the
same change. Use the project context helper's `affected` command to find
candidates, then inspect the changed behavior; a matched path alone does not
require a prose edit.

To refresh, resolve the topic, verify relevant `source_of_truth` symbols and
tests, update the owning overview/details, and repair metadata or incoming links.
Preserve useful constraints rather than replacing facts with background prose.

To add a recurring/requested domain, choose a stable id, verify current code,
write its overview and only necessary details, and register routing phrases,
source pointers, watch scopes and related topics in MAP.

Run `context.py check` and the focused context tests after maintenance. Routing
changes also need representative read/navigation exercises, including Chinese,
explicit ids, ambiguous symbols and unmatched requests. Validate real behavior
and broken-reference cases; do not enforce exact headings or prose lengths.

Use `status: deprecated` with `replaced_by` for a redirect, or remove the entry
and repair all links together. Update the canonical skill if routing changes;
thin adapters do not carry duplicate routing rules.
