# Context names and paths

`agent_context/MAP.yaml` is the sole topic registry. Use stable kebab-case topic
ids and `topics/<id>/<id>.md` for the overview; use descriptive detail names
such as `configuration.md` or `execution.md` in the same directory.

Choose aliases for natural-language subsystem names and keywords for distinctive
symbols, flags and configuration fields, including useful Chinese phrases.
Avoid broad words such as `save`, `task` or `step` that drown out better matches.
Shared symbol names may remain ambiguous when their owners really differ.

Keep source paths repository-relative, context paths relative to `agent_context/`,
and Markdown links relative to their containing document. Use watch directories
for change scope without treating those directories as an all-files read list.
The maintenance reference in the project context skill defines the schema.
