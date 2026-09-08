# Writing agent context

Put operational facts first: owning entry points, resolution/lifecycle,
invariants, likely failures, change sites and focused validation. Use only the
sections the topic needs; avoid reproducing entire API reference tables.

The MAP `paths` entry is a short overview. Move specialized configuration,
examples or execution details to linked files when they obscure common lookups.
Explain when to follow each link. Do not list all details in `paths` or preload
related topics.

Give each cross-cutting contract one owner. Other topics summarize the boundary
and link to it; skills own procedural scaffolds, AGENTS owns global rules, and
topics own implementation facts. Do not copy skill templates into topic prose.

Distinguish defaults from examples, optional dependencies from unconditional
requirements, and concrete tensor/frame contracts from abstract interfaces.
Avoid approximate source line numbers and unqualified benchmark claims.
Keep illustrative snippets minimal; verify behavioral examples against source.
Link human-facing Sphinx docs instead of copying them.
