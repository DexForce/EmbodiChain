# Writing agent context

Put operational facts first: owning entry points, resolution/lifecycle,
invariants, likely failures, change sites and focused validation. Use only the
sections the topic needs; avoid reproducing entire API reference tables.

The MAP `paths` entry is a short navigation overview: scope, owning entry points,
essential resolution/lifecycle, high-risk invariants, change sites and validation.
Link specialized details by the question they answer. A lookup for one entry
point should not require reading backend tuning or an algorithm implementation.
Do not list all details in `paths` or preload related topics.

Details retain durable cross-file knowledge that changes a future decision.
Exact field inventories/defaults, local algorithms, example parameters and fix
narratives belong in source, tests, tutorials or the PR. Link those owners when
needed. Before adding a paragraph, identify the decision it supports and check
whether an existing owner already explains it. Edit that passage in place.

Give each cross-cutting contract one owner. Other topics retain a short local
boundary constraint and link to the full rule; they do not repeat its defaults,
conversion tables or exceptions. A brief frame/unit warning at a consumer can
remain useful. Skills own procedural scaffolds, AGENTS owns global rules, and
topics own implementation facts. Do not copy skill templates into topic prose.

Distinguish defaults from examples, optional dependencies from unconditional
requirements, and concrete tensor/frame contracts from abstract interfaces.
Avoid approximate source line numbers and unqualified benchmark claims.
Keep illustrative snippets minimal; verify behavioral examples against source.
Link human-facing Sphinx docs instead of copying them.
