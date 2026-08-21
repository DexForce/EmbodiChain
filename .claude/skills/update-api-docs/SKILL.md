---
name: update-api-docs
description: Claude adapter for the canonical EmbodiChain update-api-docs skill.
---

# Update API Docs - Claude Adapter

Canonical source: `.agents/skills/update-api-docs/`

## When to use

- the API docs checker or CI reports missing public exports
- a change adds or updates an API declared through `__all__`
- API-reference pages and source descriptions need synchronization

## Start here

1. Run the read-only checker to discover missing import paths.
2. Follow `.agents/skills/update-api-docs/SKILL.md` to generate the docs.
