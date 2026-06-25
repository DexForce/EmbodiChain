---
name: release
description: Use when preparing, drafting, validating, or publishing an EmbodiChain release, including GitHub release notes, version tags, and release artifacts.
---

# Release

Use this skill when the task is about an EmbodiChain release flow:

- drafting or reviewing GitHub release notes
- comparing a new release against previous tags
- checking release artifacts, versioning, or docs publication
- preparing the tag/release payload for `v*` releases

## Start here

1. Read `references/release-patterns.md`.
2. Identify the target version and the previous released tag.
3. Use the release workflow in `.github/workflows/` as the source of truth for build and publish behavior.

## Release note shape

EmbodiChain releases consistently use:

- a GitHub release title that may be short, e.g. `V0.2.2`
- a first body heading like `EmbodiChain v0.2.2 Release Notes`
- `Highlights`
- `Breaking Changes` when needed
- `New Features`
- `Improvements`
- `Bug Fixes`
- `Documentation`
- `All Commits Since <previous tag>` when the release is substantial
- a short contributor summary

Keep bullets short and action-oriented. Prefer the same wording style used in prior releases: feature-focused, PR-linked, and tied to the author.

Use this exact bullet pattern for release-note items whenever attribution is known:

- `Added emissive light mode to randomize_indirect_lighting by @yuecideng in #274`

Format rules:

- lead with the past-tense change description
- append `by @<author> in #<pr>` for each merged PR bullet
- keep `All Commits Since` entries attributed the same way
- if a change has no PR number yet, resolve it before drafting the release notes

## What to check

- version bump is consistent with the previous tag
- release notes mention dependency upgrades and compatibility breaks explicitly
- docs changes are included when the release affects navigation, APIs, or public workflows
- release build artifacts are produced for tagged pushes
- multi-version docs behavior is preserved for new tags

## Output guidance

When asked to draft a release, produce:

1. the release title
2. a concise summary paragraph
3. sectioned bullets in the repo's existing style
4. any release blockers or verification notes
