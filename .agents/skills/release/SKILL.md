---
name: release
description: Use when preparing, drafting, updating, reviewing, or publishing an EmbodiChain release, including release notes, demo media, version tags, and release artifacts.
---

# Release

Use this skill for release-note drafts and incremental updates, demo presentation
and attachment cleanup, or version/tag/build/publication work.

## Start here

1. Identify the requested operation: write/edit notes, manage media, or publish.
   Preserve the user's version, destination and existing publication authorization.
   A note or media edit alone does not authorize tagging or publishing.
2. Read [release-patterns.md](references/release-patterns.md) for repository style.
3. For drafting, updating, shortening or illustrating notes, follow
   [release-notes.md](references/release-notes.md). It covers change inventory,
   concise editing, demo presentation and attachment cleanup.
4. Identify the target version, previous released tag and exact candidate commit.
   Use `.github/workflows/` as the source of truth for build, PyPI and docs behavior.

## Release note shape

Use a title such as `V0.3.3` and a first heading such as
`EmbodiChain v0.3.3 Release Notes`, followed by a short summary and relevant sections:

- `Highlights`: major capabilities and selected demos, grouped by subject.
- `Breaking Changes`: actionable migration guidance when needed.
- `New Features`, `Improvements`, `Bug Fixes`, `Documentation`: concise, curated
  changes; avoid repeating the same explanation across sections.
- `Compatibility Notes`: dependency profiles and material backend limitations
  when they need more space than the summary.
- `All Commits Since <previous tag>`: a folded, complete PR list for substantial
  releases, followed by a short contributor summary and full-changelog link.

Use this exact attribution pattern for merged PR bullets:

- `Added emissive light mode to randomize_indirect_lighting by @yuecideng in #274`

Lead with a past-tense change, then append `by @<author> in #<pr>`. Resolve
missing PR numbers/authors before including a change. Apply the same attribution
to the complete PR list; media tables may place attribution beside the feature.

## Existing drafts and media

Read the current release by ID before editing, including its body, actual tag,
draft/publication state and attachments. Reconcile concurrent edits before writing.
Preserve unrelated content and media, and verify the saved result.

Give each demo one primary presentation. Prefer an existing PR's original MP4
attachment when a player suits the request; use GIF when automatic looping is
required. Do not add duplicate GIF/player/link/compilation displays by default.
Respect explicit requests to retain both. Removing a display and deleting a
hosted file are separate operations; clean attachments only within the requested
scope after checking references and replacement playback.

## Publication checks

For publication, verify the release commit's version, dependency metadata,
bundled extension packages and tag-build outcome. Main/PR CI success does not
establish that the tag-only distribution gate passed. Keep local checks, CI,
physical qualification and historical demo results distinct.

Check that publishing reuses validated distributions and preserves older
versioned docs. Report actual workflow outcomes rather than promising them from
configuration alone. Resolve draft-only media URLs before publication and verify
public playback/download access afterward.

Keep verbose build logs, recording journals and internal implementation detail
out of public notes. A draft may retain concise blockers; replace resolved
preparation instructions with actual release verification when publishing.

## Output guidance

For a draft/update, provide the title or saved-release link, a concise description
of the change, and any material blockers. State whether the release remains a
draft. For publication, report the release/tag and verified distribution status;
identify any remaining external-workflow limitation precisely.
