# Writing and maintaining release notes

## Build the change inventory

Use the previous **published** tag and an exact candidate SHA. Inventory the
commits actually reachable in that range; a PR search by date is a discovery
shortcut, not proof that a change belongs to the release. Resolve PR number,
author and merge commit, including PRs merged through another feature branch.
Deduplicate the complete list and derive counts/contributors from that inventory.
Respect the repository's `.github/release.yml` exclusions.

For an incremental update, compare the existing draft's documented scope with
the new candidate. Add newly included PRs once and keep prior editorial work.
Verify final dependency pins and compatibility claims against current code:
superseded release-candidate versions or old SDK restrictions must not remain
as the release's current contract. Historical recordings keep their original
runtime labels even when the release dependency changes.

## Edit an existing draft safely

Locate the intended release rather than creating a second draft. Save its body,
release ID, actual `tag_name`, publication state and asset inventory before editing.
A draft may have an `untagged-*` tag; do not assume the intended version is already
its usable CLI identifier, and do not create a tag merely to edit notes/media.

Prepare the replacement body locally. Re-read the release before writing and
reconcile any intervening changes instead of replacing another editor's work.
Use a structured API argument or JSON/body file so newlines, HTML and URLs survive
unchanged. After saving, check the body, intended draft state and retained assets.
Do not silently change the candidate commit during a narrow layout/text edit.

## Keep the public narrative short

- Lead with the resulting capabilities and user-visible behavior. Highlights
  need not repeat a long introductory feature list.
- Group demos by meaningful subject. A small table works for comparable tasks,
  robots or sensor outputs; preserve the user's ordering and aspect ratios.
- Keep the curated sections selective. A feature illustrated in Highlights does
  not also need the same paragraph in New Features and Improvements. Put minor
  refactors and exhaustive entries in the folded complete PR list.
- Preserve migration actions, dependency upgrades and material limitations when
  shortening. Distinguish actual API removals from retired development-only paths.
- Fold long robot/end-effector galleries, detailed compatibility notes and the
  full PR list when useful. Do not remove distinct demos merely because they
  share a robot or scene style.
- Prefer one short provenance statement over per-frame recording details,
  machine journals, repeated test counts or broad qualification disclaimers.
  Keep numerical claims only when they explain the feature and retain their
  actual source, code/runtime scope and comparison conditions.

Updating notes does not require rerunning unchanged simulations or training.
Do not describe an existing PR recording as a fresh release-wide evaluation.

## Choose and deduplicate demo media

Treat **presentation** and **hosting** separately. A video can render inside the
body while living in an existing PR attachment, outside Release Assets.

For each demonstration, choose one primary representation:

| Requested experience | Preferred presentation |
| --- | --- |
| Click-to-play original footage | MP4 player, preferably reusing the original PR attachment |
| Automatic looping preview | GIF; do not promise MP4 autoplay/loop support |
| Bulk/source download only | A download link without a second inline player |

Honor an explicit request for both GIF and MP4, but do not introduce that duplicate
layout as the default. When asked to deduplicate, consolidate repeated GIF/player
versions, bare-video URLs and download links that render extra players, and
compilations repeating individually displayed clips. Keep distinct scenes,
models, passive demos and learned-policy inference identifiable.

For original PR videos, reuse the canonical attachment URL and attribution.
Avoid copied, expiring signed CDN URLs. For new media, use an available supported
GitHub attachment mechanism scoped to the intended repository; inspect installed
tool capabilities rather than assuming a particular CLI supports upload flags.
Do not post unrelated PR/issue comments or rewrite someone else's PR merely to
obtain a hosting URL without authorization.

An ordinary MP4 download URL (including a Release Asset or external model host)
is not proof that GitHub will embed a player. Test the intended route/markup.
A typical player cell is:

```html
<video src="ATTACHMENT_URL" controls width="400"></video>
```

Replace `ATTACHMENT_URL` with the canonical URL returned by the attachment host.
GitHub may sanitize dimensions and autoplay/loop attributes; validate the rendered
result instead of promising browser behavior. Do not store tokens or temporary
signed URLs in release text.

Prefer the original MP4 over transcoding its preview. When only GIF survives,
an MP4 conversion retains that preview's quality and timing; label it as such.
Resizing/resampling must not silently crop the task, accelerate playback, or be
presented as rerunning the simulator.

## Validate replacements and clean attachments

Check media dimensions, duration and decoding; verify downloaded/uploaded bytes
with size and SHA-256 when transferring files. Preview the actual Markdown layout
(for example with GitHub's Markdown rendering API), checking the rendered players
and their sources, not just the number of `<video>` tags in the input. Links can
expand into additional players. Exercise playback when introducing a new host
or upload route. An authenticated download alone does not prove anonymous access
for a public release; verify access for the intended audience, including after
publication.

For a requested cleanup:

1. Identify the exact asset IDs and all body references, including tables,
   thumbnails, folded sections and download links.
2. Save a recoverable source copy before deleting the last hosted copy. Verify
   any replacement and update references before deleting the selected assets.
3. Re-read the draft to detect concurrent changes. Delete only the requested,
   now-unused files; do not equate all Release Assets with redundant demos.
4. Verify the final inventory, body links, distinct players and publication state.

A note-edit request alone does not authorize deleting hosted originals. Preserve
files when only removing a repeated display. Archives/ZIP bundles are separate
requests, not an automatic part of release-note maintenance.
