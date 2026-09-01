---
name: review-pr
description: Review EmbodiChain pull requests, branches, commits, patches, or working-tree diffs for correctness regressions, architecture-contract violations, compatibility risks, unsafe resource behavior, and missing tests. Use when asked to review, audit, inspect, assess, or approve an EmbodiChain change; produce prioritized, evidence-backed findings without modifying the change unless the user explicitly asks for fixes.
---

# Review EmbodiChain Changes

Perform a read-only, defect-focused review. Trace each change through the
project's simulation, environment, task, planning, learning, configuration,
and packaging boundaries before deciding whether it is safe.

## Review contract

- Treat review and implementation as separate tasks. Do not edit files,
  approve a remote PR, submit comments, or change Git state unless explicitly
  requested.
- Report actionable defects introduced or exposed by the reviewed change.
  Ignore cosmetic preferences and issues enforced mechanically by Black unless
  they hide a correctness problem.
- Support every finding with a reachable scenario, the violated contract, and
  a concrete impact. Do not promote speculation to a finding.
- Read enough surrounding code, callers, registries, configuration loaders,
  tests, and documentation to disprove a candidate issue before reporting it.
- Review the tests as production code: confirm that they exercise the intended
  behavior, fail on the old behavior when relevant, and do not merely mirror
  the implementation.
- Continue through the full diff after finding an issue. For a large PR, keep a
  file or subsystem coverage ledger and review dependency foundations before
  their consumers.

## 1. Resolve the review target

Read the applicable `AGENTS.md` instructions first. Determine the exact delta
from the user's request and available metadata.

For a local working tree, inspect all tracked, staged, and untracked changes:

```bash
git status --short
git diff --stat
git diff
git diff --cached
git ls-files --others --exclude-standard
```

For a branch or commit range, determine the real base branch from PR metadata,
the upstream configuration, or the user's request. Record any assumed base,
then use its merge base:

```bash
git merge-base <base> HEAD
git diff --stat <merge-base>...HEAD
git diff --find-renames <merge-base>...HEAD
```

For a GitHub PR, read its title, body, base/head branches, commits, changed
files, checks, and diff with a read-only GitHub connector or `gh pr view` /
`gh pr diff`. Do not checkout, pull, rebase, or fetch merely to perform a
review. For a stacked PR, review only the layer's base-to-head delta; mention
dependency assumptions separately.

If the target remains ambiguous and different choices would materially change
the findings, ask for the base or PR identifier instead of guessing.

## 2. Build a change model

1. State the intended behavior in one or two sentences from the request, PR
   body, issue, tests, and diff. Treat the description as intent, not proof.
2. Inventory changed files by subsystem and identify changes to public APIs,
   serialized configuration, registration, defaults, execution order, tensor
   contracts, resource ownership, and package contents.
3. Read `agent_context/MAP.yaml`. Match changed symbols and paths to topic IDs,
   then load only the matched topic files from each topic's `paths`. Verify
   relevant facts against current `source_of_truth` files. Follow
   `related_topics` only when the diff crosses that contract boundary.
4. If no topic matches, use `rg --files` and `rg -n` to locate callers,
   implementations, registries, exports, tests, config loaders, and entry
   points. Do not read `docs/source/` unless public documentation is part of the
   review surface.
5. Read the common checks and only the affected subsystem sections in
   [references/review-matrix.md](references/review-matrix.md).

Do not review a changed function in isolation. Trace at least one complete
resolution path from external input or configuration to the changed behavior
and its observable output, error, state transition, or side effect.

## 3. Run focused review passes

Apply every relevant pass, using the matrix for subsystem-specific contracts.

### Correctness and state

Check happy paths, boundary values, invalid input, partial batches, exception
paths, state transitions, mutation and aliasing, ordering, defaults, and stale
caches. For tensor code, verify shapes, indexing, dtype, device, broadcasting,
autograd behavior, and empty or singleton batches. For robotics code, verify
units, coordinate frames, joint/link identity, limits, and timing.

### Integration and architecture

Trace imports, `__all__`, registries, entry points, factory dispatch,
configuration composition, task discovery, semantic lowering, package data,
and CLI paths. Check both sides of every changed contract, especially when the
producer and consumer live in different packages or configuration files.

### Compatibility and migration

Look for unannounced breaks to public Python APIs, YAML/JSON schemas, saved
checkpoints or datasets, environment IDs, defaults, component ownership, task
package discovery, and third-party extension points. Accept a breaking change
only when it is intentional, consistently implemented, documented, and covered
by migration or clear failure behavior appropriate to the project.

### Resources, concurrency, and performance

Check simulator and renderer lifecycle, cleanup queues, GPU/VRAM ownership,
device initialization, asynchronous writers, process boundaries, cancellation,
safe stop behavior, locks, deterministic seeding, and error cleanup. Flag
performance only when the diff introduces a material algorithmic, allocation,
synchronization, or per-environment regression on a reachable hot path.

### Validation and documentation

Check whether focused tests cover the changed ownership boundary, including a
negative or failure path when validation logic changed. Select the smallest
read-only command that can confirm or refute a candidate issue; do not run the
full suite by default. Start with static evidence, and do not launch live
simulation, GPU, renderer, or distributed tests unless the user requests them
or static evidence cannot resolve a material candidate. Before any command
likely to take more than two minutes, explain its scope and why a narrower probe
is insufficient. Keep these responsibilities distinct:

- Use `$review-pr` to analyze change safety and report defects.
- Use `$pre-commit-check` for comprehensive pre-commit gates and proportional
  validation.
- Use `$pr` to draft, label, push, or create PRs.
- Use the matching `add-*` or `update-*` skill only after the user asks to fix
  a finding.

Require public docs or agent-context updates only when the change makes those
artifacts materially incomplete or incorrect. A behavior change covered by an
`agent_context/MAP.yaml` topic must update its mapped context according to the
project context update contract.

## 4. Prove each candidate finding

Before reporting a candidate:

1. Locate the smallest changed line range that causes or exposes the problem.
2. Confirm the behavior differs from the chosen base and is not merely an
   unrelated pre-existing issue.
3. Identify a valid input, configuration, runtime state, or caller that reaches
   the line.
4. Check for guards, normalization, cleanup, retries, or downstream handling
   that may invalidate the concern.
5. Explain the observable consequence: wrong result, crash, silent data
   corruption, unsafe motion, compatibility break, leaked resource, or credible
   test/CI escape.
6. Run a narrow reproduction or test when static evidence is not decisive and
   the command is safe and proportionate.

If one of these cannot be established, omit the finding or record it as an
explicit open question. Do not use vague language such as "might break" without
the conditions that make it break.

## 5. Assign priority

- **P0 — Critical:** Unconditional or broadly reachable catastrophic impact,
  such as destructive data loss, unsafe robot behavior, or a repository-wide
  outage. Stop and surface it immediately.
- **P1 — High:** A common path is broken, a release or core workflow is blocked,
  or correctness/safety is seriously compromised. Fix before merge.
- **P2 — Medium:** A real defect affects a narrower but supported scenario,
  architecture contract, or maintainability boundary. Normally fix before
  merge.
- **P3 — Low:** A limited, non-cosmetic defect with minor impact. Fix when
  practical.

Do not inflate priority because a subsystem is important; rank the demonstrated
impact and reachability of this specific change.

## 6. Write the review

Put findings first, ordered by priority and then file order. Use one item per
root cause:

```text
[P1] Short imperative title
path/to/file.py:<line>

Under <reachable condition>, this code <incorrect behavior>. <Why existing
guards/tests do not prevent it and what impact follows>. <Concise correction
direction, when useful>.
```

Keep the cited line range tight and prefer changed lines. Make the title
specific enough to understand without opening the body. Do not include a full
patch unless the user asks for one.

Make each cited location clickable when the environment supports it. For a
remote PR, prefer an immutable head-commit blob link anchored to the smallest
changed line range. For a local target, use the environment's clickable
absolute or workspace-resolvable file link with a line number.

After the detailed findings, always provide a distinct findings-summary table.
Localize the labels to the response language, preserve the same priority and
file order as the detailed findings, and keep one row per root cause:

| Priority | Location | Defect | Impact | Recommended action |
| --- | --- | --- | --- | --- |
| `P0`-`P3` | Clickable `path:line` | Concise root cause | Observable consequence | Concise correction direction |

This table summarizes rather than replaces the evidence-backed finding bodies.
If there are no actionable findings, still render one row with `N/A` for the
priority and location and `No actionable findings` for the defect. A separate
review-summary table does not satisfy a request to summarize findings in
table form.

Then provide a compact review-summary table. Localize the labels to the
response language, keep cells concise, and use `None` or `N/A` instead of
omitting a row:

| Review item | Result |
| --- | --- |
| Review target | `<base>...<head>`, PR number, commit range, or local working tree |
| Scope | `<count>` changed files; affected subsystems |
| Findings | `P0: <n>; P1: <n>; P2: <n>; P3: <n>` |
| Merge assessment | `Block`, `Changes requested`, `Non-blocking findings`, or `No actionable findings` |
| Validation | Focused commands and outcomes, or `Not run` |
| Residual risks | Important untested or unavailable surfaces, or `None identified` |

Use `Block` when any P0 or P1 finding exists, `Changes requested` when the
highest finding is P2, `Non-blocking findings` when only P3 findings exist, and
`No actionable findings` when every count is zero.

After the review-summary table, add only the supporting sections that need more
detail:

- **Open questions / assumptions** — facts that could change the conclusion.
- **Validation details** — commands run, results, and important checks not run.
- **Residual-risk details** — untested GPU, renderer, hardware, distributed,
  or live simulation paths.

If there are no actionable findings, say so explicitly and still identify
material residual risks or validation gaps. Do not claim the change is proven
correct solely because tests pass.
