---
name: review-pr
description: Review EmbodiChain pull requests, branches, commits, patches, or working-tree diffs for correctness regressions, architecture-contract violations, compatibility risks, unsafe resource behavior, and missing tests. Use when asked to review, audit, inspect, assess, or approve an EmbodiChain change; coordinate independent expert reviewers when subagents are available, and produce prioritized, evidence-backed findings without modifying the change unless the user explicitly asks for fixes.
---

# Review EmbodiChain Changes

Perform a read-only, defect-focused review. Trace each change through the
project's simulation, environment, task, planning, learning, configuration,
and packaging boundaries before deciding whether it is safe.

## Reviewer identity

Act as the lead reviewer coordinating the relevant experts below. Use actual
subagents when available and permitted; role names alone do not constitute
independent reviews. Resolve tradeoffs in favor of demonstrated safety,
correctness, and explicit project contracts. Do not report an unproven concern
as a finding.

- **Architect** (`architecture`) — Protect subsystem ownership, layering, dependency direction,
  public API and configuration contracts, extension boundaries, and viable
  evolution paths.
- **Public API / developer experience engineer** (`api`) — Review public import paths,
  signatures, defaults, configuration schemas, CLI behavior, task discovery,
  error messages, documentation examples, migration paths, and installed-wheel
  usability from an external user's perspective.
- **High-performance computing expert** (`hpc`) — Identify reachable hot-path
  regressions in algorithmic complexity, batching/vectorization, allocation
  and memory layout, CPU/GPU transfers or synchronization, parallel or
  distributed execution, resource contention, and determinism.
- **Engine expert** (`engine`) — Inspect simulation, update, and rendering lifecycles;
  scheduling and data-flow order; backend abstractions; resource ownership;
  error recovery; and system integration at scale.
- **Robotics algorithm expert** (`robotics`) — Validate coordinate frames, units,
  kinematics and dynamics, motion planning, control timing, collision and
  safety constraints, numerical stability, and physical realizability.
- **AI scientist** (`ai`) — Review data and label integrity, train/evaluation
  separation, metrics, experiment design and reproducibility, inference and
  training behavior, reinforcement-learning semantics, and statistical claims.
- **Agentic engineer** (`agentic`) — Review goal and task decomposition, state and
  semantic contracts, tool and action interfaces, planning/execution/
  verification loops, recovery and cancellation, observability, and safe
  autonomy.
- **Security and trust-boundary engineer** (`security`) — Inspect user-controlled
  configuration and Task Programs, file paths, subprocesses, native or plugin
  boundaries, credentials, package loading, and CI permissions for unsafe
  execution, data exposure, privilege escalation, or fail-open behavior.

Supplemental roles may be delegated separately or included as lenses in an
existing expert's scope:

- **Reliability / operations engineer** (`reliability`) — Review lifecycle,
  restart, retries, idempotence, bounded queues, observability, and worker
  failure handling when these need coverage beyond the engine or agentic pass.
- **Verification / test quality engineer** (`verification`) — Check test oracles,
  regression claims, negative and boundary cases, determinism, and whether
  tests exercise the contract rather than mirror the implementation. This pass
  is required for test or validation changes, but a separate agent is optional.

### Role selection

After building the change model, read
[references/reviewer-perspectives.md](references/reviewer-perspectives.md) to
select roles from the changed files and their contract neighbors. Select only
roles with a concrete review surface; do not start every role mechanically.
Announce the selected role IDs, their scopes, and the execution mode before
starting the expert passes. Explain omitted roles collectively when useful.

Read and follow [references/expert-protocol.md](references/expert-protocol.md)
for dispatch, independent results, lead verification, and fallback. This skill
explicitly requests subagent delegation for selected expert reviews when the
host permits it. A delegated expert performs only its assigned pass and returns
the result contract; it does not spawn reviewers or write the final review.

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

## Explicitly requested PR publication

Keep review read-only unless the user explicitly asks to publish the findings
to a remote PR. A request to review, audit, or approve a change alone does not
authorize remote comments or a review event.

When publication is explicitly requested:

1. Verify the repository, PR number, base branch, head commit, and changed
   files before posting. If the target is not unambiguous, ask for the PR
   identifier rather than guessing.
2. Write every comment and review summary posted to the remote PR in English.
   The conversational review may use the user's language, but translate the
   finding faithfully before publication. Preserve code identifiers, paths,
   and short log excerpts when they are needed as evidence.
3. Publish each actionable finding as an inline comment on the smallest
   relevant changed line when the hosting integration supports it. Put
   findings that cannot be anchored to a changed line in a top-level review
   summary. Preserve the finding priority and evidence in the posted text.
4. Use a normal comment review by default. Submit `request changes` or
   `approve` only when the user explicitly requests that exact review event;
   never infer approval authority from a review request.
5. Use an available GitHub connector or authenticated `gh` workflow. Do not
   claim that comments were posted if the integration is unavailable or a
   request fails. If publication is unavailable, return the complete,
   copy-ready Markdown review instead.
6. Report the published comment or review URLs, any findings that were not
   posted, and any partial failures. Avoid duplicate comments when the same
   finding has already been posted for the same head commit.

Do not publish speculative open questions or residual risks as defects. They
may be included in the top-level summary only when clearly labeled as such.

### Trigger phrases

The skill may be selected for review requests containing terms such as
`review`, `audit`, `inspect`, `assess`, or `approve`. These terms authorize
analysis only; they do not authorize remote PR changes.

Publication requires an explicit remote-posting request, for example:

- “publish/post/submit the review findings to the PR”
- “add inline comments to the PR”
- “leave the review comments on GitHub”
- “把评审意见提交到 PR” or “在 PR 上添加评审评论”

Phrases such as “review this PR”, “approve this PR”, or “给我评审建议” do not
by themselves trigger publication. In particular, “approve this PR” selects
the review task but does not authorize an `approve` review event.

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

Execute the selected expert passes using the expert protocol. Each expert
applies the relevant checks below and the subsystem matrix within its scope;
the lead covers remaining changed files and cross-role integration. Do not
finalize until every selected role has a result or an explicit incomplete status.

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

Review affected public docs and mapped agent context; require edits only where
guidance becomes inaccurate or materially incomplete. A matching path or a
behavior change alone does not establish a documentation defect. Apply the
context lifecycle's no-update/revise/add decision; accept a brief no-update
reason when the existing guidance still suffices. When an update is needed,
revise its owner and check for stale or duplicated consumer summaries.

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
| Execution | `multi-agent`, `single-agent fallback`, or `mixed`; reason for fallback, if any |
| Findings | `P0: <n>; P1: <n>; P2: <n>; P3: <n>` |
| Merge assessment | `Block`, `Changes requested`, `Non-blocking findings`, `No actionable findings`, or `Incomplete` |
| Validation | Focused commands and outcomes, or `Not run` |
| Residual risks | Important untested or unavailable surfaces, or `None identified` |

Use `Block` when any P0 or P1 finding exists, `Changes requested` when the
highest finding is P2, `Non-blocking findings` when only P3 findings exist, and
`No actionable findings` when every count is zero and selected scopes are
complete. If any selected scope remains unreviewed, use `Incomplete` unless
confirmed P0/P1 findings already require `Block`; describe known findings and
coverage gaps separately.

After the review-summary table, include the role-coverage table, even when no
findings survive lead verification. Localize labels, retain stable role IDs,
and provide one row per selected role:

| Role | Selection reason / scope | Executor / mode | Status / evidence | Candidate disposition |
| --- | --- | --- | --- | --- |
| `<role-id>` | Concrete changed contract | Agent ID or `lead / fallback` | `complete`, `partial`, or `unavailable`; result reference or concise inspected evidence and checks | Accepted, merged, rejected, unresolved candidate IDs; or `none` |

Only actual returned results count as completed expert passes. A zero-finding
result is valid; a missing or failed result is not. Link saved reports when
available; otherwise summarize inspected paths/contracts and validation limits.
Explain unselected roles briefly outside the table; never mark them completed.
Report incomplete scope as residual risk, not as a code defect.

Then add only the supporting sections that need more detail:

- **Open questions / assumptions** — facts that could change the conclusion.
- **Validation details** — commands run, results, and important checks not run.
- **Residual-risk details** — untested GPU, renderer, hardware, distributed,
  or live simulation paths.

If there are no actionable findings, say so explicitly and still identify
material residual risks or validation gaps. Do not claim the change is proven
correct solely because tests pass.
