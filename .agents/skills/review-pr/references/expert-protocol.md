# Expert review protocol

## Lead preparation and dispatch

1. Pin the review target: repository, base/head commits (or a local diff
   snapshot including untracked files), changed-file inventory, and intended
   behavior. All reviewers must use the same delta. Read PR files at the pinned
   head, not an unrelated working tree. If the target changes, reconcile the
   affected passes before calling the review complete.
2. Select relevant role IDs using `reviewer-perspectives.md`. Usually two to
   four roles suffice for a substantive change; a narrow change may need only
   one. This is not a quota: cover every affected contract, with bounded scopes
   and intentional overlap at integration boundaries.
3. When subagents are available and permitted, dispatch independent selected
   passes using the host's subagent tool (for example `spawn_agent`). Respect
   concurrency limits, queue remaining roles, and keep the lead working on
   uncovered files and integration while experts run. Do not substitute
   user-owned tasks/threads for subagents. Tool availability alone does not
   override host restrictions or a user's request for single-agent review.
4. Give each expert the packet below. Prefer a fresh context containing raw
   artifacts rather than the lead's candidate findings or another expert's
   conclusions. Independent first passes reduce anchoring; exchange evidence
   later when resolving a concrete disagreement. An expert may inspect contract
   neighbors beyond its initial file list when needed to establish reachability.

Use a task packet with these fields (Markdown is sufficient):

```text
Assignment: delegated expert pass; do not delegate further.
Role: <stable role ID and its responsibilities from SKILL.md>
Target: <repository; base/head SHA or snapshot; diff access>
Scope: <changed files/contracts; questions to investigate, without predicted answers>
Context: <applicable AGENTS.md; canonical skill directory; matched context paths>
Constraints: read-only review; no Git mutation, edits, or remote publication;
             follow the skill's proportional validation limits.
Return: the expert result contract below, including zero findings and limitations.
```

An expert loads its assigned role, applicable review contract, matrix sections,
and evidence standard from the canonical skill; it must not restart the lead's
dispatch workflow. The lead alone owns the final report. Remote publication,
when separately requested, remains governed by SKILL.md.

## Expert result contract

Every selected role returns a result, even when it finds no actionable defect:

- **Identity and target:** role ID, actual agent ID when available, reviewed
  base/head or snapshot identifier.
- **Coverage and status:** `complete` or `partial` for the assigned scope;
  inspected changed files and important callers/contracts; explicit omissions.
- **Candidates:** use stable local IDs such as `robotics-1`. For each, give
  proposed priority, smallest changed location, reachable input/state, violated
  contract, observable impact, and supporting code or reproduction. Explain
  guards/tests checked, how behavior differs from the base, and a correction
  direction when useful. Return `none` rather than inventing a finding.
- **Validation and limitations:** actual commands/results, checks not run,
  unresolved questions, and unavailable runtime or dependency surfaces.

This is an evidence summary, not a request to expose private reasoning. Do not
require one defect per expert. A complete static pass may still disclose that
GPU/hardware execution was not tested; distinguish that limit from files or
contracts the expert could not inspect.

## Lead verification and synthesis

Wait for every dispatched result, or record its failure and missing scope.
Check that each result used the pinned target and covers the assigned contract;
request a focused follow-up for an omitted scope or unsupported claim. Do not
label a launched, timed-out, or missing reviewer as complete.

For each candidate, the lead independently checks the cited code and reachable
scenario under SKILL.md's proof standard. Agreement between experts is not
proof, and a minority finding may still be correct. Deduplicate by root cause,
retain contributing candidate IDs, and assign final priority by demonstrated
impact. Record each candidate as accepted, merged into another candidate,
rejected with a short evidence-based reason, or unresolved. Unresolved claims
belong in open questions or residual risks, not confirmed findings.

The lead checks the full changed-file inventory against expert and lead
coverage, reviews remaining integration gaps, then produces the findings,
findings-summary, review-summary, and role-coverage tables from SKILL.md.
The coverage table records actual executors, results, and candidate dispositions;
it must not imply that naming a role proves an independent review occurred.

## Fallback and interrupted work

If subagent tools are unavailable, prohibited, explicitly disabled by the user,
or fail, disclose the specific reason. Execute each remaining selected role as
a separate lead pass with the same result contract and explicit role scope.
Label this `single-agent fallback`; if any independent expert results were
also used, label the overall run `mixed`. Do not claim independent contexts
or fabricate agent IDs for lead passes.

Temporary capacity limits normally require queuing, not silent fallback.
After a dispatch or worker failure, one bounded retry is reasonable; otherwise
complete the missing scope locally or report it as incomplete. Preserve valid
results and rerun only missing coverage. If the task ends with an unreviewed
scope, report `partial`/`unavailable` and the reason, and apply SKILL.md's
incomplete-review assessment. Never translate an unavailable expert into
"no findings" or an approval recommendation.
