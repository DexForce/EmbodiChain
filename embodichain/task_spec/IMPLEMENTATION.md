# TaskSpec E2 implementation plan

Spec: user-supplied GenSim architecture / TaskSpec design for PR #531,
plus the accepted next-step plan in the conversation.

## Global Constraints

- Keep one public Task Program / Atomic Skills executor and existing Gym lifecycle.
- TaskTemplate owns normative goals; legacy plan hashes keep their meaning.
- Preserve v1 bundles and candidates. Version all new protocol paths explicitly.
- Only evidence-backed required checks pass. Never certify unavailable checks.
- Observe and freeze final task evaluation before successful data submission/reset.
- Keep pure numerical algorithms in compute, provider reads in lab, and task recipes in GenSim.
- Keep source changes in the existing isolated worktree; do not alter main or PR #531 directly.
- Create one PR targeting ljd/action_engine_refactor after focused tests and review.
- Do not claim GPU/physical qualification without a measured run.

### Task 1: Public execution safety prerequisites

Work in lab/sim/atomic_actions, lab/task_program compiler/integrations,
gen_sim/task_engine/_task_program/services.py (only phase protection declarations),
and associated tests/context. Do not edit GenSim coordinator, workflow, bundle,
runner or TaskSpec/evaluation modules.

Fix recovery continuity when no previous tracking ownership was established.
Test first empty -> valid, valid -> changed ownership rejected, repeated empty -> failure/no commands.
Introduce a controlled typed registered-lowerer phase-protection declaration,
validated and bound by the public compiler to existing gates/guards. Reuse
current verification policy and runtime; do not infer protection from skill name.
Bind GenSim registered Pick/Place/held-move protection where required for E2.
CPU failure-injection must exercise missing acquisition, held loss and
release-before-retreat guards. Preserve existing built-in HandOver protections.
Keep this a focused commit; black==26.3.1 . before committing, stage only owned files.
Report paths, commits, test commands/results and known physical limits.

### Task 2: E2 template and measured episode acceptance

Use TaskSpec explicit input plus an E2 seed adapter (upright with explicit threshold,
fallen/not-upright init). Reject unsupported legacy semantics without dropping them.
Derive scene requirements in current owner, carry template through coordinator and
bundle, and capture observed instance after reset/settling. Validate initial goal
and reject trivial/invalid episodes. Freeze final upright evaluation after cleanup
and before data submission/reset, preserving separate program/task results.
Reuse a shared compute tilt measurement from existing stability policies.
Only the bounded E2 contract can open v2 execution; unsupported predicates,
invariants, temporal constraints or task families stay gated.
Assemble content-addressed instance, witness and single-episode certificate references.
Keep failure evidence and do not alter public reset/retry ownership.
Validate wrong-but-stable orientation, row-local failure, stale evidence/fingerprints,
and success-data submission refusal on failed evaluation with CPU tests.

### Task 3: Feasibility reporting, review and PR

Invoke existing FeasibilityBroker in preparation; report static evidence separately
from provider-free preflight and physical execution. Do not unconditionally mark
unperformed static checks complete. Run focused tests, Black, API docs/context gates,
and available physical smoke only with verified prerequisites. Review full branch,
fix findings, commit/push own branch, and create PR targeting #531 head branch.
