# Baseline Limits

The public implementation is pinned to commit
`2620929c82132df130ebeb43e69a2eb96791cba0`. Task Engine must not patch or
subclass its execution Session/Bridge to conceal a missing behavior.

## Registered Calls Do Not Inherit Phase Protection

The public compiler grounds terminal effects for registered calls, but its
phase acquisition/release gates and in-flight held-object guards are generated
only for the exact built-in Pick, Place and HandOver call types. Selecting the
same Atomic Skill descriptor does not select the same protection contract.

A CPU CLI probe uses the real SemanticCallCompiler, the real GenSim Pick
lowerer, and unchanged public CPU scene/robot fixtures. Both calls ground to
PickUp with a terminal effect monitor:

| Semantic Call | Acquisition gates | In-flight held guards |
| --- | --- | --- |
| Built-in Pick | destination_acquired | destination_attached |
| gen_sim.pick.probe | none | none |

Reproduce from the repository root:

```bash
/home/dex/miniconda3/envs/embodichain040/bin/python -B \
  outputs/gen_sim/baseline_only_audit_JJ4DKu/registered_phase_protection_probe.py
```

The adjacent registered_phase_protection.log records the result. This is
grounding evidence, not a simulated dropped-object rollout. The probe checks
that public implementation files still match 2620929c before running.

Affected task routes include constrained E2 and upright E1 acquisition,
registered relative/stack placement, coordinated E5, and effect-preserving
registered motion. Built-in HandOver retains its receiver acquisition gate
and source/destination guards. A failed terminal effect or post-policy still
blocks later calls; this does not prove interruption during the current call.

SemanticLowering has no field for a registered phase gate or held guard.
Registered lowerers cannot replace curated semantics or return skill options;
InvokeCfg has no per-call policy selection. Task Engine must not fill this gap
by replacing the compiler, runner, Session or Bridge, modifying private
invocation state, or treating a terminal stability check as an in-flight guard.

Retaining built-in calls where their declared inputs suffice is preferable,
but there is no built-in coordinated E5 call. Complete qualification therefore
requires a separate decision on the public extension contract. Normal physical
successes, including the frozen multi-seed runs, are not evidence that this
failure-path requirement is met. Migration of the missing protection is stopped
at this boundary; no private workaround is installed.

## Initial Empty-Plan Recovery

A CPU-only probe using the real AtomicActionEngine and ExecutionSession was
rerun after restoring the public files. It exercised three distinct cases:

| Initial plan | Retry budget | Observed result |
| --- | --- | --- |
| Valid | 1 | A command is emitted normally |
| Empty failure | 0 | FAILED, with no command emitted |
| Empty failure, then valid | 1 | The second plan raises ValueError before execution |

The third case reports:

```text
Recovery replans must preserve endpoint tracking source fingerprints and projector routes; start a new invocation to change feedback ownership.
```

The empty first plan has not established tracking ownership, but the baseline
continuity check rejects the first valid replacement's tracking routes.
Evidence is retained in the workspace under
`outputs/gen_sim/baseline_only_audit_JJ4DKu/initial_empty_plan_recovery.log`.
The earlier standalone reproduction is in
`outputs/gen_sim/baseline_gate_2620929c/reproduce.md`.

Passing a normal physical rollout does not prove this recovery case works.
No local Session override, private eligibility update, retry-budget reduction,
or hidden execution retry is used as a substitute for fixing the baseline.

The frozen full task2_1 seed-1 run also reached this defect, without selecting
or restarting an execution attempt. Its first gen_sim.pick.step_01 plan had
320 filtered grasp proposals but no feasible pre-grasp candidate. A replan
then triggered the tracking-continuity ValueError. The public workflow recorded
semantic_call_failed, deactivated the row and did not run later calls.
The raw report is retained at
outputs/gen_sim/task2_1/20260906_180545/attempts/scene_0001/action_attempts/action_0001/execution_report.json.
The audit wrapper's generic error about a successful single attempt also covers
nonzero CLI exit codes; it must not be interpreted as evidence of a retry here.

## Grasp Binding Port

The baseline GraspGoal accepts an explicit grasp pose. Its PickUpOptions
contains downstream_object_target_poses and one shared (3,) world approach
direction. It has no release-clearance fields, object-axis approach weight,
or upper_half region semantics. Task-owned filtering now supplies that region
and release screening through the existing sampling interface, not new Goal
fields. Existing normal Pick look-ahead remains compiler-owned; the unused
custom registered-Pick future-target declaration was removed.

The current CPU probe in outputs/gen_sim/baseline_grasp_mask_probe.py uses the
unchanged public PickUp planner with six- and seven-joint mock robot layouts.
It verifies mixed success/failure and all-failure candidate rows, no commands
for failed rows, and no planning-time TaskState mutation. This is interface
evidence, not physical qualification of those robot layouts.

The baseline rejects skill_options returned by registered lowerers. Thus a
task's approach policy must be bound before execution, using task-scoped call
aliases and baseline presets. This does not recreate the withdrawn Bx3
per-environment approach option. All final physical runs must use regenerated
bundles with the current adapter contract.
