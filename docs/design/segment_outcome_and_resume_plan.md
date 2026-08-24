# Program-Segment Outcome and Checkpoint Resume

- Status: outcome propagation and natural segment fragments implemented;
  checkpoint resume deferred
- Scope: Expert Program demonstration generation in simulation
- Date: 2026-08-24
- Related design: `docs/design/declarative_expert_program_plan.md`

## 1. Decision

EmbodiChain should support segment-aware data qualification and optional
checkpoint-based suffix collection, but it should not add success semantics to
atomic-action `TrajectorySegment` values.

The current layers already have the right semantic boundary:

- an atomic action verifies one semantic call through `SkillRuntime` and its
  effect monitor;
- an Expert Program segment combines runtime success, post-policy success, and
  application validators;
- `DemoSegmentResult` records the resulting per-environment outcome.

The original gaps were downstream of that decision:

1. trajectory buffers and online sampling did not expose the segment outcome as
   a dense frame annotation;
2. failed rows are permanently removed from the current program run;
3. there is no typed, auditable way to restore the entry state of a later
   segment;
4. a discontinuous restore cannot currently be represented without making the
   resulting data look like one continuous episode.

The design therefore has two parts:

1. persist the existing program-segment acceptance result as trajectory/data
   quality metadata;
2. add an opt-in simulation-only fragment collector that may restore a
   qualified entry checkpoint for the next segment after the current segment
   fails.

The default continuous-episode path remains fail-closed and unchanged.

Implementation scope for the current change stops after outcome propagation,
causal-continuity-aware sampling, and persistence of naturally executed
segments as independent fragments. Checkpoint values, simulator state restore,
and resume-after-failure execution remain future work. Consequently every
currently recorded frame has ``continuity_id == 0``.

## 2. Terminology

Three existing meanings of "segment" must remain separate.

| Term | Owner | Meaning | Success boundary |
|---|---|---|---|
| Atomic trajectory segment | `ActionPlan.segments` / `TrajectorySegment` | Named frame range such as approach, close, lift, or retreat inside one action | None; the enclosing action owns planning, recovery, and terminal effect verification |
| Program segment | `SegmentCfg` / `CompiledProgramSegment` | Logical transaction containing semantic calls, post-policies, and validators | Runtime success AND post-policy success AND all validators |
| Dataset fragment | Demo/trajectory recorder | One continuous, independently usable sequence of state-action transitions | The owning program segment was accepted and no restore occurs inside the fragment |

This design uses **entry checkpoint of segment K** to mean a physical and
symbolic state from which segment K may begin. For `K > 0`, that checkpoint is
qualified only after segment `K - 1` has succeeded. It is not the terminal
success state of segment K.

If an application literally wants to start after segment K's terminal state,
that is a different operation (`resume_after(K)`) and skips segment K. It is
outside the first implementation.

## 3. Existing canonical segment outcome

No second segment-success evaluator should be introduced. For each
participating environment row, the bridge already has the required inputs:

```text
runtime_ok
  = SkillResult.success_mask

post_ok
  = AND of every post-policy result

validator_ok
  = AND of every compiled segment validator result

accepted
  = participant AND runtime_ok AND post_ok AND validator_ok
```

The public outcome should preserve both `accepted` and the first authoritative
failure phase. A boolean alone is insufficient for diagnostics and recovery
policy.

Implemented stable outcome kinds are:

```text
succeeded
runtime_failed
post_policy_failed
validation_failed
cancelled
truncated
not_attempted
```

``restore_failed`` remains reserved for the deferred checkpoint implementation;
it is not exposed by the current result type because no restore can occur.

`DemoSegmentResult.successes` remains the compatibility boolean view. Its
source must stay the bridge's accepted mask rather than a new call to
`is_task_success()`.

### Open-loop calls

An atomic call marked `open_loop=True` proves only command completion. A
program segment containing such a call must have an explicit application
validator before its accepted state may qualify a later checkpoint. This rule
is enforced only when checkpoint capture/resume is enabled; it does not change
ordinary Expert Program execution.

## 4. Data contract

The implementation remains compatible with EmbodiChain's current LeRobot
``>=0.4.4,<0.5`` dependency. It does not require a dataset-format upgrade:
dense qualification is stored as additive numeric ``annotation.*`` features,
the fragment's segment instruction becomes its LeRobot task, and richer
program provenance stays in EmbodiChain's JSONL sidecar. This also keeps the
program-segment acceptance meaning separate from LeRobot reward/task success.

### 4.1 Dense frame annotations

Expert rollout annotations include:

```text
segment_accepted: bool
segment_attempt_id: int64
continuity_id: int64
```

`segment_accepted` is filled retroactively for the segment's complete frame span
when `_end_demo_segment_recording()` receives the terminal result. It does not
replace `valid`; `valid` continues to mean that a buffer slot contains a real
transition.

The name intentionally avoids collision with LeRobot's task/reward success
fields: this value means that the owning Expert Program segment passed runtime,
post-policy, and validator qualification.

`segment_attempt_id` distinguishes retries or repeated attempts of the same
compiled segment occurrence. `continuity_id` increments whenever state is
restored without an environment action.

String-valued provenance stays in sidecar metadata:

```text
program_segment_id
checkpoint_id
checkpoint_source
resume_reason
```

### 4.2 Sampling

The online sampler's normal segment mode selects only windows for which:

```text
valid == true
segment_accepted == true
segment_id is constant
continuity_id is constant
```

Boundary sampling must also require a constant `continuity_id`. A state restore
is never a learnable transition and must not appear inside a sampled window.

### 4.3 Persistence

Two output modes remain intentionally different:

- **continuous episode**: current behavior; commit only a causally continuous
  episode according to existing episode policy;
- **segment fragments**: every accepted segment is committed as an independent
  dataset episode/fragment, with source-program and checkpoint provenance.

A failed segment followed by a restored successful segment must never be saved
as one successful continuous episode. Episode-level `completed` and `success`
must remain false for a row that crossed a restore boundary, even if the
program cursor later reaches the end.

Failed fragments may be retained only when an existing explicit
`save_failed_episodes`/future `save_failed_fragments` policy requests them.

## 5. Checkpoint contract

### 5.1 Checkpoint value

Introduce an immutable environment-bound value conceptually equivalent to:

```python
@dataclass(frozen=True, slots=True)
class SegmentEntryCheckpoint:
    schema_version: int
    checkpoint_id: str
    program_id: str
    program_fingerprint: str
    segment_id: str
    segment_index: int
    scene_registry_id: str
    robot_profile_id: str
    environment_fingerprint: str
    physical_state: TensorDictBase
    task_state: TaskState
    metadata: Mapping[str, JSONValue]
```

The exact serialized physical-state type may reuse trajectory-state storage,
but the checkpoint abstraction is stricter than a trajectory frame.

It must contain or reconstruct:

- robot root pose, complete qpos/qvel, and controller targets;
- registered articulation root pose, qpos/qvel;
- registered rigid-object pose and linear/angular velocity;
- any task-specific attachment or constraint state required at the boundary;
- the verified semantic `TaskState` used for later call grounding;
- compatibility and provenance fingerprints.

Camera frames, rewards, runner cursors, pending effects, and command buffers are
not checkpoint state.

### 5.2 Source of checkpoints

The entry checkpoint for the next segment cannot be derived from a row whose
current segment failed. It must come from one of:

1. a previously qualified successful rollout;
2. another compatible successful vector-environment row;
3. an environment-owned deterministic state materializer.

The first implementation should use a pre-qualified checkpoint store and fail
closed when no compatible entry exists. Cross-row copying is allowed only when
environment and randomization fingerprints match exactly.

Segment 0 uses the ordinary environment reset state. Entry checkpoint K is
captured immediately before segment K starts, after the prior segment's
post-policy and validator have accepted the source row.

### 5.3 Runtime port

Stateful storage and simulator mutation remain outside the declarative Expert
Program schema. The fragment executor receives an explicit environment port:

```python
@runtime_checkable
class SegmentCheckpointPort(Protocol):
    def resolve_entry(
        self,
        *,
        program: CompiledProgram,
        segment_index: int,
        env_mask: torch.Tensor,
    ) -> SegmentEntryCheckpoint:
        ...

    def restore_entry(
        self,
        checkpoint: SegmentEntryCheckpoint,
        *,
        env_mask: torch.Tensor,
    ) -> SegmentRestoreResult:
        ...
```

`SegmentRestoreResult` owns disjoint `restored_mask` and `failed_mask`, the
full-batch merged `TaskState`, and JSON-safe provenance. A partial restore must
not mutate healthy rows.

## 6. Restore barrier

A restore is legal only after the previous segment has reached a terminal
boundary and its action iterator, cancellation handshake, post-policies, and
validators are complete.

For failed rows, the restore sequence is:

1. confirm that no runtime command, acknowledgement, or effect request is
   pending;
2. safe-hold every target armed by the failed segment;
3. resolve and validate the next segment's compatible entry checkpoint;
4. restore physical state for selected rows, including controller targets;
5. synchronize simulation-side caches and obtain a fresh measured scene;
6. install the checkpoint's verified `TaskState` for restored rows;
7. validate the restored entry state through the checkpoint port;
8. reseed pending rollout observations and trajectory pre-action state;
9. increment `continuity_id` and begin a new fragment;
10. construct a fresh Expert Program runtime/bridge for the selected segment.

A fresh runtime is preferred to mutating the completed bridge. It resets
observation-provider baselines, scene revisions, evidence collectors, command
buffers, and clocks while reusing the immutable compiled program and profile
contracts. `SkillRuntime` already accepts an initial `TaskState`; the adapter
needs to expose that existing construction parameter.

No restore step is represented as `env.step()`, and no synthetic action is
written for the state jump.

## 7. Execution modes

The implemented collector-owned configuration, rather than an Expert Program
field, is:

```python
@configclass
class DemoExecutionCfg:
    mode: Literal["continuous", "segment_fragments"] = "continuous"
    save_failed_fragments: bool = False
```

Both modes stop at the current fail-closed execution boundary. A future resume
extension should add an explicit failure policy only together with the live
checkpoint port; the current config deliberately cannot request unsupported
restore behavior.

The mode is selected through the collection API:

```python
generate_function(
    env,
    execution_cfg=DemoExecutionCfg(mode="segment_fragments"),
)
```

One source program row may produce multiple LeRobot episodes in fragment mode.
Accepted fragments are saved by default; ``save_failed_fragments=True`` is an
explicit diagnostic-data opt-in. The existing CLI ``max_episodes`` accounting
therefore remains on continuous mode until a fragment-count quota is defined.

The existing `DemoSegment.failure_policy` continues to mean batch behavior
(`batch_abort` versus `row_independent`). It must not be overloaded with
cross-segment recovery semantics.

### Future resume state machine

```text
READY(k)
  -> RUNNING(k)
  -> POST_AND_VALIDATE(k)
      -> accepted -----------------------> READY(k + 1)
      -> failed + continuous mode -------> STOPPED
      -> failed + fragment resume
           -> resolve entry(k + 1)
           -> restore + entry validation
               -> restored --------------> READY(k + 1), new continuity_id
               -> restore failed --------> STOPPED
```

Successful and restored rows may join the same next-segment batch at the
shared segment barrier. Rows that cannot be restored remain inactive.

## 8. Public result semantics

Do not redefine `DemoEpisodeResult.completed`. The current implementation adds
``successful_fragment_count_by_env`` and retains the existing continuous
success fields. Checkpoint-dependent views remain deferred:

```text
recovered_by_env
program_exhausted_by_env
```

Each current segment/fragment result additionally records:

```text
attempt_id
continuity_id
outcome_kind
```

``resumed_from_checkpoint`` and ``checkpoint_id`` are deferred with resume.

Current fields distinguish the first three natural-execution cases below;
deferred checkpoint fields will distinguish the final two:

- a fully successful continuous expert trajectory;
- a failed complete episode retained for diagnostics;
- a successful independent natural segment fragment;
- a successful independent suffix fragment restored from a checkpoint;
- a failed restore that emitted no controller commands.

## 9. Recommended change sites

The first implementation should remain outside atomic-action planning.

| Area | Recommended change |
|---|---|
| `embodichain/lab/gym/envs/demo.py` | Add execution config, outcome/provenance fields, and a fragment-oriented executor path; keep current continuous defaults |
| `embodichain/lab/gym/envs/embodied_env.py` | Retroactively write segment outcome annotations and provide a safe reseed hook after out-of-band state restore |
| `embodichain/lab/gym/utils/trajectory_state.py` | Add row-selective checkpoint capture/restore and controller-target restoration; do not silently claim unsupported constraints |
| `embodichain/lab/gym/envs/expert_program/bridge.py` | Allow one selected compiled segment to run from an explicit initial eligibility mask and `TaskState`; keep validator as the acceptance commit point |
| `embodichain/lab/gym/envs/expert_program/environment.py` | Assemble a fresh runtime from an explicit initial `TaskState` and selected segment index |
| simulation Expert Program integration | Implement the checkpoint port and post-restore synchronization/entry validation |
| dataset recorders | Persist fragment provenance and `segment_accepted` without treating a fragment as a continuous episode |
| online data engine | Filter unsuccessful segments and reject windows crossing `continuity_id` |

`TrajectorySegment`, `ActionPlan`, `ExecutionSession`, atomic recovery policy,
and semantic effect monitors require no API change for the initial feature.

## 10. Delivery phases

### Phase A: outcome propagation

1. Expose the bridge's accepted mask as the sole segment outcome.
2. Add dense `segment_accepted`, `segment_attempt_id`, and `continuity_id`
   annotations.
3. Update recorders and online sampling to consume them.
4. Keep execution fail-closed.

This phase is implemented.

### Phase B: checkpoint primitives

Deferred; not part of the current implementation.

1. Add immutable checkpoint/result values and an explicit port.
2. Extend simulation state capture/restore for selected rows and controller
   targets.
3. Add compatibility fingerprints and restored-entry validation.
4. Support only restore-safe boundaries with no unrepresentable live
   constraint state, unless a task-specific materializer handles it.

### Phase C: fragment collector

1. Commit naturally executed accepted segments independently. **Implemented.**
2. Execute one selected compiled segment through a fresh runtime. **Deferred.**
3. On failure, optionally restore entry of the next segment and continue under
   a new `continuity_id`.
   **Deferred.**
4. Preserve failed attempt metadata without promoting the collection session
   to continuous episode success. **Implemented.**

## 11. Validation surface

Implemented focused unit tests cover:

- runtime, post-policy, and validator masks combining into one accepted mask;
- retroactive per-frame segment outcome annotation;
- online sampling excluding failed segments and cross-restore windows;
- natural accepted segments being saved as separate synchronous and
  asynchronous LeRobot episodes;
- failed fragments requiring explicit opt-in;
- accepted prefix fragments not promoting whole-episode success.

Deferred checkpoint work must additionally cover:

- missing, incompatible, and stale checkpoints failing before commands;
- row-selective restore leaving healthy peers unchanged;
- controller targets matching restored qpos on the first subsequent step;
- fresh runtime receiving the checkpoint `TaskState`;
- a restored Place/Handover entry preserving required held-object state;
- restore entry validation failure remaining terminal;
- restored suffix data being saved as a separate fragment, never a successful
  continuous episode;
- safe stop and no pending buffered action at every restore boundary.

An end-to-end simulation qualification should deliberately fail one middle
segment, restore a pre-qualified next entry, complete the suffix, and verify:

1. the failed segment is not sampled as successful data;
2. no sampled chunk crosses the state jump;
3. the suffix has the correct segment and checkpoint provenance;
4. the overall continuous episode remains unsuccessful;
5. the restored suffix fragment is independently accepted.

## 12. First supported task

`ExpertProgramRepeatedPickPlace-v1` is the safest initial qualification target.
Its segment boundary occurs after Place, settling, and object-near-target
validation, where the gripper should no longer own a held-object relation.

Open Drawer should not be the first checkpoint-resume target: the passive
articulation, handle contact, and open-loop Slide semantics require stronger
entry-state reconstruction and validation. Hardware execution remains out of
scope until a device integration can provide an authoritative restore port.
