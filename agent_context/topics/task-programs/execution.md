# Task Program assurance and execution

Read this when the request needs these details. [Topic overview](task-programs.md).

## Effect assurance and acceptance

Every `SkillPolicyPreset` explicitly selects one authority:

- `verified`: semantic state advances only from measured effect evidence;
  curated Pick/Place/HandOver calls require monitor mappings.
- `projected`: monitor mappings are forbidden and the action plan's expected
  symbolic effect is projected after command completion.

Projected execution proves the intended trajectory path completed; it does not
prove physical task success.

Keep these boundaries separate:

1. effect monitor: one Semantic Call's physical postcondition and recovery;
2. segment post-policy: environment advancement such as settling; and
3. segment validator: task or dataset acceptance.

`TaskProgramDemoBridge` never calls `env.step()`. It yields lazy
`DemoSegment` values so the ordinary Gym executor retains stepping, recording,
reward, reset, and persistence. Final success is published only after every
segment lifecycle completes normally.

The bridge declares `progress_total_steps` only for deterministic open-loop
Pick/Place segments with fixed interpolation samples and no recovery, runner
holds, feedback settling, post-policies, or parallel execution. It links only
the current segment's calls to their policy presets; progress display never
performs full workflow analysis or downstream look-ahead. Handwritten
trajectory tasks use the same `DemoSegment` field for a known fixed trajectory
and settle suffix; paths whose emitted action count depends on runtime state
remain indeterminate in both execution styles.

## Demonstration outcome and persistence

The bridge's accepted mask remains the sole segment-quality authority. The
common demo executor records it in `DemoSegmentResult.successes`, classifies the
first authoritative failure phase in `outcome_kinds`, and writes
`segment_accepted`, `segment_attempt_id`, and `continuity_id` for every real
rollout frame. Task Program segment metadata uses `task_program_id`; LeRobot
fragment sidecars copy it to the provider-neutral `source_program_id` field.

`DemoExecutionCfg` is collector-owned, not part of the Task Program language.
Its default `continuous` mode preserves episode semantics. Its
`segment_fragments` mode persists eligible naturally executed segments as
independent LeRobot episodes, with failed fragments requiring explicit opt-in.
Current execution remains fail-closed and every frame has `continuity_id == 0`;
checkpoint capture, state restore, and suffix resume are deferred. Recorder
commit, deduplication and partial-failure behavior are owned by
[data persistence](../data-pipeline/persistence.md).

## Parallel and registered calls

Parallel execution requires disjoint resource claims and runtime targets,
control-grid alignment, conflict-free symbolic writes, an authoritative
`ParallelCommandSafetyValidator`, and an explicit fail-fast barrier. Resource
disjointness alone is not physical safety evidence. Nested parallel blocks are
rejected.

A registered call payload is declarative and executable-free. The immutable
integration must provide exactly one matching lowerer factory and fingerprint
its call ID, revision, and target Atomic Skill descriptor. Use this extension
to expose shared Atomic Skills; do not place task-local motion generators in a
lowerer.

Registered lowerers can declare a typed `RegisteredPhaseProtectionKind` and
return a matching `RegisteredPhaseProtection` from `SemanticLowering`.
The compiler checks effect ownership and binds the declared endpoint/object
through the preset's existing measured-effect monitor. Acquisition gates motion
on attachment and guards subsequent held phases; release guards the input hold
and gates retreat on detachment. Release gates omit terminal geometric separation,
which can only be observed after retreat. Retention guards verified task state
without adding a terminal effect or changing symbolic-state ownership. Guard-only
calls validate their declared segment names against each active plan before
dispatch. Projected presets do not install measured protections.

GenSim registered Pick and relative Place use these phase declarations. Its
configured held-move service opts in with
`phase_protection: held_object_v1` and a matching monitor mapping; omitted
configuration keeps legacy exported bundles on their existing behavior.

## MLLM boundary

`embodichain.agents.mllm.task_program` accepts untrusted JSON, reuses the
canonical decoder/validator/compiler, and requires a trusted host adapter. The
model cannot choose integrations, robot resources, live providers, or
executable lowerers. The current frontend permits the constrained node/call
subset enforced by its decoder; Task Program itself supports the complete
language.
