# Repeated Pick/Place Combined Generation Design

## Status and context

This is the proposed design for the complete combined-generation demo in
`embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/`. It follows
[issue #670](https://github.com/DexForce/EmbodiChain/issues/670), builds on
[PR #681](https://github.com/DexForce/EmbodiChain/pull/681), and depends on the
contracts described by `2026-09-24-pr681-foundation-hardening.md`.

The design makes the following slice concrete:

- four configurable cube initial-pose reference families (`m = 4` by default);
- four configurable Affordance branches per family (`a = 4` by default);
- four configurable spatial trajectory variants per family (`t = 4` by
  default);
- sixteen generic physical execution slots (`n = 16` by default);
- one visual profile attached to each physical rollout, with balanced visual
  assignment that does not multiply the physical candidate count; and
- local, inspectable artifacts without writing LeRobot records.

The default logical physical budget is therefore `m × a × t = 64` complete
episode recipes. The recipe count, family count, branch count, variant count,
and slot count are all configurable. The defaults describe the showcase, not
hard-coded limits in the source-neutral generation package.

This document is a design specification, not a claim that the complete
pipeline already exists. Existing `generation.demo.yaml` remains the small
T4 trajectory-only example until this design is implemented and accepted.

## Goals

The implementation will provide one runnable combined-generation vertical
slice for the repeated Pick/Place Task Program. A run must:

1. enumerate deterministic cube initial poses while leaving authored drop
   targets unchanged;
2. combine an Affordance branch and a spatial trajectory variant into one
   episode-level candidate recipe;
3. execute all three Pick/Place cycles in `program.yaml` using the recipe's
   per-cycle schedule;
4. apply exactly one visual profile to the same physical rollout, including
   all three cycles;
5. run through generic source-neutral coordinator and slot contracts rather
   than a Task Program-specific expansion algorithm;
6. execute on any compatible slot, independently of `env_id` and family
   affinity;
7. record requested choices, actual choices, fallback reasons, seeds, measured
   validation, and visual provenance; and
8. produce local artifacts that can be inspected or merged into a future
   LeRobot writer without changing candidate identity contracts.

## Non-goals

This slice does not:

- change the Task Program DSL, `program.yaml`, semantic-call vocabulary, or
  authored drop poses;
- make `motion.expansion` import Task Program, Gym, Atomic runtime, or a
  simulator;
- vary timing or control `dt`, remap phase indices, or silently retime a
  candidate;
- implement coverage-per-cost scheduling, T9 cost models, or observation
  fan-out;
- render multiple visual variants after one physical rollout;
- write physical episodes to LeRobot while its schema is being refactored;
- vary global renderer state independently for each environment in one
  `SimulationManager`; or
- claim qualified physical success merely because a configured command sequence
  completed.

## Design vocabulary

| Term | Meaning | Owning layer |
| --- | --- | --- |
| Reference family | One complete, validated initial scene state. In this task it changes only the cube's initial world pose. | Scene/reference provider |
| Affordance branch | One deterministic grasp/interaction proposal family selected during a Pick call. | Affordance proposal port |
| Trajectory variant | One spatially perturbed trajectory on the unchanged control grid. | Source-neutral expansion |
| Candidate recipe | The complete choices and provenance for one three-cycle physical episode. | Candidate coordinator |
| Visual profile | Appearance and camera parameters applied to one physical episode. | Visual profile port / environment |
| Physical slot | A reusable simulator environment row with restore, execution, observation, and validation capability. | Host execution |
| Physical geometry family | A measured-geometry lineage used for coverage accounting. It excludes visual profile identity. | Coverage/validation |
| Source adapter | A port converting one source kind into the common trajectory-template contract. | `motion.expansion` boundary |

The terms “generation” and “expansion” are intentionally different. A
Generation Profile describes the whole job. Expansion names the internal
proposal and transformation stages.

## Ownership and dependency direction

The dependency direction is:

```text
task deployment + environment
        │ resolves physical capabilities and owns env rows/sensors
        ▼
host generation coordinator ──► physical slot pool ──► measured validator
        │                                  │
        │                                  └─► visual profile port
        ▼
source-neutral motion.expansion ──► local EpisodeSink
        ▲
        │ edge adapter only
Task Program / Atomic / MotionGenerator / handwritten sources
```

### Source-neutral core

`embodichain.lab.sim.motion.expansion` owns value-only contracts and
deterministic logical work:

- `ReferenceFamilyProvider` and reference-family values;
- `AffordanceProposalPort` request/result values;
- `SourceAdapter`, `TrajectoryTemplate`, and spatial operators;
- `CandidateRecipe`/`CandidateSpec` identity and provenance;
- deterministic per-call schedules and fallback bookkeeping;
- compatibility keys, logical queues, and candidate budgets; and
- coverage values, generation records, and the generic `EpisodeSink` contract.

It never restores a simulator, calls `env.step()`, changes a camera, writes a
file, or imports Task Program/Gym/Atomic runtime modules.

### Host execution layer

`embodichain.lab.sim.motion.execution` owns:

- physical-slot reservation and release;
- full robot/scene restore and verification;
- applying a visual profile to selected environment rows;
- invoking planners and executing commands;
- measured validation and observation capture; and
- calling an `EpisodeSink`.

The host may use Task Program, Atomic, or Gym APIs through adapters, but those
dependencies do not flow back into the expansion core.

### Task Program edge adapter

`TaskProgramSourceAdapter` exposes a grounded Atomic plan as a common source
template. It does not own candidate identity, slot assignment, reset,
coverage, visual state, or persistence. A task without a Generation Profile
continues through the existing bridge unchanged.

## Task-local files and configuration ownership

The task directory will contain the following new deployment assets:

```text
repeated_pick_place/
├── generation.demo.yaml                 # existing T4 trajectory-only example
├── generation.combined.yaml             # canonical combined profile
├── generation_profiles/
│   ├── cube_initial_pose.yaml           # four legal reference families
│   └── rgb_visual.yaml                  # registered visual profiles
├── env.generation.yaml                  # 16-row environment and sensors
└── task.ur5.generation.yaml             # UR5 showcase deployment
```

The same generation environment pattern can be duplicated for Franka after the
UR5 vertical slice is stable. Existing `env.yaml`, `task.franka.yaml`,
`task.ur5.yaml`, and `program.yaml` retain their current behavior.

### Environment deployment ownership

`env.generation.yaml` owns all environment fields:

- `physics`, `num_envs: 16`, episode limits, simulation cadence, and arena;
- the dynamic cube and its fixed physical material/mass properties;
- RGB camera and observation sensor construction;
- one global base sun/background; and
- the global sun's bounded visual profile hook.

The environment config is the only owner of `num_envs`, camera resolution,
renderer, and control timing. A Generation Profile may request a registered
observation/visual capability but cannot override those physical fields.

### Canonical Generation Profile

`generation.combined.yaml` is a strict, callable-free sidecar. Its canonical
shape is:

```yaml
schema_version: 1

source:
  kind: task_program
  source_id: repeated_cube_pick_place
  source_revision: config:repeated_pick_place_v1
  unit_scope: episode
  template_id: repeated_pick_place_episode

scene_randomization:
  enabled: true
  reference_family_count: 4
  profile: cube_initial_pose
  change_scope: cube_initial_pose_only
  keep_drop_targets: true

affordance:
  enabled: true
  branches_per_family: 4
  max_proposals: 128
  fallback: next_surviving_branch

trajectory:
  variants_per_family: 4
  spatial:
    enabled: true
    operators: [joint_residual, via_points]
    joint_offset_scale: 0.05
    via_count: 1
  timing:
    enabled: false

visual:
  enabled: true
  profile_file: generation_profiles/rgb_visual.yaml
  profiles: [rgb_canonical, rgb_material_01, rgb_camera_01, rgb_light_01]
  apply_per_rollout: true
  balance: round_robin
  restore_before_next_candidate: true

observation:
  profiles: [rgb_lowres]
  capture_rgb: true
  capture_video: false

scheduling:
  policy: round_robin_fifo
  candidate_budget: 64
  reference_family_budget: 4
  queue_max_items: 64
  queue_max_bytes: 536870912

execution:
  max_inflight: 16
  compatibility: strict
  restore_initial_state: true
  cancel_policy: safe_finish_and_drain

persistence:
  sink: local_artifact
  output_dir: /tmp/repeated-pick-place-generation
  write_trajectories: true
  write_observations: true
  write_manifest: true
```

The canonical Python representation will extend the existing
`TrajectoryGenerationJobCfg` rather than introduce a task-specific config
class. The exact field names in the YAML are normative for the new profile;
the semantic rules above are normative for every decoder.
The decoder accepts exactly one of `trajectory` and the legacy
`augmentation` section. A legacy `augmentation` mapping is normalized into
`trajectory`; both sections together are rejected. Existing profiles keep
working under their current action-level, single-slot constraints.

Cross-field validation happens before simulator construction or candidate
assignment:

1. `candidate_budget >= m × a × t` for the requested default/full showcase;
2. `reference_family_budget >= m` when all families are requested;
3. `max_inflight <= env.num_envs` after the environment is resolved;
4. the initial-pose provider exposes at least `m` legal poses;
5. every visual and observation profile ID is registered exactly once;
6. every editable phase has an explicit kind and operator permission; and
7. the only permitted global visual parameter is the bounded `main_light`
   sun operation; all declarations in one batch must resolve it identically.

The profile does not encode robot joint limits, `control_dt`, backend,
`num_envs`, recorder construction, or arbitrary Python callables.

## Reference-family generation

`CubeInitialPoseProvider` returns immutable, deterministic
`ReferenceFamilySpec` values. Each value contains:

- `reference_family_id`, for example `cube_pose_00` through `cube_pose_03`;
- the cube's world `SE(3)` pose;
- the unchanged authored drop-target IDs and poses;
- a deterministic seed and profile revision;
- a scene signature; and
- optional preflight reachability/collision evidence.

Only the cube initial pose changes. Drop targets, robot initial pose, joint
order, physical parameters, and Task Program repeat structure remain fixed.
The provider samples from a bounded safe region, validates collision and
reachability, and fails pre-simulation if it cannot produce the requested
number of legal families. A family ID is part of candidate provenance, not a
physical slot identity.

The default profile contains four authored deterministic poses rather than
unbounded random sampling. A future provider may generate more poses while
retaining the same contract and seed hierarchy.

## Candidate recipe and identity

One candidate is one complete three-cycle episode. It is not one isolated
`Place` call and it must not reuse one branch or variant ordinal for every
cycle. `source.unit_scope: episode` defines the identity and scheduling unit;
the Task Program edge adapter may still export one grounded Place template at
each cycle boundary. The episode coordinator owns the recipe and passes the
cycle-specific choice into each of those action-level adapter calls.

The logical recipe contains:

```text
CandidateRecipe
  candidate_id
  job_id / profile_revision
  reference_family_id
  physical_geometry_family_id       # provisional lineage; measured aliasing may merge it
  cycle_schedule[0..2]
    pick.affordance_requested
    place.trajectory_requested
  visual_profile_id / visual_seed
  source_unit_id / source_revision
  deterministic_seed_digest
```

`CandidateIdentity` is derived from the complete source and scene identity,
not from `env_id`. The provisional physical geometry key deliberately excludes
`visual_profile_id`; visual provenance remains in the candidate digest and
artifact metadata. This permits visual balance without falsely claiming new
physical geometry coverage.

For the default budget, recipes enumerate:

```text
reference_family f ∈ [0, 4)
affordance base a ∈ [0, 4)
trajectory base t ∈ [0, 4)
total = 4 × 4 × 4 = 64

a0 = (recipe_index // T) mod A
t0 = recipe_index mod T
```

Candidate zero is the canonical nominal recipe: family zero, nominal
Affordance branch, nominal trajectory, and canonical visual profile. All other
recipes use deterministic rotations so the three Pick/Place calls exercise
different choices. One normative schedule is:

```text
recipe_index = f * (A * T) + a0 * T + t0

if recipe_index == 0:
    affordance(c) = 0
    trajectory(c) = 0
else:
    affordance(c) = (a0 + c + 2*f) mod A
    trajectory(c) = (t0 + 2*c + f) mod T

visual(recipe_index) = 0                          if recipe_index == 0
                        recipe_index mod V         otherwise
```

Here `c` is the zero-based Pick/Place cycle, and `A`/`T` are the configured
branch and trajectory counts, and `V` is the number of registered visual
profiles. Implementations may use an equivalent
balanced/orthogonal permutation, but the permutation must be deterministic,
documented in the schedule digest, and must not make candidate zero
non-nominal.

Every requested choice is recorded before proposal. If a requested branch,
trajectory variant, or visual profile is rejected, the coordinator tries the
next deterministic survivor in the configured order. The record contains the
requested ordinal, actual ordinal, rejection code, and fallback count. If no
survivor remains, only that candidate recipe is rejected; other families and
slots continue.

The seed hierarchy is independent of simulator row assignment:

```text
seed(job, profile_revision, family, recipe_index, cycle, factor, retry)
```

No call to `env_id` or process-global RNG may alter recipe identity. After
measured validation, coverage may alias provisional geometry families when the
measured geometry is equivalent; that aliasing never changes the original
candidate or artifact ID.

## Affordance, replanning, and trajectory order

The physical episode follows this order for each candidate:

1. reserve a compatible physical slot;
2. restore robot, cube, target state, and preparation epoch for the selected
   reference family;
3. verify the restored state;
4. apply exactly one visual profile to the selected environment row and apply
   any declared global sun state once for the batch;
5. validate the visual/observation binding;
6. for each of the three cycles:
   1. build an `AffordanceSamplingContext` from the current measured state;
   2. request the recipe's branch, record selected pose/provenance, and reject
      misaligned or unsuccessful rows;
   3. replan the Pick using the selected affordance;
   4. export the grounded Place source plan through the source adapter;
   5. select the recipe's spatial trajectory variant on permitted phases;
   6. rebuild a complete Atomic `ActionPlan` on the identical `dt` grid;
   7. validate endpoints, contact/release samples, joint limits, and path;
   8. execute the Pick/Place segment and capture actions, qpos, object pose,
      timestamps, and RGB observations; and
   9. verify the cube pose against the current drop target before advancing;
7. run full-episode measured validation;
8. submit one local artifact record; and
9. release the exact slot reservation after a definitive receipt.

Visual application occurs after restore and before the first command. Visual
state is held constant for the whole episode, so all three cycles share the
same camera/material/light profile. Visual changes do not change physical
geometry or planner inputs unless a sensor contract explicitly says so; the
observation stream records what was actually rendered.

The trajectory transformer may edit only explicitly permitted spatial phases.
Approach/contact/release/hold samples and phase endpoints remain exact. Timing
variation is disabled in this profile. A rebuilt plan must include all command,
velocity, tracking, named-segment, dependency, effect, guard, and recovery
metadata from the source plan.

## Generic source adapters

The same coordinator consumes four source kinds:

1. handwritten templates;
2. MotionGenerator `PlanResult` rows;
3. grounded Atomic `ActionPlan` values; and
4. Task Program calls through the Atomic adapter.

Each adapter implements the source-neutral `SourceAdapter` port:

```python
export_template(
    source: SourceValue,
    *,
    context: SourceContext,
) -> TrajectoryTemplate
```

Adapters own joint order, phase annotations, allowed operators, validator ID,
and source revision. They do not own physical slot state or visual behavior.
The Task Program adapter is an edge package and delegates template conversion
to the Atomic adapter; `motion.expansion` remains free of Task Program and
Atomic imports.

## `m` reference families and `n` generic physical slots

`PhysicalSlotPool` is a host-owned generic port. It exposes reservation,
restore, release, cancellation, and capability inspection without embedding a
family-to-slot mapping.

The default pool has `n = 16` compatible environment rows from
`env.generation.yaml`. Any compatible candidate may run on any free row. A
candidate's identity remains unchanged when it moves between rows or is
restarted after a definitive pre-command failure.

The strict compatibility key contains:

- scene case and physical backend;
- initial-state schema and preparation epoch;
- complete joint order and controlled-joint mapping;
- phase layout and control `dt`;
- robot/solver/calibration revision;
- measured-validator revision; and
- observation tensor schema and renderer capability.

Visual profile ID is not a physical geometry key. Per-environment visual
parameters are allowed within one compatibility bucket. This deployment also
supports one explicitly declared global sun operation. Because the sun is a
single renderer object, a batch selects at most one global sun state; multiple
profiles that declare it must agree, and the selected state applies to all
rows. Unsupported global renderer mutations remain rejected.

Scheduling is family round-robin with FIFO order within each family. It is not
coverage-per-cost. Candidate and artifact queues are bounded; producers block
or receive backpressure before allocating beyond the configured budget. Every
exception and cancellation path releases the exact reservation and removes
the candidate from the appropriate queue.

## Visual randomization and observations

Visual randomization is physically co-located with each rollout but logically
independent of physical candidate enumeration:

- one physical candidate receives one visual profile;
- one profile is used for all three cycles;
- the profile count does not multiply `m × a × t`;
- profile assignment is deterministic and balanced over accepted physical
  candidates; and
- candidate zero uses the canonical profile.

The visual profile port resolves named profiles into seedable operations using
the existing environment functors:

- `randomize_visual_material` for per-environment object appearance;
- `randomize_camera_extrinsics` and `randomize_camera_intrinsics` for RGB
  camera variation; and
- the host light port for the single global sun.

The global sun is reset to its authored state before each rollout. A declared
`global_sun` profile applies one color and intensity to the whole batch, while
per-environment material and camera operations remain row-scoped. Unsupported
global state is rejected with an actionable compatibility error instead of
silently affecting other renderer objects.

The profile's ID, seed, resolved parameters, application status, and renderer
capability are recorded in the candidate schedule digest and artifact
manifest. There is no post-rollout re-render queue and no observation fan-out
that creates additional variants.

Low-resolution RGB is the default for sixteen concurrent slots. A separate
showcase option may capture a high-resolution video for selected candidates,
but it does not alter the recipe count or physical validation result.

The default `rgb_visual.yaml` registers four profiles so the balance is
observable without multiplying physical candidates:

| Profile | Per-environment change | Global state |
| --- | --- | --- |
| `rgb_canonical` | authored cube material and camera calibration | unchanged |
| `rgb_material_01` | deterministic cube base-color/roughness material | unchanged |
| `rgb_camera_01` | deterministic camera extrinsic/intrinsic perturbation within sensor bounds | unchanged |
| `rgb_light_01` | unchanged | global `main_light` sun color and intensity (`<= 10`) |

The profile resolver validates that per-environment operations are seedable and
that the only supported global operation is the bounded `main_light` sun.

## Measured validation and local persistence

`MeasuredValidator` accepts an episode only when all required evidence is
present:

- the slot restore matched the selected reference family within tolerances;
- all commands, qpos, poses, timestamps, and RGB frames are finite and aligned;
- every rebuilt trajectory stayed within joint/path/contact constraints;
- the Task Program completed all three Pick/Place cycles without runtime
  failure; and
- the cube pose after each Place matches the corresponding authored target.

Validation output is an immutable `ValidationResult` with check IDs, measured
values, and a validator revision. “Configured completion” and “measured
acceptance” are separate statuses in the manifest.

The default `EpisodeSink` is local and atomic. A run writes:

```text
<output-dir>/
├── manifest.json
├── schedule.jsonl
├── candidates/<candidate-id>/
│   ├── generation.json
│   ├── physical_episode.json
│   ├── trajectory.pt
│   ├── observations/rgb.npz
│   └── video.mp4                     # optional showcase capture
└── failures/<candidate-id>.json
```

`generation.json` contains source, family, Affordance, trajectory, visual,
seed, fallback, slot, and schedule provenance. `physical_episode.json`
contains measured validation and per-cycle poses. The manifest records profile
and schema revisions, counts, accepted/rejected/write-failed totals, and the
absence of a LeRobot commit. The local sink implements the same generic
`EpisodeSink` interface that a future LeRobot sink can implement; no current
code depends on LeRobot receipt fields.

## Intended showcase command

Once the task-local deployment and host adapter are implemented, the complete
showcase is invoked explicitly with the generation environment and profile:

```bash
python examples/sim/motion/repeated_pick_place_generation_combined.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.generation.yaml \
  --generation-profile embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.combined.yaml \
  --output-dir /tmp/repeated-pick-place-generation \
  --device cuda --headless
```

The command is intentionally profile-driven: changing the task deployment or
profile does not require editing the Task Program. `--device cpu` is valid for
static/smoke validation when the simulator backend supports it. The launcher
prints the resolved `m`, `a`, `t`, `n` values, schedule digest, accepted and
rejected counts, and the manifest path. It must not describe projected
completion as measured success.

## Failure, cancellation, and backpressure semantics

Failures are classified at the narrowest owning boundary:

| Failure | Scope | Required behavior |
| --- | --- | --- |
| Unknown field, count mismatch, illegal global visual request, phase mismatch | Whole job | Reject before simulator construction. |
| No legal initial pose | Whole family/job | Reject the family or fail preflight according to profile strictness; never invent a pose. |
| Affordance proposal/replan failure | Candidate and cycle | Record requested choice and reason, try deterministic fallback, otherwise reject only that candidate. |
| Trajectory variant rejection | Candidate and cycle | Try the next survivor; preserve source plan and release candidate if none survives. |
| Visual application/observation binding failure before first command | Candidate | Release reservation; no physical episode is counted. |
| Restore, command, validator, or runtime failure | Candidate | Stop safely, record measured failure, release the exact slot, and continue if cancellation is not requested. |
| Definitive local write failure | Candidate | Mark `write_failed`; do not count as accepted and release after cleanup. |
| Ambiguous sink exception | Submission | Return `write_pending` with candidate/commit/submission IDs; retain reconciliation state rather than guessing. |
| Cancellation | Job | Stop new assignments, safe-finish running slots, drain bounded writes, release reservations, and write a terminal manifest. |

No fallback may mutate another candidate's schedule or consume another
candidate's slot reservation. A failed candidate's seed and requested choices
remain auditable.

## Acceptance and test strategy

Tests are simulator-free unless they exercise the configured vertical slice.
They cover:

### Configuration and identity

- strict canonical and legacy profile decoding;
- exactly-one `trajectory`/`augmentation` rule;
- configurable `m`, `a`, `t`, and `n` bounds;
- deterministic enumeration of all 64 default recipes;
- candidate zero nominality;
- balanced per-cycle branch/variant schedules; and
- stable identity independent of `env_id`.

### Expansion and source boundaries

- all four source adapters satisfy the same protocol;
- Affordance → replan → trajectory → rebuild ordering;
- same-grid `dt`, endpoint, contact, and phase-permission preservation;
- deterministic fallback and complete provenance; and
- no Task Program/Gym/Atomic import from `motion.expansion`.

### Slot, validation, and persistence lifecycle

- sixteen generic slots accept any compatible family;
- strict compatibility rejection and per-env reservation isolation;
- queue item/byte bounds and backpressure;
- restore, validator, cancellation, exception, and ambiguous-write cleanup;
- local artifact atomicity and manifest counters; and
- visual seed/profile isolation across environment rows.

### Task-local vertical slice

- deployment inspection for `task.ur5.generation.yaml`;
- three full Pick/Place cycles with four cube poses, four Affordance branches,
  and four trajectory variants;
- one visual profile per physical episode with balanced profile IDs;
- deterministic schedule and artifact digests across two runs; and
- no LeRobot write or receipt claim.

The showcase may run with a reduced candidate budget for a smoke test, but the
acceptance run uses `m=4`, `a=4`, `t=4`, `n=16`, and produces the complete
64-recipe schedule. Low-resolution RGB is used for the concurrent run; a
single high-resolution replay is optional and separately reported.

Static checks include profile decoding, package-data inclusion, task
deployment inspection, focused expansion/Atomic/Task Program tests, `black .`,
`git diff --check`, and the project context affected check. Live simulator
execution is reported with its backend/device and is not inferred from CPU
tests.

## Compatibility and rollout

1. `generation.demo.yaml` remains the documented trajectory-only T4 example.
2. `generation.combined.yaml` is opt-in and does not change ordinary task
   execution when no profile is supplied.
3. Existing action-level `augmentation` profiles are decoded through a legacy
   normalization path; they cannot silently opt into episode-level Affordance,
   visual, or multi-slot behavior.
4. The original task/environment files are not edited to add generation policy.
5. The local sink is the only enabled persistence implementation in this
   rollout. LeRobot integration can consume the same manifest and episode
   contracts later.
6. Any unsupported capability fails explicitly during preflight rather than
   being approximated by a different factor or scheduler.

## Completion criteria

The design is implemented when the following are all true:

- the canonical combined profile loads and validates against the resolved
  sixteen-row environment;
- four legal cube initial-pose families are enumerated without changing drop
  targets;
- 64 deterministic episode recipes combine four Affordance and four spatial
  trajectory choices, with per-cycle rotations and auditable fallback;
- any compatible physical slot can run any recipe, with exact reservation
  cleanup;
- one visual profile is applied per physical rollout and balanced without
  multiplying physical candidate count;
- all three Pick/Place cycles complete through the existing Task Program edge
  adapter and measured validator;
- local artifacts contain complete generation, physical, visual, observation,
  and validation provenance;
- the rerun produces identical schedule and identity digests; and
- the implementation makes no LeRobot persistence or physical-success claim
  beyond the evidence in the artifacts.
