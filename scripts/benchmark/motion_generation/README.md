# Planner Motion Generation & Atomic Skill Benchmark

Shared planner benchmark framework for fixed motion-generation cases and
physics-backed Atomic Actions. All Atomic Actions call the selected planner
through the adapter-owned `MotionGenerator`.

Design background and roadmap: see [`BENCHMARK_DESIGN.md`](./BENCHMARK_DESIGN.md).

## Run

```bash
python -m scripts.benchmark.motion_generation.run_benchmark --suite smoke
python -m scripts.benchmark.motion_generation.run_benchmark --suite coverage
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_curobo --device cuda
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_curobo --device cuda \
  --record-video --video-case-limit 0
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_curobo --device cuda --no-headless
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_curobo_randomized --device cuda
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_curobo_pose_batch --device cuda
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_door_pour_align_push --device cuda
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_spatial_generalization --device cuda
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_dual_ur5_pgi_hand_over --device cpu
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_dual_ur5_pgi_coordinated_pickment --device cpu
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_dual_ur5_pgi_coordinated_placement --device cpu
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_dual_ur5_pgi_hand_over_spatial_generalization --device cpu
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_dual_ur5_pgi_coordinated_pickment_spatial_generalization --device cpu
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_dual_ur5_pgi_coordinated_placement_spatial_generalization --device cpu
python -m scripts.benchmark.motion_generation.run_benchmark \
  --extra-baselines ik_interpolate toppra
```

Artifacts land under `outputs/benchmarks/<suite-name>/<timestamp>/`
(`resolved_suite.yaml`, `case_manifest.json`, `trials.jsonl`, `aggregates.json`,
`report.md` with exactly three tables). Atomic Task videos, when enabled, land
in that run's `videos/` directory.

## Implemented

- Extensible planner, scenario, robot, Atomic Action, and object registries
- `free-space-common` track with fixed manifests and start-state bins
- `atomic-task` track with frozen robot/object/task manifests and common physics replay
- Single-arm Atomic Task slice: Franka + PGI with `MoveEndEffector`,
  `MoveJoints`, `PickUp`, `MoveHeldObject`, `Place`, `Press`, `Slide`, and `Twist`
- `atomic_franka_pgi_curobo_smoke_v3` uses the Microwave articulation for
  `Press`/`Twist` and the Drawer articulation for `Slide`; contact targets are
  the actual `button_cap`, `cap_1`, and `large_handle_bar` links
- Contact-skill replay records peak signed target-joint displacement as a
  diagnostic. `task_success` follows the common motion/execution result and
  does not gate on the object's final articulation state.
- Deterministic 16-seed generalization sweep for the original six-skill subset,
  with bounded robot-start, target, held-object, and object-pose randomization
- Translation-only pose-batch coverage suite (`atomic_franka_pgi_curobo_pose_batch`):
  one simulator environment row per independently perturbed object/target pose
  sample. Each action in a case is planned over all eight rows in one batched
  Atomic Action goal and planner call (one benchmark case). Configure
  `randomization.pose_batch_size` together with a matching single-value
  `batch_sizes: [K]` and the
  `*_translation_jitter_m` bounds in the suite YAML. The shipped track covers
  all eight skills: cube actions use fixed grasps, while the per-skill
  `articulation_position_jitter_m` bounds translate the actual Microwave/Drawer
  roots for `Press`, `Slide`, and `Twist`. `Slide` samples one PGI grasp in
  target-local coordinates and screens the translated batch together.
- `atomic_franka_pgi_door_pour_align_push_v1` adds `OpenDoor`, `Pour`,
  `AxisAlign`, and `PushObject` on the same embodiment, replayed at 16 physics
  steps per waypoint. Each case freezes its grasp and skill poses; success is
  read from the replayed scene: the hinge angle at the end of the opening
  segment, the held object's peak rotation about its own axis while pouring,
  the held object's axis after alignment, and the pushed object's final pose.
- `atomic_franka_pgi_spatial_generalization_v6` is the strict spatial
  generalization baseline for all ten task-level Franka skills: `PickUp`,
  `MoveHeldObject`, `Place`, `Press`, `Slide`, `Twist`, `OpenDoor`, `Pour`,
  `AxisAlign`, and `PushObject`. Tracks group only skills with the same spatial
  contract, so object-only skills are never mislabeled as target-shift tests.
  Its isolated robustness domains cover object translation and orientation,
  plus target translation and orientation for skills with a Cartesian goal.
  Fixed-base articulations additionally vary root height and roll/pitch;
  held-object goals additionally vary target height and roll/pitch. Each domain
  keeps assets, physics, robot, and task semantics fixed and contains eight
  deterministic samples per seed. Every track also contains a shared planar
  task-frame domain and a skill-appropriate composed domain. These domains use
  deterministic per-dimension stratification, including both configured
  interval bounds, so object, target, support, and task-frame changes are
  evaluated together without replacing the isolated-factor populations.
  Domain overrides
  select `B=1` for nominal cases and `B=8` for shifted cases; generated poses
  are frozen in the case manifest and restored before every planner replay.
- Dual-arm slice on the tutorial dual UR5 + PGI robot (`dual_arm` robot
  provider): `HandOver`, `CoordinatedPickment`, and `CoordinatedPlacement`,
  one suite each, reusing each tutorial's robot settings and scene. These
  skills reject cuRobo, so TOPPRA plans them. Each case compiles the skill
  once with a reference TOPPRA generator while the manifest is built and
  freezes every pose the skill requests for each arm; a row is motion-valid
  only when both arms pass their own frozen poses in order. HandOver's sampled
  pickup and receiving grasps are frozen from the same compile.
  `CoordinatedPickment` solves its keyframes with robot IK and never calls the
  planner, so its trajectory is the same for every planner. It samples its
  grasps during compilation with the case's own `grasp_sample_count`, so cases
  in one track cannot change each other's grasps. It succeeds only when both
  grippers end at least `gripper_hold_gap` past open and short of their grasp
  command (a gripper that misses closes to the command) and the object is
  within its position and full-rotation tolerance. `CoordinatedPlacement`
  compares the placed object's full rotation relative to its support. All
  three providers consume the shared pose-batch object/target translation
  ranges. Placement randomizes both objects on independent deterministic
  streams and additionally accepts
  `support_object_position_jitter_m` and `support_object_yaw_jitter_rad`
  overrides; its grasp and target transforms are rebuilt per environment row.
  The corresponding `*_spatial_generalization` suites add three-seed nominal,
  object-XY, object-yaw, and target-relative-position populations, plus
  stratified task-frame and composed-pose populations. Coordinated
  Pickment varies target height, yaw, and roll/pitch. Coordinated Placement
  varies target height/yaw, separately varies the support object's XY/yaw, and
  applies shared task-frame XY/yaw transforms to both objects while preserving
  their relative layout. HandOver's composed population additionally varies
  delivery height; Coordinated Placement combines subject, support, target,
  and task-frame pose changes in one domain.

  | Atomic Skill | Shared workspace domain | Composed domain | Simultaneously varied dimensions |
  | --- | --- | --- | --- |
  | `PickUp`, `Pour`, `AxisAlign` | `task_frame_planar` | `object_planar_pose` | subject XY + subject yaw |
  | `MoveHeldObject` | `task_frame_planar` | `object_target_spatial_pose` | subject XY/yaw + target XYZ/yaw/roll/pitch |
  | `Place` | `task_frame_planar` | `object_target_planar_pose` | subject XY/yaw + target XY/yaw |
  | `PushObject` | `task_frame_planar` | `object_target_position` | subject XY/yaw + target XY |
  | `Press`, `Twist`, `Slide`, `OpenDoor` | `task_frame_planar` | `articulation_spatial_pose` | articulation XYZ/yaw/roll/pitch |
  | `HandOver` | `task_frame_planar` | `object_target_position` | subject XY/yaw + delivery XYZ |
  | `CoordinatedPickment` | `task_frame_planar` | `object_target_spatial_pose` | subject XY/yaw + target XYZ/yaw/roll/pitch |
  | `CoordinatedPlacement` | `task_frame_planar` | `object_support_target_task_frame_pose` | subject XY/yaw + support XY/yaw + target XYZ/yaw + task-frame XY/yaw |

  | Atomic Skill | Subject XY | Subject Z | Subject yaw | Subject roll/pitch | Target XY | Target Z | Target yaw | Target roll/pitch | Support XY/yaw | Task-frame XY/yaw |
  | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | `PickUp`, `Pour`, `AxisAlign` | ✓ |  | ✓ |  |  |  |  |  |  | ✓ |
  | `MoveHeldObject` | ✓ |  | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |
  | `Place` | ✓ |  | ✓ |  | ✓ |  | ✓ |  |  | ✓ |
  | `PushObject` | ✓ |  | ✓ |  | ✓ |  |  |  |  | ✓ |
  | `Press`, `Twist`, `Slide`, `OpenDoor` | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  | ✓ |
  | `HandOver` | ✓ |  | ✓ |  | ✓ | ✓ |  |  |  | ✓ |
  | `CoordinatedPickment` | ✓ |  | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |
  | `CoordinatedPlacement` | ✓ |  | ✓ |  | ✓ | ✓ | ✓ |  | ✓ | ✓ |
- Track configs may declare `grippers` (one entry per gripper) instead of
  `gripper`, `fixtures` (static tables and slabs that are never parked), and
  cases may keep extra objects in the scene through `scene_object_poses`.
- Default matrix: cuRobo (`primary_baseline`); IK / TOPPRA optional diagnostics.
  TOPPRA also serves Atomic Skills and joint-space cases; Atomic Skill calls
  that leave planner options unset receive the suite's TOPPRA limits and
  sample count.
- Direct, batched NMG ONNX adapter (`candidate`, enabled when a model path is supplied)
- Lifecycle timing: construct / prepare / cold / warm
- Distinct planning, motion-valid, execution, task-success, and physical-effect
  diagnostic stages
- Planning latency, execution wall time, end-to-end time, nominal trajectory
  duration, simulated task-completion time, controller tracking RMSE, and
  task-specific object lift/articulation-joint displacement
- One Markdown report: Time & Memory, Success & Other Metrics, Leaderboard
- Optional Atomic Task headless replay videos after measured evaluation
  (`--record-video`, `--record-failed-video`)

For visual inspection, `--record-video --video-case-limit 0` records every
successful measured Atomic Task case. The limit is global for the run, so
`--video-case-limit 1` intentionally emits at most one video, not one video per
skill. Add `--record-failed-video` to capture failed cases as static debug
scenes. Use `--no-headless` instead to open the live simulator viewer.

## Extend

- Planner: register a `PlannerAdapter`, expose its `MotionGenerator`, and
  declare the `atomic_action` capability.
- Robot: register a `RobotProvider` and select it under `robot`; gripper-action
  suites also declare the gripper control part and open/grasp qpos under `gripper`.
- Object: add another `objects` entry for built-in `cube`/`mesh`, or register a
  new object-kind factory.
- Atomic skill: implement and register an `AtomicSkillCaseProvider`; the runner,
  artifact schema, aggregation, and report stay unchanged. Optional provider
  hooks cover skills with a precondition or their own sampling:
  `prefix_invocations()` compiles actions such as the PickUp that a Pour needs,
  `initial_scene()` supplies observations such as an OpenDoor hinge angle,
  `frozen_grasps()` makes a skill that samples its own grasp use the manifest
  grasp, and `object_rotation_segment()` names the window in which the replay
  measures the held object's peak rotation.

## Current limits

- `collision-deployment` and obstacle-aware common-input tracks
- Atomic Task pose-batch execution uses one configured simulator batch per
  track (the supplied pose-batch suite uses `B=8`). General per-row
  geometry-sampled antipodal candidate selection remains `B=1`; the supplied
  articulated `Slide` case instead shares one target-local candidate across
  translation-only rows.
- The supplied single-arm pose-batch suites cover only Franka + PGI
- The v3 suite's Microwave and Drawer are physical simulation articulations,
  but cuRobo still receives an empty external collision world. Contact replay
  validates target-joint actuation; it does not guarantee collision safety
  against non-target appliance links or other environment geometry
- Dual-arm suites run on TOPPRA only; no collision-aware planner currently
  accepts the coordinated skills
- Latency-budget Pareto sweeps, confidence intervals, subprocess isolation


## Atomic Skill contract foundation (#669)

This migration layer provides shared contracts, versioned domain populations,
capability-aware denominators, and recomputable generalization statistics while
preserving existing skill scoring. Official embodiment loading and new physical
success criteria remain follow-up work. The three existing report tables remain
unchanged; domain summaries are rendered as report metadata so the table count
stays stable.

Suites may also declare the official embodiment identity while the legacy robot
provider remains active during migration:

```yaml
embodiment:
  component: ../../embodichain_tasks/configs/components/embodiments/franka_panda.yaml
  overrides: {}
```

The resolved component and overrides are recorded in `case_manifest.json`; the
static resolver also records endpoint bindings, declared capabilities, runtime
services, and a stable component hash. Runtime `RobotCfg` construction and
simulator binding remain the next embodiment integration layer.

### Domain provenance

Add one or more `domains` alongside `scenario` in an existing track. A single
legacy `domain` entry remains supported. This YAML fragment shows the new
fields; retain the track's existing `config`:

```yaml
tracks:
  - id: atomic-nominal
    scenario: atomic_task
    domains:
      - id: nominal
        version: v1
        kind: nominal
      - id: held_out_objects
        version: v1
        kind: held_out
        objects: [unseen_box]
```

The identity requires nonempty `id`, `version`, and a kind of `nominal`,
`robustness`, or `held_out`. Optional `overrides`, `objects`,
`required_capabilities`, perturbation metadata, and frozen-case metadata are
resolved without mutating the source suite. Domain overrides produce separate
effective tracks, while their `track_group` remains stable for aggregation.
Existing suites without this declaration retain `domain: null`; randomized
suites are not silently classified as nominal.

The runner snapshots this identity onto generated cases and carries it into all
associated lifecycle/trial records. `case_manifest.json` now uses
`case_schema_version: 3` and contains the optional `domain`, `track_group`,
embodiment, and required-capability fields on each case. The manifest also
records the resolved embodiment and planner configuration hashes at the run
level. Each track/batch population is written before any planner evaluates that
population; later populations extend the manifest.

### Provider and evaluator interfaces

`atomic_skill.py` owns `AtomicSkillCaseProvider`; the previous import from
`scenarios.atomic_task` remains available. All 15 registered Atomic Skills use
v1 physical evaluators. Terminal skills score settled replay state; staged
skills declare compiled-coordinate replay windows and score the resulting raw
simulator traces. Providers override `physical_evaluator()` to return an
implementation of `contracts.PhysicalStageEvaluator`:

```python
def evaluate(
    self,
    case: BenchmarkCase,
    observation: ExecutionObservation,
    motion_outcomes: tuple[CaseOutcome, ...],
) -> tuple[PhysicalEvaluation, ...]:
    ...
```

The evaluator receives frozen task inputs, measured replay observations, and
motion-validation results. It receives neither the scenario/engine nor a
`CompiledTrajectory` or `projected_context`. A provider instead implements
`replay_observation_windows()` while it still has the compiled segment map. Each
window selects raw TCP pose, active-object pose, named scene-object poses,
articulation-joint, or gripper-joint channels and may include terminal settle.
`ExecutionObservation` carries the sampled traces plus configured gripper
commands; missing optional measurements remain unavailable evidence rather than
successful effects. Push contact uses an explicitly labeled measured object-
motion proxy; it is not reported as contact-sensor evidence.

### Atomic Action runtime evidence

The runtime foundation in
`embodichain/lab/sim/atomic_actions/evidence.py` provides versioned
`PhysicalEvidenceRequest`, `PhysicalEvidenceBatch`, and
`PhysicalEvidenceFrame` values plus a `PhysicalEvidenceProvider` port. The
`ExecutionRunner` can collect one aligned frame after each fresh observation
and pass it to an evidence-aware effect verifier, phase-effect gate, or
held-object guard. Existing verifier callbacks remain compatible.

Atomic Skill primitives continue to declare only an
`EffectVerificationRequirement`; they do not read simulator objects or mutate
physical state. Benchmark stage evaluators and runtime Atomic Action verifiers
can consume the same evidence provider through separate adapters. Task Program,
Gym/RL, and external decision-model adapters remain outside this benchmark
foundation.

Return one `PhysicalEvaluation` per environment, each containing the same ordered
stage IDs. `StageOutcome.status` distinguishes `passed`, `failed`, `not_reached`,
and `not_applicable`. Failed stages require a failure code; measurements retain
finite numeric values with units in their names. After a failed or unreached
stage, subsequent stages cannot be reached. Duplicate/missing environment rows,
duplicate stage IDs, and inconsistent stage order are evaluator errors, recorded
by the existing runner as `metric_evaluation_error`.

Task success requires at least one applicable stage and every applicable stage
to pass. An all-inapplicable or incomplete result cannot establish success.
Physical success also remains gated by motion validity and execution success.
Raw outcomes retain `stages` and `failure_stage`; legacy outcomes have empty
stages. Measured reports now include chained stage rates when evaluators provide
stages, while domain summaries expose coverage, nominal success, mean, worst,
variance, retention, and eligibility.

### Follow-up ownership

- **Embodiment and case integration:** build against `AtomicSkillCaseProvider`,
  reuse official component loading, and keep case generation independent of
  planner outcomes. Do not duplicate the runner lifecycle or artifact writers.
- **Physical evaluation:** add provider-declared replay windows and measured
  contact evidence when a new stage needs them, implement counterexample tests,
  then opt the provider into `physical_evaluator()` individually.
- **Framework and statistics:** add domain generators, physical observation
  collection, and richer confidence intervals on top of the current domain
  identity and aggregation contracts.

Contract tests can run without the simulator engine:

```bash
python -m pytest -q tests/benchmark/motion_generation/test_contracts.py
```

Runner/evaluator integration and existing benchmark regressions use the normal
EmbodiChain environment:

```bash
python -m pytest -q tests/benchmark/motion_generation
```
