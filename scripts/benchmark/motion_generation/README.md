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
  --suite atomic_dual_ur5_pgi_hand_over --device cpu
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_dual_ur5_pgi_coordinated_pickment --device cpu
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_dual_ur5_pgi_coordinated_placement --device cpu
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
  within its position and full-rotation tolerance. `CoordinatedPlacement` compares the placed
  object's full rotation relative to its support.
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
- The supplied pose-batch suite covers only Franka + PGI and cuRobo
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

### Spatial domains

A domain with `perturbations` samples spatial parameters instead of using the
track's `seeds`. Each skill provider declares the parameters it exposes in
`spatial_parameters`; a domain may sample only declared parameters, and every
skill in the track must declare every sampled parameter.
`atomic_franka_pgi_spatial` is the reference suite.

```yaml
    domains:
      - {id: nominal, version: v1, kind: nominal}
      - id: far_sector
        version: v1
        kind: held_out
        perturbations:
          samples: 8          # feasible cases kept per case template
          seed: 2             # draw k uses case seed `seed + k`
          max_attempts: 64    # default 4 x samples
          parameters:
            object.position: {radius_m: [0.55, 0.65], azimuth_deg: [-30, 30]}
```

| Parameter | Layer | Region | Case config key | Skills |
|---|---|---|---|---|
| `object.position` | scene layout | cylinder | `object_position_in_base_m` | pick_up, place, push_object |
| `articulation.position` | scene layout | cylinder | `articulation_position_in_base_m` | open_door |
| `robot.start_offset` | robot start | uniform, one per arm joint | `start_qpos_offset_rad` | push_object, open_door |
| `target.offset` | task goal | uniform, 3 values | `target_object_offset_m` | place, push_object |
| `door.open_angle` | task goal | uniform | `open_angle_rad` | open_door |

- **Frame.** A `cylinder` region is an annular sector in the controlled arm's
  solver root frame, the frame `Robot.compute_fk()` composes kinematics in:
  `radius_m` from its z axis, `azimuth_deg` from +x toward +y, sampled uniformly
  over the sector's area. Without `height_m` the object keeps its configured
  height, so it still rests on its support. A `uniform` region takes `low` and
  `high` in the parameter's own unit.
- **Interaction geometry is fixed.** Approach direction, pre-grasp, lift and
  contact distances are not spatial parameters, so a success-rate change is
  attributable to where the task is.
- **Feasibility.** A draw the task cannot pose raises `InfeasibleCaseError`
  during manifest generation: independent IK cannot reach a frozen pose, no
  reachable grasp remains, or the start leaves the joint-limit margin. The draw
  is replaced, so each template keeps exactly `samples` cases; fewer than
  `samples` feasible draws within `max_attempts` rejects the domain. Tracks
  without `perturbations` still stop on these errors.
- **Manifest.** Each sampled case records `spatial_sample` (domain, draw,
  frame, sampled quantities) and `spatial_feasibility` (draws and infeasible
  draws by reason for its template).
- **Aggregation.** Domain rows add `robustness_*` and `held_out_*` mean, worst
  and retention, so the two kinds are not pooled, plus `spatial_draws` and
  `spatial_infeasible_rate`. `aggregates.json` adds `spatial_breakdown`:
  success per 5 cm band of base-frame radius for every sampled position.

### Provider and evaluator interfaces

`atomic_skill.py` owns `AtomicSkillCaseProvider`; the previous import from
`scenarios.atomic_task` remains available. Existing providers continue to use
`task_result()`. A migrated provider overrides `physical_evaluator()` to return
an implementation of `contracts.PhysicalStageEvaluator`:

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
`CompiledTrajectory` or `projected_context`. `ExecutionObservation` currently
contains the existing measured replay summary, plus the target joint at the end
of the declared articulation segment, the held object's peak signed rotation
in a provider-declared window, and each configured gripper's final joint
positions. Missing optional measurements
mean unavailable evidence; the follow-up evaluation work must extend observation
collection before claiming grasp/release/retention or intermediate-stage success.

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
- **Physical evaluation:** extend `ExecutionObservation` with measured segment
  and contact evidence, implement evaluators and counterexample tests, then opt
  providers into `physical_evaluator()` individually.
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
