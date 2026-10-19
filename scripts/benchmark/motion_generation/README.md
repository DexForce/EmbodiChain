# Motion Planning & Atomic Skills Benchmark

One benchmark framework provides two independently evaluated families: Motion
Planning (`scenario: free_space`) and Atomic Skills (`scenario: atomic_skill`).
They share robot configuration, planner adapters, resolved motion limits,
trajectory validation, and raw artifacts. Atomic Skills additionally compile
skills, replay them in physics, and evaluate measured physical stages.

Planner-backed skills use the adapter-owned `MotionGenerator`. Exact Cartesian
skills may instead use `ik_interp`, and coordinated skills may own their IK
path. Records identify the effective motion source; these results do not count
as evidence that the selected planner backend produced the trajectory.

Design background and roadmap: see [`BENCHMARK_DESIGN.md`](./BENCHMARK_DESIGN.md).

## Run

`motion-generation` selects this benchmark; `--suite` selects its Motion
Planning or Atomic Skills workload. Other benchmark commands select separate
domains:

| Command | Workload |
| --- | --- |
| `embodichain benchmark motion-generation --suite motion_panda_free_space_smoke` | Motion Planning |
| `embodichain benchmark motion-generation --suite atomic_franka_pgi_smoke` | Atomic Skills |
| `embodichain benchmark atomic-action --smoke` | Individual action profiling over object and position presets |
| `embodichain benchmark rl` | RL training and evaluation |
| `embodichain benchmark robotics-kinematic-solver` | FK/IK accuracy and speed |
| `embodichain benchmark grasp-pose-generator` | Grasp sampling and pose selection |
| `embodichain benchmark workspace-analyzer` | Workspace analysis |
| `embodichain benchmark rendering` | Offscreen rendering and denoising |

Run from a source checkout with the domain command above or its Python module:

```bash
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite motion_panda_free_space_smoke
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite motion_panda_free_space_coverage
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_smoke --device cuda
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_smoke --device cuda \
  --record-video --video-case-limit 0
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_smoke --device cuda --no-headless
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_randomized --device cuda
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_pose_batch --device cuda
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
  --suite motion_panda_free_space_smoke --algorithms curobo toppra ik_interpolate nmg nmg_retimed \
  --nmg-onnx-path /path/to/policy.onnx
```

Artifacts land under `outputs/benchmarks/<suite-name>/<timestamp>/`
(`resolved_suite.yaml`, `case_manifest.json`, `trials.jsonl`, `aggregates.json`,
`report.md`). The report contains three tables per represented family: Time &
Memory, Success & Other Metrics, and Leaderboard. A single-family run has three
tables; a mixed Motion Planning / Atomic Skills run has six. Atomic Skill videos,
when enabled, land in that run's `videos/` directory. `aggregates.json` and `report.md` record the
independent `report_schema_version` used to interpret their fields.

## Evaluation contract

Family selection comes from `TrackCfg.scenario`; there is no second family
configuration to keep in sync.
Each family ranks its tracks independently, with no combined overall score.

| Family | Population | Timed operation | Success views |
| --- | --- | --- | --- |
| Motion Planning | Joint/EEF targets, waypoints, batches, path shapes and start bins | `planner_call`, exposed as `planning_time_ms` | `geometry_valid_rate`; separate `timed_motion_valid_rate` and ranking |
| Atomic Skills | Representative motion integration cases plus interaction and generalization cases | `atomic_action_compile`, exposed as `compile_time_ms` | Physical stages, `physical_skill_success_rate`, execution and end-to-end success |

Geometry validity checks finite joint positions, ordered targets, and joint
position limits. Solved-time validity additionally requires valid timing and
applicable dynamic limits. Raw NMG and fixed-clock IK interpolation can rank for
geometry, but their nominal clocks have no timed-success score or timed rank.
NMG with retiming, TOPPRA and cuRobo participate in the solved-time view when the
recorded output has a solved clock. Dynamic measurements from nominal clocks
remain diagnostics. Continuous trajectory quality uses geometry-valid outcomes
consistently across methods; missing expected outcomes reduce success/coverage.

Planning and validation resolve one motion budget: effective robot velocity
caps, intersected with any explicit suite velocity request; acceleration is a
suite-owned budget. An explicit budget can replace an unavailable robot limit,
but cannot enlarge a usable robot limit. Shipped Motion Planning suites use
robot velocity caps, `10 rad/s²` acceleration, and no jerk limit. The acceleration
value is an evaluation policy, not a hardware specification. Timing adapters
reject unavailable required limits or conflicting adapter-local limits. Missing
runtime caps fall back to positive limits in the robot's assembled URDF before
considering a suite request.

Scenario-owned IK uses scalar `MotionPolicy` limits, so a per-joint suite budget
is conservatively reduced to its minimum and recorded in `policy_motion_limits`.
Planner backends receive the full per-joint vector. `CoordinatedPickment` uses
its own IK projection with a fixed sample clock and remains nominal in timing
reports.

The three Motion Planning suites (`motion_panda_free_space_smoke`,
`motion_panda_free_space_coverage`, `motion_ur5_free_space_coverage`) use
version 2 populations with `target_types: [eef, joint]` and
`joint_threshold_rad: 0.02`. EEF and joint cases share reachable reference inputs
but have distinct identities and goal validation. Configurations that omit
`target_types` retain the legacy EEF-only population and case identities.
`reference_qpos` is diagnostic for EEF cases and the actual ordered target for
joint cases; adapters must not use it to substitute a supplied EEF goal.

All three suites enable cuRobo, TOPPRA and IK interpolation. Raw/retimed NMG rows
are configured as an optional pair: select them with `--algorithms` and supply
`--nmg-onnx-path` (or set `enabled` and the model path in YAML). Smoke uses one
warmup and three measured repeats; Panda coverage and UR5 coverage use three
warmups and twenty repeats. UR5 now covers batches 1, 8 and 64.

Atomic smoke retains one `MoveEndEffector` and one `MoveJoints` case to validate
skill compilation, parameter forwarding and replay. Repeated primitive scans
are removed from the Franka randomized/pose-batch and UR5 randomized suites;
all their interaction skill populations remain. Full planning scans belong to
Motion Planning. Existing specialized spatial and dual-arm suites retain their
physical-task role.

Report schema 3 adds family-specific views in `aggregates.json` while retaining
raw trial and legacy aggregate fields. Geometry and timed leaderboard rows are
labeled by `ranking`; Atomic latency is compile latency, with execution and
end-to-end time reported separately. New runs require report schema `3`; custom
suites explicitly selecting an older version must migrate before running. Existing
artifacts keep their original version and must not be relabeled.

## Implemented

- Extensible planner, scenario, robot, Atomic Action, and object registries
- `free-space-common` track with fixed manifests and start-state bins
- `atomic-skill` track with frozen robot/object/task manifests and common physics replay
- Single-arm Atomic Skill slice: Franka + PGI with `MoveEndEffector`,
  `MoveJoints`, `PickUp`, `MoveHeldObject`, `Place`, `Press`, `Slide`, and `Twist`
- `atomic_franka_pgi_smoke_v4` uses the Microwave articulation for
  `Press`/`Twist` and the Drawer articulation for `Slide`; contact targets are
  the actual `button_link`, `knob_link`, and `large_handle_bar` links
- Contact-skill replay records target-joint traces and peak signed displacement.
  Physical evaluators score the declared stage evidence; legacy `task_success`
  combines physical success with valid motion and successful execution.
- Deterministic 16-seed generalization sweep for four interaction skills,
  with bounded robot-start, target, held-object, and object-pose randomization
- Translation-only pose-batch coverage suite (`atomic_franka_pgi_pose_batch`):
  one simulator environment row per independently perturbed object/target pose
  sample. Each action in a case is planned over all eight rows in one batched
  Atomic Action goal and planner call (one benchmark case). Configure
  `randomization.pose_batch_size` together with a matching single-value
  `batch_sizes: [K]` and the
  `*_translation_jitter_m` bounds in the suite YAML. The shipped track covers
  six interaction skills: cube actions use fixed grasps, while the per-skill
  `articulation_position_jitter_m` bounds translate the actual Microwave/Drawer
  roots for `Press`, `Slide`, and `Twist`. `Slide` samples one PGI grasp in
  target-local coordinates and screens the translated batch together.
- `atomic_franka_pgi_door_pour_align_push_v1` adds `OpenDoor`, `Pour`,
  `AxisAlign`, and `PushObject` on the same embodiment, replayed at 16 physics
  steps per waypoint. Each case freezes its grasp and skill poses; success is
  read from the replayed scene: the hinge angle at the end of the opening
  segment, the held object's peak rotation about its own axis while pouring,
  the held object's axis after alignment, and the pushed object's final pose.
- `atomic_franka_pgi_spatial_generalization_v8` is the strict spatial
  generalization baseline for all ten task-level Franka skills: `PickUp`,
  `MoveHeldObject`, `Place`, `Press`, `Slide`, `Twist`, `OpenDoor`, `Pour`,
  `AxisAlign`, and `PushObject`. Tracks group only skills with the same spatial
  contract, so object-only skills are never mislabeled as target-shift tests.
  Its isolated robustness domains cover object translation and orientation,
  plus target translation and orientation for skills with a Cartesian goal.
  Fixed-base articulations additionally vary root height and roll/pitch;
  held-object goals additionally vary target height and roll/pitch. Each pose
  domain keeps primitive parameters, physics, robot, and task semantics fixed
  and contains eight deterministic samples per seed. Every track also contains
  a shared planar task-frame domain and a skill-appropriate composed domain.
  These domains use deterministic per-dimension stratification, including both
  configured interval bounds, so object, target, support, and task-frame changes
  are evaluated together without replacing the isolated-factor populations.
  Workspace-shell domains sample normalized XY offset radii from the nominal
  task pose toward the configured boundary and record each row as `near`,
  `mid`, or `far` in the case manifest. The generalization summary reports
  bucket-level success rates, sample counts, and failure-stage counts. The six
  cube-based skills additionally run a first-stage parameterized primitive
  sweep. The built-in `cube` primitive varies only its `size` parameter: 40 mm
  and 60 mm uniform scales plus approximately volume-preserving cuboids
  elongated along X, Y, or Z. One YAML anchor owns each parameter value, and
  both the runtime object override and domain provenance reference that anchor.
  Four primitive-only tracks give these domains their own 50 mm nominal
  population and generalization summary, separate from pose shifts. They retain
  the same logical object id, physical parameters, skill configuration, seeds,
  and nominal scene pose. The resolved parameters are recorded in
  `resolved_suite.yaml` and each case's `object_config`. Domain overrides select
  `B=1` for nominal and primitive cases and `B=8` for pose-shift cases;
  generated poses are frozen in the case manifest and restored before every
  planner replay. Additional registered primitives and mesh-backed geometries
  can extend the same replacement structure in later stages.
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

  | Atomic Skill | Cube primitive parameters | Workspace/relative shell | Shared workspace domain | Composed domain | Bilateral mirror | Simultaneously varied dimensions |
  | --- | --- | --- | --- | --- | :---: | --- |
  | `PickUp`, `Pour`, `AxisAlign` | 40/60 mm + X/Y/Z aspect | `object_workspace_shell` | `task_frame_planar` | `object_planar_pose` |  | subject XY + subject yaw |
  | `MoveHeldObject` | 40/60 mm + X/Y/Z aspect | `target_relative_shell` | `task_frame_planar` | `object_target_spatial_pose` |  | subject XY/yaw + target XYZ/yaw/roll/pitch |
  | `Place` | 40/60 mm + X/Y/Z aspect | `target_relative_shell` | `task_frame_planar` | `object_target_planar_pose` |  | subject XY/yaw + target XY/yaw |
  | `PushObject` | 40/60 mm + X/Y/Z aspect | `target_relative_shell` | `task_frame_planar` | `object_target_position` |  | subject XY/yaw + target XY |
  | `Press`, `Twist`, `Slide`, `OpenDoor` |  | `articulation_workspace_shell` | `task_frame_planar` | `articulation_spatial_pose` |  | articulation XYZ/yaw/roll/pitch |
  | `HandOver` |  | `target_relative_shell` | `task_frame_planar` | `object_target_position` | ✓ | subject XY/yaw + delivery XYZ |
  | `CoordinatedPickment` |  | `target_relative_shell` | `task_frame_planar` | `object_target_spatial_pose` | ✓ | subject XY/yaw + target XYZ/yaw/roll/pitch |
  | `CoordinatedPlacement` |  | `support_relative_shell` | `task_frame_planar` | `object_support_target_task_frame_pose` | ✓ | subject XY/yaw + support XY/yaw + target XYZ/yaw + task-frame XY/yaw |

  | Primitive domain | Kind | Parameter override | Value in metres | Geometry change |
  | --- | --- | --- | --- | --- |
  | `cube_uniform_40mm` | `cube` | `size` | `[0.04, 0.04, 0.04]` | uniform scale down |
  | `cube_uniform_60mm` | `cube` | `size` | `[0.06, 0.06, 0.06]` | uniform scale up |
  | `cube_aspect_x` | `cube` | `size` | `[0.07, 0.042258, 0.042258]` | X elongated |
  | `cube_aspect_y` | `cube` | `size` | `[0.042258, 0.07, 0.042258]` | Y elongated |
  | `cube_aspect_z` | `cube` | `size` | `[0.042258, 0.042258, 0.07]` | Z elongated |

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
- Motion Planning defaults: cuRobo (`primary_baseline`), IK interpolation and
  TOPPRA (`diagnostic_baseline`), with optional paired NMG candidates. Atomic
  suites retain their own planner selections. Adapter-backed Atomic motion
  consumes the shared resolved suite limits and configured sample count.
- Direct, batched NMG ONNX adapter (`candidate`, requiring an explicit model
  path and algorithm selection)
- Lifecycle timing: construct / prepare / cold / warm
- Distinct planning, motion-valid, execution, task-success, and physical-effect
  diagnostic stages
- Planning latency, execution wall time, end-to-end time, nominal trajectory
  duration, simulated task-completion time, controller tracking RMSE, and
  task-specific object lift/articulation-joint displacement
- One Markdown report with Time & Memory, Success & Other Metrics, and
  Leaderboard tables for each represented family
- Optional Atomic Skill headless replay videos after measured evaluation
  (`--record-video`, `--record-failed-video`)

For visual inspection, `--record-video --video-case-limit 0` records every
successful measured Atomic Skill case. The limit is global for the run, so
`--video-case-limit 1` intentionally emits at most one video, not one video per
skill. Add `--record-failed-video` to capture failed cases as static debug
scenes. Use `--no-headless` instead to open the live simulator viewer.

## Extend

- Planner: register a `PlannerAdapter` and declare its target capabilities.
  To support Atomic Skills, also expose its `MotionGenerator` and declare
  `atomic_action`. Free-space checks EEF/joint support per case.
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
- Atomic Skill pose-batch execution uses one configured simulator batch per
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
supporting provider-owned physical evaluation. Domain summaries remain report
metadata; each represented family keeps its own three report tables.

Suites can select an official embodiment instead of the legacy `robot` provider:

```yaml
embodiment:
  component: embodichain_tasks/configs/components/embodiments/ur5_dh_pgi_140_80.yaml
  overrides: {}
```

The runner builds the component's `simulation` robot using the shared robot
decoder, preserves its TCP, and supplies its selected motion endpoint and
gripper command preset to the existing scenario/planner pipeline. Do not repeat
inline `robot` configuration or conflicting track-level gripper commands.
Multiple motion resources require an explicit `embodiment.resource_id`; this
selects one resource, not a coordinated dual-arm benchmark. Relative component
paths resolve from the suite file; packaged component references work from any
working directory. Independent camera acquisition and Task Program runtime
services are outside this direct-simulation host.

`atomic_ur5_embodiment` demonstrates PickUp with nominal and object-XY domains:

```bash
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_ur5_embodiment --device cuda
```

The component identity, bindings, declared capabilities, services, and stable
hash are recorded in `case_manifest.json`. The suite retains the existing
evaluation and three-table report. This example requires live GPU physics and
cuRobo; configuration tests do not establish physical task success.

### Domain provenance

Add one or more `domains` alongside `scenario` in an existing track. A single
legacy `domain` entry remains supported. This YAML fragment shows the new
fields; retain the track's existing `config`:

```yaml
tracks:
  - id: atomic-nominal
    scenario: atomic_skill
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

`atomic_skill.py` owns `AtomicSkillCaseProvider`; `scenarios/atomic_skill.py`
owns `AtomicSkillScenario`. All 15 registered Atomic Skills use
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
`physical_skill_success` remains an independent measured result. The legacy
`task_success` / `end_to_end_success` additionally requires valid motion and
successful execution.
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
