# Motion Planning & Atomic Skills Benchmark Design

## Current implementation and design history

This document preserves the original benchmark proposal and its roadmap. The
current execution contract below and the [README](./README.md) describe shipped
behavior; later proposal examples and implementation phases are historical
design targets unless explicitly marked implemented. Collision-aware common
inputs, process isolation, full confidence intervals and recovery benchmarks
remain roadmap work.

The implementation uses one framework with two scenario-owned report families:
`free_space` produces Motion Planning results; `atomic_skill` produces Atomic
Skills results. They share adapters, robot configuration, effective motion
limits, trajectory validation and raw artifacts. They maintain separate
populations and rankings, without a combined score.

The domain command is `embodichain benchmark motion-generation`; `--suite`
selects either family. The separate `embodichain benchmark atomic-action-profile`
command diagnoses individual actions over object and position presets, with
optional replay videos. It has its own reports and is outside this framework's
planner comparisons and leaderboards.

Motion Planning suites `motion_panda_free_space_smoke`,
`motion_panda_free_space_coverage` and `motion_ur5_free_space_coverage` use
version 2 populations with both EEF and joint goals. The matrix owns planner
scaling over batches, waypoints, path shapes and start bins. Atomic smoke retains one
`MoveEndEffector` and one `MoveJoints` integration case; repeated primitive
scans have been removed from the randomized/pose-batch suites while preserving
all interaction skill cases.

Report schema 3 has a geometry leaderboard and a separate solved-timing ranking
within Motion Planning, and a physical-task leaderboard within Atomic Skills.
Raw NMG and fixed-clock interpolation receive geometry scores but no timed
rank. A report has three tables per represented family, so mixed runs have six
tables in one file. `planner_call` latency and `atomic_action_compile` latency
are separate fields and comparison scopes.

Planning and validation share effective robot speed caps intersected with any
suite request, plus an explicit suite acceleration budget. Shipped Motion
Planning suites use robot speed caps, acceleration `10 rad/s²`, and no jerk
budget. These are evaluation policies, not substituted robot specifications.
Adapter-local conflicting budgets and unavailable limits needed for timing
are rejected.

## Original proposal

Build an extensible motion-generation benchmark for EmbodiChain that treats
Neural Motion Generator (NMG) checkpoints as candidates and cuRobo as the
primary baseline. The benchmark should evaluate three distinct questions:

1. How fast and resource-efficient is trajectory generation?
2. Is the generated trajectory accurate, safe, smooth, and executable?
3. Does the trajectory complete an Atomic Action or a multi-action task under
   physics simulation?

The original proposal centered on NMG versus cuRobo. Current Motion Planning
suites also enable IK interpolation and TOPPRA as diagnostic baselines; NMG
and NMG with retiming are optional paired candidates requiring a supplied policy.

## Implemented vertical slice

The first physics-backed slice is available as
`suites/atomic_franka_pgi_smoke.yaml`. It runs Franka + PGI with cuRobo only,
and covers `MoveEndEffector`, `MoveJoints`, antipodal-grasp `PickUp`,
`MoveHeldObject`, `Place`, `Press`, `Slide`, and `Twist`. Skills compile through the same `AtomicActionEngine`. Planner-backed skills
use `MotionPolicy(strategy="motion_gen")`; exact Cartesian skills use
`ik_interp` and expose their effective motion source separately. The selected
planner does not receive credit for a trajectory it did not generate.
The `atomic_franka_pgi_smoke_v4` contact cases use the same articulated
assets and placements as their Atomic Action tutorials. `Press` and `Twist`
use `Microwave/microwave.urdf`; `Slide` uses
`Drawer/model_split_links_with_inertials.urdf`. Their target and sampling
configuration is:

| Skill | Target link | Target joint | `sample_count` | `hand_interp_steps` |
|---|---|---|---:|---:|
| `Press` | `button_link` | `button_joint` | 256 | 12 |
| `Slide` | `large_handle_bar` | `cabinet_to_drawer` | 140 | 12 |
| `Twist` | `knob_link` | `knob_joint` | 256 | 20 |

Cases freeze targets derived from the actual link geometry rather than
synthetic semantic targets.

Physical replay samples the target joint from each action's effect segment
through release, retract, and the final hold. Peak signed displacement remains
an explicit diagnostic, including for a spring-loaded button that rebounds,
and provider evaluators score physical stage evidence. `physical_skill_success`
is independent of trajectory diagnostics; legacy `task_success` additionally
requires valid motion and successful controller execution. Each contact skill
owns whether its physical criterion is a peak, terminal or staged measurement.

The optional `atomic_franka_pgi_pose_batch` suite exercises the same
runner with translation-only pose perturbations. Its `pose_batch_size` is
mapped directly to the simulator batch (`B=8` in the supplied YAML): each
environment row is one independently sampled object or target pose, and the
corresponding Atomic Action goal is planned in one vectorized call. Explicit
`GraspGoal.object_pose` values keep PickUp and the PickUp→MoveHeldObject/
Place composites grounded to the settled per-row cube pose. The coverage track
contains six interaction skills; the two pure-motion primitives remain in smoke. Press, Slide, and Twist perturb the physical
Microwave/Drawer root translation before freezing the live target-link pose;
Slide samples one PGI grasp once in target-local coordinates and screens its
translated copies together. General per-row geometry-sampled antipodal
candidate selection remains a future extension.

The pose-batch suite uses the asset, target, and sampling settings above. The
four-skill `atomic_franka_pgi_randomized` suite uses the same `Press`
settings. All three suites configure `physics.hold_steps=80` and
`physics.hold_sim_steps=2` for the final pose hold. The standalone `Press` and
`Twist` tutorials each use 256 final-pose updates, also with two physics steps
per update.

The shared runner now selects planners, scenarios, robots, Atomic Action case
providers, and object kinds through registries. Cases freeze the full robot
start state, explicit targets/grasp, object configuration, difficulty factors,
and independent sequential-IK evidence before measured planner calls. Reports
keep planning, kinematic motion validity, controller execution, and physical
task success separate, with three tables per represented report family.

Run the slice with:

```bash
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_smoke --device cuda
```

Record every successful measured case (including all three articulated
interactions) with:

```bash
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_smoke --device cuda \
  --record-video --video-case-limit 0
```

`--video-case-limit` is a global run limit, so a value of `1` records at most
one case. Add `--record-failed-video` when failed cases should be retained as
static debug scenes. For live inspection, open the simulator viewer with:

```bash
python -m scripts.benchmark.motion_generation.run_benchmark \
  --suite atomic_franka_pgi_smoke --device cuda --no-headless
```

## Historical motivation

Before the shared framework, the NeuralPlanner benchmark provided useful latency, memory, rollout,
and endpoint-error measurements on fixed Franka waypoint sets in planner-only
mode. It did not measure dynamic feasibility, execution tracking or physical
task completion and had no cuRobo comparison. The current shared framework
addresses these gaps; collision-world qualification remains future work.

NMG will continue to evolve toward obstacle conditioning, multimodal motion,
physics-aware refinement, closed-loop recovery, and cross-embodiment
adaptation. A versioned suite, fixed case manifests, capability-aware tracks,
and stage-specific outcomes are needed so future results remain comparable and
failures remain diagnosable.

## 1. Objectives and scope

The benchmark should support:

- regression testing across NMG checkpoint and model revisions;
- paired comparison between NMG and cuRobo for success, trajectory quality,
  collision safety, latency, throughput, and memory;
- optional IK plus interpolation and TOPPRA diagnostics;
- incremental tracks for obstacle tokens, multimodal generation, Analytic
  Policy Gradient (APG) refinement, closed-loop recovery, and new robots;
- actionable failure attribution for training-data and model iteration.

The benchmark must not collapse every measurement into one opaque composite
score. In particular, `PlanResult.success` must not be treated as equivalent to
physical task success.

## 2. Capability boundaries and historical design implications

### 2.1 Current NeuralPlanner capability boundary

`embodichain/lab/sim/motion/planners/neural_planner.py` currently:

- supports `MoveType.EEF_MOVE` and `MoveType.JOINT_MOVE`;
- loads robot-compatible ONNX checkpoints; the benchmark configures Franka
  and UR5 policies separately, including joint count and periodic features;
- derives the maximum number of waypoints from checkpoint `waypoint_max`;
- updates rollout state through FK, which is a kinematic model loop rather than
  sensor-driven simulation or real-robot recovery;
- reports a fixed nominal `dt`, with velocity and acceleration estimated from
  joint-position finite differences. `PlanResult` requires `dt` whenever it
  carries positions, so the nominal value cannot be withheld; adapters instead
  declare `timing_is_solved`, and a row that does not solve timing reports no
  comparable duration. The default is nominal: cuRobo, TOPPRA and the explicitly retimed NMG
  adapter declare solved timing. Fixed-clock IK interpolation and raw NMG
  remain nominal;
- has no explicit obstacle, collision, multimodal-sampling, or APG-refinement
  input/output interface yet.

Robot velocity limits are resolved from the current Robot API, including mimic
constraints. Historical backends sometimes returned placeholder values; an
explicit suite budget may supply an unavailable value, while usable robot caps
remain upper bounds. Shared resolution feeds both adapters and validation.
Acceleration and jerk have no automatic asset fallback and remain suite-owned.
Do not copy FR3 speed constants into a suite using a different Franka model.

The v1 benchmark must run within these constraints while reserving
capability-gated tracks for future features. An unsupported capability must be
reported as `unsupported`, not silently converted into success or failure.

### 2.2 Historical gaps and implemented replacements

The original script used one Franka start state, fixed offsets, one environment
and best-hit waypoint errors that could accept out-of-order motion. It lacked
cuRobo, fixed manifests, batch coverage, physical replay and provenance.

The current runner and providers own fixed seeded cases, lifecycle timing,
batch matrices, robot/task replay and raw artifacts. Ordered target matching
requires monotonic arrivals; EEF position/orientation must agree at the same
sample, while joint cases use an explicit per-joint error threshold. Reports
have Time & Memory, Success & Other Metrics and Leaderboard tables for each
family. Remaining collision, isolation and statistical extensions are roadmap
items rather than current measurements.

### 2.3 Atomic Action integration

`MotionPolicy.strategy` defaults to `"ik_interp"`. Benchmark invocations that
use the selected backend explicitly set the following policy; exact Cartesian
skills retain their required IK route and record that bypass rather than
masquerading as a planner comparison:

```python
motion_policy = MotionPolicy(strategy="motion_gen")
```

The NMG checkpoint must remain confined to `NeuralPlannerCfg`. Atomic Action
scenarios, grasp sampling, objects, controllers, and task-success rules must
not contain NMG-specific branches. The planner factory should be the only
backend-specific injection point.

## 3. Architecture

```text
Suite YAML + fixed Case Manifest
                |
                v
        Scenario Providers
      /          |           \
 planner-only  trajectory   atomic-skill
      |          |           |
      +------ Planner Factory/Adapter ------+
                         |
                         v
                 Raw Trial Records
                         |
             +-----------+-----------+
             |                       |
       Metric Evaluators        Failure Classifier
             |                       |
             +-----------+-----------+
                         |
                         v
             Aggregates + Leaderboard
                         |
                         v
       one report with 3 tables per scenario family
```

The CLI selects the benchmark domain before selecting a suite:

```bash
embodichain benchmark motion-generation --suite motion_panda_free_space_smoke
embodichain benchmark motion-generation --suite atomic_franka_pgi_smoke
```

Both families use `motion-generation`; `--suite` determines their population
and scenario. Separate domains keep their commands: `rl`,
`robotics-kinematic-solver`, `grasp-pose-generator`, `workspace-analyzer`, and
`rendering`. The separate `atomic-action-profile` runner profiles individual actions
over object and position presets. See the [README](./README.md#run) for the
command matrix.

Reuse established patterns from the current benchmark system:

- dispatch through `scripts/benchmark/__main__.py`;
- Atomic Action `smoke/coverage/full` profiles, case sweeps, physics replay,
  and physical-success rules;
- RL benchmark suite YAML, config/runner/reporting separation, resolved
  protocol, per-run artifacts, and compatibility-aware resume;
- the current NeuralPlanner warmup, CUDA synchronization, memory measurement,
  and optional-baseline flow;
- one Markdown report, three tables per family, and complete separate rankings.

Current framework owners are `config.py`, `runner.py`,
`scenarios/free_space.py`, `metrics/trajectory.py`, `limits.py`, `aggregation.py`
and `reporting.py`. For Atomic Skills, `atomic_skill.py` owns the provider
contract and registry, and `scenarios/atomic_skill.py` orchestrates the lifecycle.
Within `scenarios/`, `atomic_case_utils.py`, `atomic_policy.py`,
`atomic_grasps.py` and `atomic_evaluation.py` own deterministic case helpers,
motion policy and attribution, grasp services, and measured evidence/evaluators.
Base providers live in `atomic_motion_cases.py`, `atomic_manipulation_cases.py`
and `atomic_contact_cases.py`; the additional single-arm and dual-arm providers
reuse these shared owners. `atomic_replay.py` composes `AtomicReplayRuntime`
from explicit simulator, robot, object/articulation, gripper and track bindings,
so measured replay and video use the same execution path. Scene reset and
teardown stay with the scenario. This decomposition preserves behavior, suite
definitions and report schemas; wheel packaging remains future work. See the
[provider ownership guide](./README.md#provider-and-evaluator-interfaces) when
adding a skill.

The original decomposition sketch below is retained as design history, not a
filesystem inventory:

```text
scripts/benchmark/motion_generation/
├── run_benchmark.py        # thin CLI entry point
├── config.py               # suite, planner, and scenario configuration
├── registry.py             # planner/scenario/metric registries
├── runner.py               # case matrix, warmup, trials, and resume
├── artifacts.py            # manifests, JSONL, and environment metadata
├── aggregation.py          # grouping, confidence intervals, leaderboard
├── reporting.py            # three Markdown tables per family
├── planners/
│   ├── base.py
│   ├── neural.py
│   ├── curobo.py
│   ├── ik_interpolate.py
│   └── toppra.py
├── scenarios/
│   ├── reach.py
│   ├── waypoint_path.py
│   ├── obstacle.py
│   ├── perturbation.py
│   └── atomic_action.py
├── metrics/
│   ├── performance.py
│   ├── kinematic.py
│   ├── dynamic.py
│   ├── collision.py
│   ├── execution.py
│   └── task.py
└── suites/
    ├── motion_panda_free_space_smoke.yaml
    ├── motion_panda_free_space_coverage.yaml
    └── full.yaml
```

### 3.1 Extension interfaces

Runner logic should not branch on planner names. Use protocols such as:

```python
class PlannerAdapter(Protocol):
    @property
    def metadata(self) -> PlannerMetadata: ...

    def build(self, context: BenchmarkContext) -> MotionGenerator: ...

    def prepare(self, case: BenchmarkCase) -> PreparationMetrics: ...

    def warmup(self, case: BenchmarkCase) -> None: ...

    def plan(self, case: BenchmarkCase) -> PlanResult: ...


class ScenarioProvider(Protocol):
    @property
    def required_capabilities(self) -> frozenset[str]: ...

    def generate_cases(
        self,
        manifest: SuiteManifest,
        seed: int,
    ) -> Iterable[BenchmarkCase]: ...


class MetricEvaluator(Protocol):
    @property
    def required_artifacts(self) -> frozenset[str]: ...

    def evaluate(self, trial: TrialArtifacts) -> dict[str, float | bool]: ...
```

`PlannerMetadata` should include at least:

- `algorithm_id`, for example `nmg_transformer`, `curobo`,
  `ik_interpolate`, or `toppra`;
- `algorithm_role`: `candidate`, `primary_baseline`, or
  `diagnostic_baseline`;
- model revision, checkpoint path, and SHA256 when applicable;
- capabilities such as `eef_waypoint`, `joint_waypoint`, `obstacle`,
  `sampling`, `refinement`, and `closed_loop`;
- supported robots, maximum waypoint count, and input/output schema version;
- planner parameters, model parameter count, and inference dtype.

Adding an NMG architecture, refiner, or baseline should require only a new
adapter/registry entry and suite configuration, not runner, aggregation, or
reporting changes.

### 3.2 Trial data model

Each trial should have a stable key:

```text
(suite_version, track, scenario_id, case_id, algorithm_id,
 model_revision, seed, repeat, batch_size)
```

`TrialRecord` should separate:

- **identity**: the key above, robot, device, git commit, and config/checkpoint
  hashes;
- **case**: start qpos, target waypoints, obstacle/object state, and
  perturbations;
- **outcomes**: planning, motion, execution, and task success plus failure
  stage;
- **metrics**: performance, memory, trajectory, and task values.

Raw JSONL/JSON artifacts should retain numeric types. Percentage formatting
belongs only in Markdown rendering, not before aggregation.

### 3.3 Primary NMG-versus-cuRobo protocol

Motion Planning enables cuRobo as the primary baseline and TOPPRA / IK
interpolation as diagnostics. Optional `nmg` and `nmg_retimed` candidates must
use the same compatible policy and frozen inputs; their only timing difference
is `retime`. A model path alone does not select disabled algorithms: use
`--algorithms` or set `enabled` in YAML.

Retiming uses shared effective velocity and suite acceleration limits, and the
metrics independently check the output. It does not guarantee jerk bounds;
Motion Planning defaults leave jerk unconfigured. A suite may declare jerk
separately, in which case a retimed output can still fail that check. Raw
nominal-clock output participates only in geometry ranking, while solved-clock
output also participates in timed validity ranking. No shared overall rank
mixes these success meanings.

Two implemented families and a future collision extension are:

1. **Free-space common input**: cuRobo uses an empty collision world. Both
   planners receive identical start qpos and ordered EEF or joint targets,
   according to the case modality. Geometry and timed rankings remain separate.
2. **Collision-aware deployment (planned)**: both planners execute in the same
   simulation scene; cuRobo receives the correct collision world while current
   NMG does not receive obstacle tokens. This track measures deployment
   behavior and the current capability gap, not model quality under equal
   information.
3. **Atomic task**: both planners run through the same `AtomicActionEngine`,
   objects, grasps, controller, and physical success criteria.

cuRobo and the current NMG adapter both support `EEF_MOVE` and `JOINT_MOVE`;
checkpoint-specific waypoint capacity is still checked per case. EEF and joint
goals have separate case identities, explicit validity modes, and independently
checked capability requirements. Supplied joint targets are not evidence for
the EEF-only conditioning comparison.

#### Freeze the cuRobo configuration

Every run must record and hash:

- `max_attempts` and `max_planning_time`;
- `interpolation_dt` and `collision_activation_distance`;
- `use_cuda_graph`, actual fallback state, and `warmup_iterations`;
- robot sphere-fit settings and collision-sphere buffer;
- obstacle representation, collision cache, and `multi_env`;
- static/dynamic obstacle names and world-content hash;
- `preserve_plan_samples`.

World representation and sphere fitting are part of the baseline definition
and must not change silently between checkpoint comparisons.

The primary leaderboard should use a frozen operational configuration, for
example checkpoint-default NMG `max_steps` and fixed cuRobo `max_attempts`.
Also add a latency-budget sweep:

- sweep NMG `max_steps`;
- sweep cuRobo `max_attempts`, with a common `planning_budget_ms`;
- retain success-latency Pareto data.

`CuroboPlannerCfg.max_planning_time` currently validates the budget after the
plan; it is not a preemptive real-time deadline. The outer benchmark must
record actual wall latency and `budget_compliance_rate` whether or not a
planner supports interruption.

#### Lifecycle and timing

cuRobo lazily creates and caches a backend for each
`(control_part, batch_size, multi_env, move_type)`. First use may include
robot sphere fitting, voxel-world generation, CUDA graph capture, and warmup.
NMG has checkpoint loading, actor construction, and
device transfer.

Report the following separately for both:

```text
planner_construct_ms
backend_prepare_ms
cold_plan_ms
warm_plan_ms
```

The planning-latency leaderboard must use only `warm_plan_ms`.
`backend_prepare_ms` represents one-time deployment cost; `cold_plan_ms`
represents the first real case. Every batch size and goal type needs its own
prepare/warmup phase.

cuRobo always plans on CUDA. The primary comparison should therefore use the
same CUDA device and fp32 interface. NMG CPU results may be reported as a
separate characterization, not ranked against cuRobo CUDA latency.

#### Multi-waypoint and sample policy

cuRobo plans multiple waypoint segments sequentially. NMG consumes the full
waypoint sequence in one model invocation. The primary performance metric is
the total cost of one `MotionGenerator.generate()` call for the same high-level
input. Also report `num_segments` and `cost_time_per_segment_ms`, but do not
replace total-latency ranking with per-segment latency.

- Planner-native quality: set cuRobo `preserve_plan_samples=True` to retain
  native collision-checked samples and `dt`.
- Common path metrics: resample derived copies from both planners by the same
  arc-length procedure.
- Atomic Action/common execution: use the same action `sample_count` and let
  `MotionGenerator` perform common resampling, but do not call the
  resampled result a native-timing trajectory.

#### Collision worlds

- Use `CuroboWorldCfg.multi_env=False` when every batch row has the same
  robot-relative obstacle layout.
- Use `multi_env=True` when obstacle poses differ relative to each robot.
- Supply per-environment dynamic poses through `dynamic_obstacle_names` and
  `CuroboPlanOptions.dynamic_obstacle_poses`.
- Dynamic obstacles must use `cuboid` or `mesh`, not the `sphere`
  representation that cannot be updated by the original object name.
- Revalidate collision success with an independent simulator/common
  validator. cuRobo `success=True` is not benchmark ground truth.

## 4. Layered evaluation

### 4.1 L0: generation performance

L0 isolates planner computation and does not execute the trajectory.

Sweep:

- batch size: `1, 8, 64`, with larger batches in the full profile;
- waypoint count: `1, 3, min(5, model_max)`, plus supported maxima;
- start state: nominal, random reachable, near joint limit, near singularity;
- path shape: direct, L-turn, S-curve, orientation-only, and combined
  translation/orientation;
- target-distance and orientation-delta bins;
- primary device/dtype: same CUDA device and fp32 interface for NMG and cuRobo;
- separate NMG CPU/fp16/bf16 characterization;
- cold start and warm steady state.

Timing boundaries:

- measure `planner_construct_ms` and `backend_prepare_ms` separately;
- measure `cold_plan_ms` for the first real input;
- measure `warm_plan_ms` after fixed warmup;
- exclude setup, case generation, reporting, FK metrics, and validation;
- call `torch.cuda.synchronize()` before and after CUDA timing.

Primary metrics:

- latency p50/p95/p99;
- `latency_per_env_ms`;
- `cost_time_per_segment_ms` for explaining multi-waypoint scaling;
- trajectories per second;
- rollout steps and policy steps per second;
- CPU RSS delta, GPU allocation delta, and peak GPU memory;
- real-time factor only when trajectory duration has clear semantics.

For Atomic Action tracks, separate `action_planning_ms`,
`physics_execution_ms`, and `task_end_to_end_ms`.

### 4.2 L1: trajectory quality and executability

Distinguish three evaluation views:

1. **path-only**: resample by arc length and compare geometry;
2. **native-timing**: use each planner's own `dt` and derived duration;
3. **common-execution**: use the same controller, control dt, and simulator.

Do not directly compare NMG's fixed nominal `dt=0.01` against IK interpolation
with no meaningful duration. Report unavailable native timing as `N/A`.
Dynamic fairness should come from common execution or common
time-parameterization.

Use `preserve_plan_samples=True` for cuRobo native-timing evaluation and the
original NMG `PlanResult`. Recompute endpoint, constraint, collision, and
smoothness metrics from output trajectories rather than trusting either
planner's internal success flag.

#### Goal and waypoint metrics

- final translation error in mm;
- final rotation geodesic error in degrees;
- ordered waypoint success rate;
- waypoint translation/rotation mean, p95, and maximum error;
- completed waypoint ratio;
- time or step to final target.

Define ordered arrival as:

```text
t_i = the first sample satisfying t_i > t_(i-1), and
      position_error(t_i) <= pos_threshold, and
      rotation_error(t_i) <= rot_threshold
```

The waypoint sequence succeeds only if every valid waypoint has a matching
`t_i`. Continuous error statistics may use monotonic dynamic programming to
jointly match waypoints and trajectory samples. Position and orientation must
not select unrelated best samples.

#### Kinematic and dynamic metrics

- finite-value rate;
- joint-position-limit violation rate and maximum normalized violation;
- joint velocity, acceleration, and jerk violation rates;
- maximum/mean joint velocity, acceleration, and jerk;
- joint path length;
- Cartesian translation and rotation path length;
- path efficiency relative to a geometric lower bound or same-case reference;
- path curvature and path-only smoothness;
- time-indexed integrated squared acceleration and jerk;
- endpoint settling error and hold stability.

Velocity, acceleration and jerk are recomputed from each planner's positions
and reported timing with compact divided differences, not read from the
planner. A central difference averages neighboring steps and reads a path that
reverses every step as nearly at rest; the compact second difference equals the
true acceleration somewhere in its window, so it neither overstates nor hides
the peak. A derivative the path is too short to resolve is reported as not
applicable rather than zero.

Current schema 3 separates the two validity contracts:

```text
geometry_valid =
    finite_positions
    and ordered_waypoints_reached
    and joint_position_limits_satisfied

timed_motion_valid =
    geometry_valid
    and valid_solved_timing
    and applicable_dynamic_limits_satisfied
```

Nominal/unknown timing makes `timed_motion_valid` unavailable, not successful.
The legacy `motion_valid` field remains available for existing consumers.
Collision freedom requires separately qualified collision evidence; free-space
validity is not a collision-safety certificate.

#### Collision and physics-execution metrics

- environment collision rate;
- self-collision rate;
- minimum clearance;
- undesired-contact count and maximum contact impulse;
- controller joint-tracking RMSE and maximum error;
- executed TCP tracking RMSE;
- execution timeout rate;
- final pose error after simulation execution;
- final pose drift after a fixed stable-hold period.

Enable collision metrics only when the scenario supplies a trustworthy
collision world. In `free-space-common`, cuRobo receives an empty world. In
`collision-deployment`, cuRobo receives the full world while current NMG is an
`obstacle_unaware` candidate. The report must expose this information
asymmetry.

#### Reference-based metrics

ADE/FDE, expert joint distance, and cost ratio are diagnostic, not primary
success criteria. A single reference path can unfairly penalize valid alternate
IK branches or left/right obstacle-avoidance modes.

cuRobo may serve as a strong reference for path cost, duration, and clearance,
but it is not the only ground truth. NMG should pass whenever it satisfies the
same external constraints and task criteria, even with a different valid path.

Future generative NMG tracks should add:

- top-k feasibility/success;
- best-of-k cost;
- valid mode count and trajectory diversity;
- total sampling cost per successful sample.

### 4.3 L2: Atomic Actions and task completion

L2 uses `AtomicActionEngine` to generate a trajectory and then executes or
replays it in physics simulation. Object, contact, and robot state determine
task success.

Backend fairness:

- NMG and primary baseline cuRobo use `strategy="motion_gen"`;
- optional TOPPRA diagnostics use `strategy="motion_gen"`;
- exact Cartesian skill routes use `strategy="ik_interp"` and report their
  effective source; they do not contribute success to the selected backend;
- grasp generator, object preset, start state, target, controller, sample
  interval, seed, and physics parameters are identical;
- do not execute a fabricated trajectory after planning failure;
- restore robot, object, and simulator state before every case.

Contact tasks need explicit collision-world ownership:

- the manipulated Pick/Place target must not be treated as a generic
  non-contact obstacle during required contact phases;
- tables, environmental obstacles, and non-target objects should enter the
  cuRobo world;
- the current EmbodiChain cuRobo adapter does not expose dynamic held-object
  attachment, so `MoveHeldObject` must record
  `held_object_geometry_in_planner=false` and validate object collisions in
  simulation;
- write visible constraints into `constraint_information` for every result.

In `atomic_franka_pgi_smoke_v4`, the Microwave and Drawer are present as
physical articulations in the simulator, but the cuRobo adapter still receives
an empty external world. Consequently, the suite can validate contact-driven
target-joint actuation, but it does not establish collision safety against
non-target articulation links or other environment geometry. This limitation
must remain visible in the run metadata and report interpretation.

Suggested coverage:

| Action or sequence | Primary task-success criteria |
|---|---|
| MoveEndEffector | Planning succeeds, executed TCP reaches and holds target, no disallowed collision |
| PickUp | Approach/lift plan succeeds, `held_object` is created, minimum object lift is reached, no drop |
| MoveHeldObject | Object reaches target pose, grasp remains stable, object drift/tilt stays within threshold |
| Place | Place pose reached, release succeeds, final object pose is correct and stable |
| Pick-Move-Place | Every stage succeeds in sequence; final object pose and release state are correct |
| Press | Button joint reaches the configured peak signed displacement from the press segment through replay hold; final rebound is allowed |
| Slide | Drawer prismatic joint reaches the configured peak signed displacement from the pull segment through replay hold |
| Twist | Knob revolute joint reaches the configured peak signed displacement from the twist segment through replay hold |

Record:

- `planning_success`;
- `motion_valid`;
- `execution_success`;
- `task_success`;
- per-Atomic-Action stage success;
- task completion time;
- replan/retry count;
- task-specific pose, lift, slip, release, and contact metrics.
- articulated-contact target-joint initial/final displacement and peak signed
  displacement.

Sequence success must come from one sequential episode. Do not approximate it
by multiplying independently measured action success rates.

## 5. Scenario tracks and capability gates

Only `free_space` and `atomic_skill` are implemented scenario families. The
additional named tracks below describe extension goals; workspace bins already
exist inside free-space. Availability in the table describes intended backend
capability, not a shipped runnable scenario.

| Track | Current NMG | cuRobo | Purpose |
|---|---:|---:|---|
| `free-space-common` | Supported | Supported | Empty-world EEF and joint targets; separate geometry/timed ranks |
| `workspace-generalization` | Supported | Supported | Workspace, orientation, joint-limit, and singularity bins |
| `collision-deployment` | Executable without obstacle input | Supported | Deployment success and current capability gap |
| `atomic-skill` | Partially supported | Supported | Atomic Action and action-chain physical completion |
| `obstacle-aware-common-input` | Not yet supported | Supported | Future equal-information scene-constraint comparison |
| `multimodal` | Not yet supported | Single-output reference | Top-k coverage, diversity, and sampling cost |
| `physics-refinement` | Not yet supported | Reference | NMG initialization plus APG/trajectory optimization |
| `closed-loop-recovery` | Not yet supported | Not yet supported | Observation, target, and disturbance recovery |
| `cross-embodiment` | Not yet supported | Multi-robot configuration | Robot, DoF, control-rate, and dynamics adaptation |

Check shared `required_capabilities` at the scenario boundary, then per-case
target capability before dispatch. Free-space requires `batched` and
`empty_world` globally; each case requires EEF or joint support independently:

- `supported`: run normally;
- `unsupported`: record the reason and exclude it from that track's
  denominator;
- `error`: the adapter declared support but failed, so count it as failure.

Always report `coverage_rate` to prevent selective execution from improving
rank. Formal track eligibility requires 100% coverage of mandatory cases.
Retain ineligible algorithms in the leaderboard with `eligible=false`.

## 6. Suites and scenario matrix

### 6.1 Profiles

- `smoke`: one robot, a few deterministic cases, `B=1`, one seed; for PRs.
- Motion Planning `coverage`: workspace/path bins, `B=1/8/64`, five Panda
  seeds or three UR5 seeds, EEF and joint goals, three warmups and twenty
  measured repeats. Atomic coverage is configured separately for physical tasks.
- `full` (planned): dense workspace/OOD cases, boundary states, all objects/positions,
  obstacles, perturbations, and at least 20 seeds; for model releases.

The shipped `atomic_franka_pgi_pose_batch` coverage suite is the
translation-only Atomic Skill variant: `randomization.pose_batch_size: K` is
paired with one simulator `batch_sizes: [K]`, and each environment row carries
one deterministic object/target pose perturbation. A case therefore makes one
batched planner request for all `K` rows. It uses fixed cube grasps and moves
the real Microwave/Drawer root for Press, Slide, and Twist. The Slide case
shares one geometry-sampled target-local grasp across the translated rows;
fully independent per-row antipodal candidate selection remains a future
extension.

### 6.2 Fixed case manifests

Generate randomized scenarios into a fixed manifest before giving them to any
planner. Save:

- robot and start qpos;
- ordered target waypoints;
- object/obstacle poses and physical properties;
- target distance, orientation delta, and workspace bin;
- perturbation schedule;
- validity evidence, such as independent reachability validation;
- case schema version.

Unreachable targets may form a separate robustness track but must not enter the
normal success denominator. Case inclusion must not depend on the success of
the cuRobo run being evaluated, which would create baseline selection bias.
Use generation rules, an independent validator, or a frozen offline oracle.

### 6.3 Current Motion Planning configuration example

```yaml
schema_version: 1
name: custom_motion
suite_version: custom_motion_v1
profile: smoke
planners:
  - id: toppra
    adapter: toppra
    role: diagnostic_baseline
    enabled: true
    config: {}
protocol:
  report_schema_version: "3"
  warmup_trials: 1
  measured_trials: 3
  joint_velocity_limit_rad_s: null
  joint_acceleration_limit_rad_s2: 10.0
tracks:
  - id: free-space-common
    scenario: free_space
    config:
      target_types: [eef, joint]
      joint_threshold_rad: 0.02
      batch_sizes: [1]
      waypoint_counts: [1, 3]
      path_shapes: [direct]
      start_state_bins: [nominal]
      seeds: [11]
```

When a launcher supplies environment-derived configuration, save resolved values,
NMG checkpoint hash, cuRobo config hash, and world-content hash in run
metadata.

`free-space-common`, `collision-deployment`, and `atomic-skill` should create
isolated planner instances. Rebuild cuRobo planner/backends when static world
geometry or representation changes. Reuse a backend only for fixed geometry
whose registered dynamic-obstacle poses are updated.

## 7. Success semantics and failure taxonomy

Each case defines `primary_success`:

- L0 planner-only: `planning_success`;
- L1 trajectory: `motion_valid`;
- L2 execution: `execution_success`;
- L2 task: `task_success`.

Planner-internal `pos_eps/rot_eps` only determine when that planner stops.
Cross-algorithm `ordered_waypoints_reached`, `motion_valid`, and
`primary_success` must use suite-owned, versioned external thresholds that are
identical for every algorithm.

Retain every stage:

```text
input_valid
  -> planning_success
  -> ordered_waypoints_reached
  -> motion_valid
  -> execution_success
  -> task_success
```

Use a stable failure taxonomy:

- `invalid_case`
- `unsupported_capability`
- `checkpoint_load_failure`
- `planner_exception`
- `planner_reported_failure`
- `timeout`
- `non_finite_trajectory`
- `waypoint_miss`
- `joint_limit_violation`
- `dynamic_limit_violation`
- `self_collision`
- `environment_collision`
- `controller_tracking_failure`
- `object_not_grasped`
- `object_dropped`
- `release_failure`
- `task_goal_miss`

Store failure stage and reason in raw artifacts. Notes may summarize common
failures without adding tables beyond the three per represented family.

## 8. Fairness and statistical protocol

### 8.1 Baseline fairness

- Use the same case, start qpos, waypoint order, and external tolerance.
- Compare selected candidates and baselines within the same family and
  geometry/solved-timing ranking; preserve diagnostic roles.
- Give cuRobo an empty world in `free-space-common`.
- Record the scene/constraint information visible to each planner in
  `collision-deployment`.
- Time only planner generation; exclude FK/collision metrics.
- Do not force identical sample counts during native generation.
- Apply common path resampling only to derived copies.
- Use the same controller and control dt for execution metrics.
- Report native time parameterization and common execution separately.
- Retain baseline failures; never remove failed cases from aggregation.
- Report success-conditioned continuous metrics and all-case failure-aware
  statistics to avoid survivor bias.
- Prefer isolated subprocesses for NMG and cuRobo. cuRobo CUDA graphs,
  backends, and world caches persist and can contaminate memory or cold-start
  results based on execution order.

### 8.2 Reproducibility and confidence intervals

- Fix Python, NumPy, Torch, and simulator seeds.
- Warm up every `(planner, scenario, batch_size, waypoint_count)` separately.
- Report latency p50/p95/p99.
- Report Wilson or bootstrap 95% confidence intervals for success.
- Report mean, median, p95, and bootstrap 95% confidence intervals for
  continuous metrics.
- Record OS, CPU, GPU, driver, CUDA, Torch, DexSim, and EmbodiChain git commit.
- In addition to PyTorch allocator memory, record process-level GPU memory when
  available to capture cuRobo/CUDA-graph allocations.
- Save the resolved suite and case manifest.

### 8.3 Protocol versioning

- Version suite, case manifest, metric schema, and report schema independently.
- Increment suite version when thresholds, scenario distribution, or primary
  success changes.
- Combine leaderboard results only across identical suite/protocol versions.
- Require matching hardware class, device, dtype, and batch protocol for
  latency leaderboards.
- Historical checkpoints may be rerun on a new suite, but old results must not
  be silently relabeled as the new protocol.

## 9. Artifacts and report contract

Suggested output:

```text
outputs/benchmarks/nmg/<run_id>/
├── resolved_suite.yaml
├── environment.json
├── case_manifest.json
├── trials.jsonl
├── aggregates.json
├── report.md
└── videos/                 # optional representative success/failure cases
```

Each run produces one `report.md`, with three tables for each represented
family. A Motion-only or Atomic-only run has three tables; a mixed run has six.
The raw artifact layout remains shared, and report schema 3 adds
`motion_planning_*` and `atomic_skills_*` aggregate views.

### 9.1 Time & Memory

Recommended columns:

```text
track, scenario, comparison_mode, algorithm, algorithm_role, model_revision,
planner_config_hash, batch_size, waypoint_count, num_segments, num_trials,
planner_construct_ms, backend_prepare_ms, cost_time_ms, cold_plan_ms,
warm_plan_ms_p50, warm_plan_ms_p95, trajectories_per_second,
planning_budget_ms, budget_compliance_rate,
cpu_delta_mb, gpu_delta_mb, peak_gpu_mb
```

`cost_time_ms` is the mean primary steady-state timed operation for the row.
Current `timing_scope` is `planner_call` for Motion Planning or
`atomic_action_compile` for Atomic Skills. Their explicit aliases are
`planning_time_ms` and `compile_time_ms`; execution and end-to-end time are
separate. Compile latency is never ranked against bare planner-call latency.

### 9.2 Success & Other Metrics

Aggregate by `(track, scenario, algorithm, condition_bin)`:

```text
track, scenario, comparison_mode, algorithm, algorithm_role,
constraint_information, cases, coverage_rate, success_rate,
planning_success_rate, execution_success_rate,
task_success_rate, final_pos_err_mm, final_rot_err_deg,
waypoint_completion_rate, joint_violation_rate, dynamic_violation_rate,
collision_rate, min_clearance_m, path_efficiency, jerk_cost,
task_metric, top_failure
```

Use `N/A`, not zero, for inapplicable values.

### 9.3 Leaderboard

Each family has one leaderboard table with `track` and `ranking` columns.
Motion Planning includes independent `geometry` and `timed_motion` ranks;
Atomic Skills uses `physical_skill`. Nominal clocks receive no timed rank.
The following fields remain part of the broader reporting design:

```text
track, rank, algorithm, algorithm_role, model_revision, planner_config_hash,
eligible, coverage_rate, overall_success_rate, motion_valid_rate,
task_success_rate, latency_p95_ms, peak_gpu_mb
```

`overall_success_rate` is the macro average of `primary_success` over all
mandatory cases in the track. Sort within each track by:

1. `eligible=True`;
2. `overall_success_rate` descending;
3. `coverage_rate` descending;
4. `latency_p95_ms` ascending.

Include every evaluated algorithm in the current scope, not only the top
entries. Trajectory imitation error must not override task success in ranking.

## 10. NMG-specific diagnostics

These should not block v1, but the schema should reserve them.

### 10.1 Solution leakage

- Use Cartesian-only conditioning for the primary leaderboard.
- Keep EEF and joint-goal populations explicitly labeled; do not pass reference
  joint solutions as extra conditioning for EEF cases.
- Test the same Cartesian target from different start qpos and legal IK
  branches.
- Report sensitivity to joint-interpolation shortcuts.
- Record actual checkpoint input fields in metadata.

### 10.2 Physics refinement and APG

Create paired cases:

- `nmg_raw`
- `nmg_plus_refinement`

Report:

- hard-feasibility gain;
- task-success gain;
- trajectory-cost reduction;
- added latency and iterations;
- refinement-divergence rate.

### 10.3 Closed-loop recovery

When observation feedback is available, add:

- state-observation noise;
- target-pose motion during execution;
- external force or joint-tracking disturbances;
- object slip;
- obstacle motion.

Report recovery success, time to recover, replan count, maximum post-disturbance
error, and final task success. The current FK rollout must not be reported as
closed-loop recovery.

## 11. Historical implementation plan and remaining roadmap

The phases below record the original incremental plan, not current default
configuration. The current family separation, input matrices, shared limits
and report semantics are described at the top of this document. Obstacle
worlds, wider statistics and recovery remain follow-up work.

### Phase 0: protocol and cuRobo baseline

- Split config, aggregation, and reporting from the current script.
- Preserve the current CLI and checkpoint download behavior.
- Add a cuRobo adapter and make it the default primary baseline.
- Default the CLI/suite to `nmg curobo`; require explicit diagnostic
  IK/TOPPRA baselines.
- Separate cuRobo backend prepare, cold plan, and warm plan timing.
- Produce three report tables per represented family.
- Save resolved config, environment metadata, and trial JSONL.
- Add unit tests for report schema, full leaderboard coverage, and ordered
  waypoint metrics.

Minimum tests:

- ordered waypoint matching rejects out-of-order and split
  position/orientation hits;
- warmup trials do not enter aggregation;
- `unsupported`, `error`, and ordinary failure remain distinct;
- failed trials are not silently removed from continuous aggregation;
- the leaderboard includes every in-scope algorithm and applies
  success/coverage/latency ordering;
- single-family reports contain three Markdown tables and mixed reports six;
- planner-only smoke test with a fake NMG checkpoint;
- graceful skip when the optional cuRobo runtime is unavailable;
- cuRobo empty-world, shared-world, and multi-env dynamic-world configuration;
- Atomic Action integration with `strategy="motion_gen"`.

### Phase 1: core motion

- Add fixed manifests, randomized workspace/path cases, and batch scaling.
- Implement paired NMG-versus-cuRobo `free-space-common`.
- Add frozen operational configuration and latency-budget/Pareto sweeps.
- Add path-only, native-timing, joint, and dynamic metrics.
- Add common resampling and failure classification.
- Keep cuRobo as the primary baseline; add IK/TOPPRA only through adapters when
  requested.

### Phase 2: physics and Atomic Actions

- Add cuRobo collision worlds and `collision-deployment`.
- Parameterize planner construction in Atomic Action benchmarks.
- Explicitly separate `ik_interp` and `motion_gen`.
- Reuse current object/position/approach profiles and physical-success rules.
- Add MoveEndEffector, PickUp, MoveHeldObject, Place, and
  Pick-Move-Place.
- Add controller tracking, collision/contact, and stable-hold metrics.

### Phase 3: future NMG capabilities

- Add equal-information obstacle-aware, multimodal, APG-refinement, and
  closed-loop-recovery tracks.
- Add cross-embodiment and unseen-robot tracks.
- Add release profiles and a long-lived checkpoint leaderboard.

## 12. Current acceptance criteria and future extensions

- The current Franka NMG checkpoint and cuRobo run the same default
  `motion_panda_free_space_smoke` suite and `free-space-common` manifest.
- cuRobo is the primary baseline; TOPPRA and IK interpolation retain their
  diagnostic roles. Geometry and solved-timing ranks are independent.
- cuRobo backend preparation and steady-state planning latency are separate.
- Free-space and Atomic Skills record their actual cuRobo world configuration.
  A future collision-deployment track must qualify its collision world separately.
- NMG enters supported Atomic Action cases through
  `strategy="motion_gen"`.
- Planning, motion, execution, and task success remain distinct.
- New planners and metrics register without runner changes.
- Every baseline replays the same fixed manifest.
- Native timing and common execution are not mixed.
- Every run has raw trial artifacts, reproducibility metadata, and one
  Markdown report.
- The report has three tables per represented family and rankings covering
  every evaluated algorithm, with no cross-family or mixed-clock overall score.
- Unsupported capabilities, real failures, and runtime errors remain distinct.
- PR smoke runs use bounded populations; larger coverage runs and future full
  suites remain separately selected workloads.
