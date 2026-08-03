# Neural Motion Generator Benchmark Design

## Proposal

Build an extensible motion-generation benchmark for EmbodiChain that treats
Neural Motion Generator (NMG) checkpoints as candidates and cuRobo as the
primary baseline. The benchmark should evaluate three distinct questions:

1. How fast and resource-efficient is trajectory generation?
2. Is the generated trajectory accurate, safe, smooth, and executable?
3. Does the trajectory complete an Atomic Action or a multi-action task under
   physics simulation?

The default comparison should be NMG versus cuRobo. IK plus interpolation and
TOPPRA should remain optional diagnostic baselines rather than define the main
leaderboard.

## Motivation

The existing NeuralPlanner benchmark provides useful latency, memory, rollout,
and endpoint-error measurements, but it only exercises fixed Franka waypoint
sets in planner-only mode. It does not measure collision safety, dynamic
feasibility, execution tracking, or physical task completion. It also does not
currently include cuRobo even though both `BasePlanner` and `MotionGenerator`
support the cuRobo backend.

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

## 2. Current implementation and design implications

### 2.1 Current NeuralPlanner capability boundary

`embodichain/lab/sim/planners/neural_planner.py` currently:

- directly supports only `MoveType.EEF_MOVE`;
- uses a 7-DoF waypoint Transformer checkpoint, currently centered on Franka;
- derives the maximum number of waypoints from checkpoint `waypoint_max`;
- updates rollout state through FK, which is a kinematic model loop rather than
  sensor-driven simulation or real-robot recovery;
- reports a fixed nominal `dt`, with velocity and acceleration estimated from
  joint-position finite differences;
- has no explicit obstacle, collision, multimodal-sampling, or APG-refinement
  input/output interface yet.

The v1 benchmark must run within these constraints while reserving
capability-gated tracks for future features. An unsupported capability must be
reported as `unsupported`, not silently converted into success or failure.

### 2.2 Gaps in the current NeuralPlanner benchmark

`scripts/benchmark/motion_generation/run_benchmark.py` already handles:

- warmup trials separately from measured trials;
- CUDA synchronization;
- CPU RSS, GPU allocation delta, and peak GPU memory;
- planning latency, final TCP error, and waypoint best-hit error;
- optional IK-interpolation and TOPPRA baselines.

However, it currently uses one Franka start state, fixed waypoint offsets, and
one environment. Repeating a deterministic case mostly measures runtime
variance, not workspace coverage or generalization. It also lacks:

- joint-limit, velocity, acceleration, jerk, collision, and clearance checks;
- path length, path efficiency, and smoothness;
- physics execution and controller tracking;
- Atomic Action task completion;
- batch-scaling measurements;
- resolved checkpoint, case-manifest, seed, software, and hardware metadata.

The planner and `MotionGenerator` already register `CuroboPlanner`, but the
current NMG benchmark does not include it. The new default matrix should be
`NMG vs cuRobo`; IK interpolation and TOPPRA should not be the primary
reference.

The current `compute_waypoint_errors()` independently searches the whole
trajectory for the best sample for each waypoint. This can reward out-of-order
motion and can select different samples for the best position and orientation.
The new benchmark must use ordered waypoint matching.

The current report also creates multiple Quality/Performance tables grouped by
waypoint count plus two leaderboards. EmbodiChain benchmark convention requires
one Markdown report with exactly three tables:

1. `Time & Memory`
2. `Success & Other Metrics`
3. `Leaderboard`

### 2.3 Atomic Action integration

`ActionCfg.motion_source` defaults to `"ik_interp"`. Existing Atomic Action
benchmarks construct a TOPPRA `MotionGenerator`, but cases that do not
explicitly change `motion_source` still use local IK plus interpolation.

NMG and cuRobo Atomic Action evaluation must explicitly set:

```python
cfg.motion_source = "motion_gen"
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
 planner-only  trajectory   atomic-task
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
       one Markdown report with exactly 3 tables
```

Keep the existing CLI entry point:

```bash
embodichain benchmark motion-generation
```

Reuse established patterns from the current benchmark system:

- dispatch through `scripts/benchmark/__main__.py`;
- Atomic Action `smoke/coverage/full` profiles, case sweeps, physics replay,
  and physical-success rules;
- RL benchmark suite YAML, config/runner/reporting separation, resolved
  protocol, per-run artifacts, and compatibility-aware resume;
- the current NeuralPlanner warmup, CUDA synchronization, memory measurement,
  and optional-baseline flow;
- one Markdown report, exactly three tables, and a complete leaderboard.

Refactor the current monolithic script incrementally into:

```text
scripts/benchmark/motion_generation/
├── run_benchmark.py        # thin CLI and compatibility entry point
├── config.py               # suite, planner, and scenario configuration
├── registry.py             # planner/scenario/metric registries
├── runner.py               # case matrix, warmup, trials, and resume
├── artifacts.py            # manifests, JSONL, and environment metadata
├── aggregation.py          # grouping, confidence intervals, leaderboard
├── reporting.py            # exactly three Markdown tables
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
    ├── smoke.yaml
    ├── coverage.yaml
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

The default suite should require only:

- `nmg:<checkpoint_revision>` as the candidate;
- `curobo:<config_hash>` as the primary baseline.

IK interpolation and TOPPRA should be enabled only through
`--extra-baselines` or suite configuration. If enabled, they must still appear
in reports and the leaderboard with role `diagnostic_baseline`.

Use three paired tracks:

1. **Free-space common input**: cuRobo uses an empty collision world. Both
   planners receive identical start qpos and ordered EEF waypoints. This is the
   primary quality and performance leaderboard.
2. **Collision-aware deployment**: both planners execute in the same
   simulation scene; cuRobo receives the correct collision world while current
   NMG does not receive obstacle tokens. This track measures deployment
   behavior and the current capability gap, not model quality under equal
   information.
3. **Atomic task**: both planners run through the same `AtomicActionEngine`,
   objects, grasps, controller, and physical success criteria.

cuRobo supports both `EEF_MOVE` and `JOINT_MOVE`, while current NMG supports
only `EEF_MOVE`. The primary leaderboard must use their common `EEF_MOVE`
capability. Joint-space cases belong in cuRobo-only or diagnostic tracks.

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
robot/world YAML generation, sphere fitting, collision-cache allocation, CUDA
graph capture, and warmup. NMG has checkpoint loading, actor construction, and
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
- Atomic Action/common execution: use the same action `sample_interval` and let
  `TrajectoryBuilder` perform common resampling, but do not call the resampled
  result a native-timing trajectory.

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
2. **native-timing**: use each planner's own `dt/duration`;
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

Define `motion_valid` independently:

```text
motion_valid =
    finite
    and ordered_waypoints_reached
    and joint_limits_satisfied
    and dynamic_limits_satisfied_when_applicable
    and collision_free_when_applicable
```

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

- NMG and primary baseline cuRobo use `motion_source="motion_gen"`;
- optional TOPPRA diagnostics use `motion_source="motion_gen"`;
- optional IK-interpolation diagnostics use `motion_source="ik_interp"`;
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

Suggested coverage:

| Action or sequence | Primary task-success criteria |
|---|---|
| MoveEndEffector | Planning succeeds, executed TCP reaches and holds target, no disallowed collision |
| PickUp | Approach/lift plan succeeds, `held_object` is created, minimum object lift is reached, no drop |
| MoveHeldObject | Object reaches target pose, grasp remains stable, object drift/tilt stays within threshold |
| Place | Place pose reached, release succeeds, final object pose is correct and stable |
| Press | Press depth and valid contact/force reached, retract succeeds, no abnormal object motion |
| Pick-Move-Place | Every stage succeeds in sequence; final object pose and release state are correct |

Record:

- `planning_success`;
- `motion_valid`;
- `execution_success`;
- `task_success`;
- per-Atomic-Action stage success;
- task completion time;
- replan/retry count;
- task-specific pose, lift, slip, release, and contact metrics.

Sequence success must come from one sequential episode. Do not approximate it
by multiplying independently measured action success rates.

## 5. Scenario tracks and capability gates

| Track | Current NMG | cuRobo | Purpose |
|---|---:|---:|---|
| `free-space-common` | Supported | Supported | Empty-world, identical EEF-waypoint primary comparison |
| `workspace-generalization` | Supported | Supported | Workspace, orientation, joint-limit, and singularity bins |
| `collision-deployment` | Executable without obstacle input | Supported | Deployment success and current capability gap |
| `atomic-task` | Partially supported | Supported | Atomic Action and action-chain physical completion |
| `obstacle-aware-common-input` | Not yet supported | Supported | Future equal-information scene-constraint comparison |
| `multimodal` | Not yet supported | Single-output reference | Top-k coverage, diversity, and sampling cost |
| `physics-refinement` | Not yet supported | Reference | NMG initialization plus APG/trajectory optimization |
| `closed-loop-recovery` | Not yet supported | Not yet supported | Observation, target, and disturbance recovery |
| `cross-embodiment` | Not yet supported | Multi-robot configuration | Robot, DoF, control-rate, and dynamics adaptation |

Before case generation, check `required_capabilities`:

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
- `coverage`: workspace/path bins, `B=1/8/64`, at least five seeds, and
  representative Atomic Action object/position cases; for nightly runs.
- `full`: dense workspace/OOD cases, boundary states, all objects/positions,
  obstacles, perturbations, and at least 20 seeds; for model releases.

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

### 6.3 Example suite

```yaml
schema_version: 1
suite: nmg_coverage_v1
profile: coverage
seeds: [11, 23, 37, 53, 71]

planners:
  - id: nmg_transformer
    adapter: neural
    role: candidate
    checkpoint: ${NMG_CHECKPOINT}
  - id: curobo
    adapter: curobo
    role: primary_baseline
    config:
      max_attempts: 5
      interpolation_dt: 0.025
      use_cuda_graph: true
      warmup_iterations: 1
      preserve_plan_samples: true
      world:
        obstacle_representation: mesh
        multi_env: false

tracks:
  free-space-common:
    required_capabilities: [eef_waypoint]
    scenarios: [nominal_reach, random_reach, waypoint_path, boundary_reach]
    batch_sizes: [1, 8, 64]
    waypoint_counts: [1, 3, 5]
    world: empty
  collision-deployment:
    required_capabilities: [eef_waypoint]
    scenarios: [static_obstacle, randomized_obstacle]
    comparison_mode: asymmetric_information
  atomic-task:
    required_capabilities: [eef_waypoint]
    scenarios:
      - move_end_effector
      - pick_up
      - move_held_object
      - place
      - press
      - pick_move_place

scenario_overrides:
  randomized_obstacle:
    planners:
      curobo:
        config:
          world:
            multi_env: true

track_overrides:
  atomic-task:
    planners:
      curobo:
        config:
          preserve_plan_samples: false

protocol:
  warmup_trials: 3
  measured_trials: 20
  confidence_level: 0.95
  common_control_dt: 0.01
```

Environment variables are only configuration inputs. Save resolved values,
NMG checkpoint hash, cuRobo config hash, and world-content hash in run
metadata.

`free-space-common`, `collision-deployment`, and `atomic-task` should create
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
failures, but must not add a fourth Markdown table.

## 8. Fairness and statistical protocol

### 8.1 Baseline fairness

- Use the same case, start qpos, waypoint order, and external tolerance.
- Default to NMG candidate versus cuRobo primary baseline.
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

Each run produces one `report.md` with exactly three tables.

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
The scenario protocol or a `timing_scope` field must define that operation.

### 9.2 Success & Other Metrics

Aggregate by `(track, scenario, algorithm, condition_bin)`:

```text
track, scenario, comparison_mode, algorithm, algorithm_role,
constraint_information, cases, coverage_rate, success_rate,
planning_success_rate, motion_valid_rate, execution_success_rate,
task_success_rate, final_pos_err_mm, final_rot_err_deg,
waypoint_completion_rate, joint_violation_rate, dynamic_violation_rate,
collision_rate, min_clearance_m, path_efficiency, jerk_cost,
task_metric, top_failure
```

Use `N/A`, not zero, for inapplicable values.

### 9.3 Leaderboard

Use one table with a `track` column:

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
- Treat Cartesian plus joint target as an ablation only.
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

## 11. Implementation phases

### Phase 0: protocol and cuRobo baseline

- Split config, aggregation, and reporting from the current script.
- Preserve the current CLI and checkpoint download behavior.
- Add a cuRobo adapter and make it the default primary baseline.
- Default the CLI/suite to `nmg curobo`; require explicit diagnostic
  IK/TOPPRA baselines.
- Separate cuRobo backend prepare, cold plan, and warm plan timing.
- Produce exactly three report tables.
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
- generated reports contain exactly three Markdown tables;
- planner-only smoke test with a fake NMG checkpoint;
- graceful skip when the optional cuRobo runtime is unavailable;
- cuRobo empty-world, shared-world, and multi-env dynamic-world configuration;
- Atomic Action integration with `motion_source="motion_gen"`.

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
- Add MoveEndEffector, PickUp, MoveHeldObject, Place, Press, and
  Pick-Move-Place.
- Add controller tracking, collision/contact, and stable-hold metrics.

### Phase 3: future NMG capabilities

- Add equal-information obstacle-aware, multimodal, APG-refinement, and
  closed-loop-recovery tracks.
- Add cross-embodiment and unseen-robot tracks.
- Add release profiles and a long-lived checkpoint leaderboard.

## 12. Acceptance criteria

- The current Franka NMG checkpoint and cuRobo run the same default `smoke` and
  `free-space-common` manifest.
- cuRobo is the default primary baseline; IK/TOPPRA do not affect the default
  main leaderboard.
- cuRobo backend preparation and steady-state planning latency are separate.
- Free-space, collision-deployment, and Atomic Action tracks use correct,
  traceable cuRobo world configurations.
- NMG enters supported Atomic Action cases through
  `motion_source="motion_gen"`.
- Planning, motion, execution, and task success remain distinct.
- New planners and metrics register without runner changes.
- Every baseline replays the same fixed manifest.
- Native timing and common execution are not mixed.
- Every run has raw trial artifacts, reproducibility metadata, and one
  Markdown report.
- The report has exactly three tables and a leaderboard covering every
  evaluated algorithm.
- Unsupported capabilities, real failures, and runtime errors remain distinct.
- PR smoke runs finish within a practical budget; coverage/full run in
  nightly/release workflows.
