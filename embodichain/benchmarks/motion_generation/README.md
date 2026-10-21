# Motion Planning & Atomic Skills Benchmark

One framework evaluates two families selected by `TrackCfg.scenario`:

| Family | Scenario | Workload |
| --- | --- | --- |
| Motion Planning | `free_space` | EEF/joint goals, ordered waypoints, batches, path shapes and start bins |
| Atomic Skills | `atomic_skill` | Skill compilation, physics replay and measured physical stages |

They share planner adapters, robot configuration, motion limits and raw
artifacts, while keeping their populations and rankings separate. See the
[design contract](./BENCHMARK_DESIGN.md) for scoring, timing and fairness rules.

## Run

Run after installing EmbodiChain and the selected backend's dependencies:

```bash
embodichain benchmark motion-generation --suite motion_panda_free_space_smoke
embodichain benchmark motion-generation --suite atomic_franka_pgi_smoke --device cuda
```

The equivalent Python entry point is
`python -m embodichain.benchmarks.motion_generation`. `--suite` accepts
one of the names below or a YAML path. Use `--help` for population, trial,
output and video overrides.

### Call from another repository

```python
from embodichain.benchmarks.motion_generation import (
    PlannerSpecCfg,
    load_suite,
    run_benchmark,
)

suite = load_suite("motion_panda_free_space_smoke")
suite.protocol.measured_trials = 5
result = run_benchmark(
    suite,
    planners=[PlannerSpecCfg(id="toppra", adapter="toppra")],
    device="cuda",
    output_root="./evaluation_results",
)
print(result.aggregates)
print(result.report_path)
```

Omit `planners` to use the suite's enabled algorithms. An explicit planner list
replaces the suite's list and enables every supplied entry, independent of its
`enabled` default. Suite objects, planner specs and adapter mappings are not
modified. Pass a YAML `Path` or filename for a project-owned suite; component
paths then resolve relative to that file.

For an external algorithm, implement `PlannerAdapter` in the calling repository:

```python
from my_project.evaluation import MyPlannerAdapter

result = run_benchmark(
    suite="motion_panda_free_space_smoke",
    planners=[PlannerSpecCfg(
        id="my_model", adapter="my_model", role="candidate",
        config={"checkpoint": "./model.pt"},
    )],
    adapters={"my_model": MyPlannerAdapter},
    device="cuda",
)
```

Planner case tensors and nested parameters are copied outside timing before
capacity checks, preparation and calls, preserving the canonical manifest and
external metric inputs even when an adapter mutates its input in place.

Mappings are local to one run and can override a registered adapter for that
call. Load models in `build()` or `prepare()`, return the canonical batched
`PlanResult` from `plan(case)`, and release resources in `close()`. Declare
capabilities such as `eef_waypoint`, `joint_waypoint`, `batched` and `empty_world`
according to actual support. Timed backends declare `timing_is_solved` and use
the shared motion budgets. Atomic Skills also require `atomic_action` and an
adapter-owned `MotionGenerator`; an arbitrary free-space adapter alone does not
provide that integration.

Each call owns its simulator and returns after cleanup. Run calls serially in
one process. The suite resources are installed with the package, so callers can
run from any directory. Robot assets and optional cuRobo, TOPPRA or ONNX Runtime
dependencies still follow their normal installation and asset-resolution rules.

| Suite | Use |
| --- | --- |
| [`motion_panda_free_space_smoke`](./suites/motion_panda_free_space_smoke.yaml) | Small Panda EEF/joint planning check |
| [`motion_panda_free_space_coverage`](./suites/motion_panda_free_space_coverage.yaml), [`motion_ur5_free_space_coverage`](./suites/motion_ur5_free_space_coverage.yaml) | Planning coverage across batches, path shapes and start-state bins |
| [`atomic_franka_pgi_smoke`](./suites/atomic_franka_pgi_smoke.yaml) | Single-arm motion integration and six interaction skills |
| [`atomic_franka_pgi_randomized`](./suites/atomic_franka_pgi_randomized.yaml), [`atomic_ur5_pgi_randomized`](./suites/atomic_ur5_pgi_randomized.yaml) | Seeded single-arm interaction sweeps |
| [`atomic_franka_pgi_pose_batch`](./suites/atomic_franka_pgi_pose_batch.yaml) | Batched object/target translation perturbations |
| [`atomic_franka_pgi_door_pour_align_push`](./suites/atomic_franka_pgi_door_pour_align_push.yaml) | OpenDoor, Pour, AxisAlign and PushObject |
| [`atomic_franka_pgi_spatial_generalization`](./suites/atomic_franka_pgi_spatial_generalization.yaml) | Ten task-level skills across nominal, pose and primitive-geometry domains |
| [`atomic_dual_ur5_pgi_hand_over`](./suites/atomic_dual_ur5_pgi_hand_over.yaml), [`atomic_dual_ur5_pgi_coordinated_pickment`](./suites/atomic_dual_ur5_pgi_coordinated_pickment.yaml), [`atomic_dual_ur5_pgi_coordinated_placement`](./suites/atomic_dual_ur5_pgi_coordinated_placement.yaml) | Dual UR5 + PGI physical skills with TOPPRA |
| [`atomic_dual_ur5_pgi_hand_over_spatial_generalization`](./suites/atomic_dual_ur5_pgi_hand_over_spatial_generalization.yaml), [`atomic_dual_ur5_pgi_coordinated_pickment_spatial_generalization`](./suites/atomic_dual_ur5_pgi_coordinated_pickment_spatial_generalization.yaml), [`atomic_dual_ur5_pgi_coordinated_placement_spatial_generalization`](./suites/atomic_dual_ur5_pgi_coordinated_placement_spatial_generalization.yaml) | Dual-arm pose, task-frame and composed domains |
| [`atomic_ur5_embodiment`](./suites/atomic_ur5_embodiment.yaml) | Component-defined UR5 embodiment with nominal/object-XY PickUp domains |

Motion Planning suites enable cuRobo as the primary baseline and TOPPRA / IK
interpolation as diagnostics. To include the optional paired NMG candidates,
select their IDs and supply a robot-compatible ONNX policy:

```bash
embodichain benchmark motion-generation \
  --suite motion_panda_free_space_smoke \
  --algorithms curobo toppra ik_interpolate nmg nmg_retimed \
  --nmg-onnx-path /path/to/policy.onnx
```

A model path alone does not enable an algorithm. The paired NMG entries share
one YAML policy configuration; `nmg_retimed` adds shared-limit retiming.

For visual inspection:

```bash
embodichain benchmark motion-generation \
  --suite atomic_franka_pgi_smoke --device cuda \
  --record-video --video-case-limit 0
```

The video limit is global: `0` records every successful measured Atomic Skill
case; `1` records at most one case. Add `--record-failed-video` for static failed
case scenes, or use `--no-headless` to open the live simulator viewer. Video
capture happens after measured evaluation.

Other benchmark domains use separate commands: `rl`,
`robotics-kinematic-solver`, `grasp-pose-generator`, `workspace-analyzer`, and
`rendering`. `embodichain benchmark atomic-action-profile --action pick_up
--profile smoke` profiles individual actions over object and position presets;
its reports do not contribute to this framework's planner rankings.

## Read the results

Artifacts are saved to `outputs/benchmarks/<suite-name>/<timestamp>/`:

```text
resolved_suite.yaml
environment.json
case_manifest.json
trials.jsonl
aggregates.json
report.md
videos/                 # optional
```

`report.md` contains Time & Memory, Success & Other Metrics, and Leaderboard
for each represented family: three tables for a single family, six for a mixed
run. Motion Planning has separate geometry and solved-timing rankings. Built-in Atomic
Skills rank physical success and report motion, execution and end-to-end
success separately. Custom/legacy populations retain their declared primary
metric, which is named explicitly in the ranking. Raw NMG and fixed-clock IK interpolation have no timed rank.

Report schema `3` uses `report_views` and the `motion_planning_*` /
`atomic_skills_*` aggregate tables. `motion_timing_leaderboard` exposes the
solved-timing subset; `generalization` contains domain summaries. There is no
combined cross-family leaderboard. New suites must select schema `3`; existing
artifacts keep their original versions. See the
[report contract](./BENCHMARK_DESIGN.md#8-artifacts-and-report-contract) for
field meanings and denominator rules.

Explicitly unsupported cases are excluded from success-rate denominators; a
population with no applicable cases reports N/A consistently in metric tables
and leaderboards. Missing supported results and runtime failures count as
failures. Unsupported cases still reduce coverage and prevent eligibility.

## Configure a suite

Copy the closest suite and change its `suite_version` when altering thresholds,
populations or success semantics. Select `scenario: free_space` or
`scenario: atomic_skill` per track; a second family selector is unnecessary.

Motion Planning populations live in `tracks[].config`. The packaged v3 suites
select `sampling_method: cartesian`, with EEF and joint goals built from the
same independently screened requests. Configure `batch_sizes`, `waypoint_counts`,
`path_shapes`, `start_state_bins`, `seeds`, `batch_modes`, `distance_bins` and
`motion_components`. For example:

```yaml
sampling_method: cartesian
population_design: balanced
batch_sizes: [1, 8, 64]
batch_modes: [homogeneous, heterogeneous]
distance_bins: [near, mid, far]
motion_components: [translation, rotation, combined]
waypoint_counts: [1, 3, 5]
path_shapes: [direct, l_turn, s_curve]
start_state_bins: [nominal, random_reachable, near_limit, near_singularity]
seeds: [11, 23, 37]
target_types: [eef, joint]
```

Pure translation holds TCP orientation; pure rotation holds TCP position.
L/S shapes describe Cartesian waypoint geometry; turns need at least three
waypoints, and pure rotation uses `direct`. `balanced` rotates meaningful
shape/count pairs across seeds instead of multiplying the full geometry matrix;
`full` enumerates every meaningful pair. Each declared start population varies
across seeds. Homogeneous batches repeat one request; heterogeneous batches
sample each row. The manifest records input fingerprints and measured TCP
motion, joint margins and scaled-Jacobian conditioning. `near_singularity`
requests select the least-conditioned candidate in a bounded pool; the measured
`conditioning_class` states whether it actually crosses the near-singular
threshold. It is not assumed from the requested label.

Generation uses independent sequential IK and FK, honors tighter suite
thresholds and fails with `invalid_case` after `max_generation_attempts` rather
than silently dropping difficult inputs. Geometry validity also checks the
initial joint state against `protocol.start_state_tolerance_rad`. EEF reference
joint states remain diagnostic and are not supplied as EEF goals to adapters.

Omitting `sampling_method` retains the legacy `joint_templates` population.
Its L/S/wrist templates describe joint displacements; those labels do not certify
Cartesian geometry or measured singularity. Keep its results under their original
suite version. Shared motion budgets remain under `protocol`; conflicting
adapter-local budgets are rejected. Acceleration is a benchmark policy, not a
hardware specification.

Time and success tables retain target modality, batch mode, distance and motion
component. Success rows include unique input counts, measured difficulty and
seed-cluster bootstrap intervals at `protocol.confidence_level`. Repeated calls
and batch rows stay within their seed cluster; fewer than two seed clusters
produce N/A. Small seed populations limit interval precision, and declared
`held_out` domains do not verify a model's training split.

Atomic tracks own their objects, skill configurations, physics and pose
randomization. Packaged suites select `physics.clock: compiled`: compilation
uses `physics_dt * steps_per_waypoint` as its command period, and replay follows
the compiled arrival intervals on unchanged physics ticks. Rows share each
sample's maximum interval, with shorter rows holding; completion time includes
the actual scheduled ticks and terminal settling. Segment indices remain
unchanged. Legacy `fixed_samples` is retained when the clock is omitted and
uses `steps_per_waypoint` physics ticks for every sample. The clock and command
period are recorded in trial metadata. Use `randomization.pose_batch_size: K` with `batch_sizes: [K]`
for one vectorized skill request over `K` independently perturbed environment
rows. The supplied pose-batch suite uses fixed cube grasps; articulated Slide
shares one target-local grasp across its translated rows. Track configs may
select `grippers` for multiple hands and `fixtures` for static supports. Cases
can retain extra objects through `scene_object_poses`.

### Embodiments

Select a registered `robot` provider or an official embodiment component:

```yaml
embodiment:
  component: embodichain_tasks/configs/components/embodiments/ur5_dh_pgi_140_80.yaml
  overrides: {}
```

The runner builds the component's simulation robot and preserves its TCP,
motion endpoint and gripper command preset. Do not repeat inline `robot`
configuration or conflicting track-level gripper commands. Multiple motion
resources require an explicit `embodiment.resource_id`; this selects one
resource, not a coordinated dual-arm benchmark. Relative component paths
resolve from the suite file; packaged references work from any working
directory. The resolved identity, bindings, capabilities, services and stable
hash are recorded in the manifest.

Both robot-provider and component suites reject cases whose robot ID or
explicit embodiment ID conflicts with the resolved suite identity. An omitted
embodiment ID is filled from that identity. Component-loading tests do not
establish physical success; `atomic_ur5_embodiment` requires live GPU physics
and cuRobo.

### Domain provenance

Add `domains` beside `scenario`, retaining the track's `config`:

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

Each domain requires nonempty `id`, `version`, and a `kind` of `nominal`,
`robustness` or `held_out`. Optional overrides, objects, capabilities,
perturbations and frozen-case metadata produce separate effective tracks with
a shared `track_group`. The loader does not mutate the source suite. A single
`domain` entry is also supported; undeclared domains remain null.

Generalization eligibility requires a complete nominal baseline and at least
one complete non-nominal domain for the same track and attributed algorithm.
Retention is N/A for an incomplete or zero nominal baseline. The
[population contract](./BENCHMARK_DESIGN.md#6-domain-populations-and-generalization)
describes pose, primitive-geometry and composed domains.

## Extend

- Planner: implement and register [`PlannerAdapter`](./planners/base.py),
  declaring capabilities and timing semantics. Atomic Skills additionally
  require an exposed `MotionGenerator` and the `atomic_action` capability.
- Robot: register a [`RobotProvider`](./robots/base.py); gripper-action suites
  also configure the control part and open/grasp joint commands.
- Object: configure a built-in `cube` / `mesh` entry, or register an object-kind
  factory in [`atomic_objects.py`](./scenarios/atomic_objects.py).
- Atomic Skill: implement [`AtomicSkillCaseProvider`](./atomic_skill.py),
  register it, declare `requires_gripper` when needed, and supply a measured physical evaluator. Reuse the lifecycle
  and evidence owners below.

### Provider and evaluator interfaces

| Owner | Responsibility |
| --- | --- |
| [`atomic_skill.py`](./atomic_skill.py) | Provider contract and skill registry |
| [`scenarios/atomic_skill.py`](./scenarios/atomic_skill.py) | Scene setup/reset, cases, compilation, evaluation, video and teardown |
| [`atomic_case_utils.py`](./scenarios/atomic_case_utils.py) | Case identity, typed pose-context adaptation, provenance and goal helpers |
| [`_atomic_pose_sampling.py`](./scenarios/_atomic_pose_sampling.py) | Pure deterministic pose sampling and legacy-context decoding |
| [`_free_space_sampling.py`](./scenarios/_free_space_sampling.py) | Reachable Cartesian workloads and measured difficulty |
| [`atomic_policy.py`](./scenarios/atomic_policy.py) | Motion policy, source attribution and budgets |
| [`atomic_grasps.py`](./scenarios/atomic_grasps.py) | Frozen and case-scoped grasp generators |
| [`atomic_evaluation.py`](./scenarios/atomic_evaluation.py) | Measured evidence and shared evaluators |
| [`atomic_replay.py`](./scenarios/atomic_replay.py) | Replay runtime shared by evaluation and video |
| [`atomic_motion_cases.py`](./scenarios/atomic_motion_cases.py), [`atomic_manipulation_cases.py`](./scenarios/atomic_manipulation_cases.py), [`atomic_contact_cases.py`](./scenarios/atomic_contact_cases.py) | Base motion, manipulation and articulated-contact providers |
| [`atomic_skill_providers.py`](./scenarios/atomic_skill_providers.py), [`atomic_dual_arm_providers.py`](./scenarios/atomic_dual_arm_providers.py) | Additional single-arm and dual-arm providers |

Provider hooks cover preconditions and skill-specific sampling:
`prefix_invocations()` supplies prerequisite actions, `initial_scene()` adds
initial observations, `frozen_grasps()` reuses manifest grasps, and
`replay_observation_windows()` selects measured replay channels from compiled
segment coordinates. `object_rotation_segment()` selects a peak-rotation
measurement window.

All 15 built-in skills use `physical_evaluator()`. Implement
`contracts.PhysicalStageEvaluator.evaluate(case, observation, motion_outcomes)`
and return one `PhysicalEvaluation` per environment with consistent ordered
stage IDs. Custom providers without a physical evaluator can use the existing
`task_result()` fallback. The [physical evaluation contract](./BENCHMARK_DESIGN.md#4-physical-evaluation)
defines evidence boundaries, stage progression and success semantics.

## Current limits

- cuRobo receives an empty external collision world in the supplied suites.
  Physical contact replay does not certify collision safety against non-target
  links or environment geometry. Dual-arm suites use TOPPRA; coordinated skills
  currently reject cuRobo.
- General per-row geometry-sampled antipodal candidate selection remains
  single-environment; the supplied single-arm pose-batch suites use Franka + PGI.
- Collision-aware comparison, latency-budget sweeps, subprocess isolation and
  recovery benchmarks remain future work; seed-cluster intervals are available.
- Embodiment sensor acquisition and Task Program runtime services are outside
  this host. Other benchmark domains under `scripts/benchmark/` still require
  a source checkout.

## Validate

Contract tests run without the simulator engine:

```bash
python -m pytest -q tests/benchmark/motion_generation/test_contracts.py
```

Run loader, runner, evaluator and reporting regressions in the EmbodiChain
Python environment:

```bash
python -m pytest -q tests/benchmark/motion_generation
```
