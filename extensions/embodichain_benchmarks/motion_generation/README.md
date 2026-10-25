# Motion Planning & Atomic Skills Benchmark

The bundled `embodichain_benchmarks` extension owns evaluation orchestration,
baseline adapters, scenarios, scoring policies, reports and suite resources.
It consumes the simulation, planning and Atomic Action APIs in `embodichain`.
The extension ships in the main wheel with the same version and dependencies;
`extensions/` is a source directory, not an import prefix. After updating a
source checkout, refresh its editable mappings with `pip install -e . --no-deps`.

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
`python -m embodichain_benchmarks.motion_generation`. `--suite` accepts
one of the names below or a YAML path. Use `--help` for population, trial,
output and video overrides.

### Call from another repository

```python
from embodichain_benchmarks.motion_generation import (
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

Each call owns its simulator and returns after cleanup. Serial execution is the
default; host calls in one process must run serially. Optional `workers` and
`worker_threads` distribute whole quality cases across isolated spawned processes.
Measure throughput before increasing workers: startup, duplicated initialization
and contention can outweigh gains. Supply `case_manifest` to reuse frozen inputs,
or call `prepare_case_manifest(suite, output, device=...)` at a new output path
without running a planner.

Parallel execution requires installed adapters, headless mode and no video.
Each worker retains its inputs, raw trials, console log and process outcome;
the parent checks exact case/method/repeat coverage and recomputes aggregates
from the complete raw population. Explicit unsupported and lifecycle-error
records remain in the evidence. Worker failures raise and retain diagnostics.
`execution.json` records actual concurrency, CPU threads and elapsed wall time.

Parallel wall times and memory remain in raw records but are excluded from
report timing aggregates. Performance replay requires `workers=1`. ONNX Runtime
uses the host Torch intra-op thread budget and one inter-op thread. Serial calls
retain caller thread settings; `worker_threads` only controls spawned workers.

For frozen B=1 Free-space cases and NMG-only planners, the Python API also
accepts `quality_batch_size=4`. This groups matching K/modality requests into
physical execution slots and composes with spawned workers. The case manifest,
case/seed identities, per-case validator grids and quality weights stay intact;
unused physical slots create no extra trials. Raw valid prefixes and endpoint
finite-difference stencils are restored before individual validation, while
retimed zero-dt padding is removed. Shared batch measurements appear once in
trial metadata; per-case latency fields and timing aggregates are unavailable.
The default is one. Atomic, other backends and performance replay keep their
declared batches; quality batching requires a precomputed case manifest.
Backend-owned random generators are not reset per trial, so stochastic solver
outcomes may vary with case order or sharding even though frozen inputs match.

The suite resources are installed with the package, so callers can run from
any directory. Robot assets and optional cuRobo, TOPPRA or ONNX Runtime
dependencies follow their normal installation and asset-resolution rules.

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
for each represented family. Motion Planning additionally has Workload Success
& Uncertainty and Measured Start Difficulty tables: five tables for Motion
Planning, three for Atomic Skills and eight for a mixed run. Motion Planning has
separate geometry and solved-timing rankings. Built-in Atomic
Skills rank physical success and report motion, execution and end-to-end
success separately. Custom/legacy populations retain their declared primary
metric, which is named explicitly in the ranking. Raw NMG and fixed-clock IK interpolation have no timed rank.

Report schema `3` uses `report_views` and the `motion_planning_*` /
`atomic_skills_*` aggregate tables. `motion_timing_leaderboard` exposes the
solved-timing subset; `motion_planning_workload_summary` and
`motion_planning_measured_difficulty` add population-level uncertainty and
measured-class rates. `generalization` contains domain summaries. There is no
combined cross-family leaderboard. New suites must select schema `3`; existing
artifacts keep their original versions. See the
[report contract](./BENCHMARK_DESIGN.md#8-artifacts-and-report-contract) for
field meanings and denominator rules.

Explicitly unsupported cases are excluded from success-rate denominators; a
population with no applicable cases reports N/A consistently in metric tables
and leaderboards. Missing supported results and runtime failures count as
failures. Unsupported cases still reduce coverage and prevent eligibility.

Policy control steps are auxiliary Free-space diagnostics. Raw outcomes retain
`policy_steps_used`, `policy_completion_step` and strictly ordered
`policy_waypoint_arrival_steps` on the original pre-retiming valid row prefixes.
Initial satisfaction is step 0; missed goals and absent traces remain null.
Raw and retimed share this definition even when retiming fails. These fields
count policy controls, not physical completion or interpolated samples.
Tables report trace coverage, completion rate, conditional mean completion steps,
mean controls used and an all-case completion curve. Failures and missing cases
stay in its case-weighted denominator; a lower conditional mean alone does not
establish better overall quality.

## Configure a suite

Copy the closest suite and change its `suite_version` when altering thresholds,
populations or success semantics. Select `scenario: free_space` or
`scenario: atomic_skill` per track; a second family selector is unnecessary.

Motion Planning populations live only in `tracks[].config`, including when
constructing `SuiteCfg` directly. Edit that mapping to change a track; validation
does not replace it with another configuration. The former top-level
`free_space` field is rejected. The packaged v4 suites
select `sampling_method: cartesian`, with EEF and joint goals built from the
same screened requests, using the declared FK-only numerical solver. Configure `batch_sizes`, `waypoint_counts`,
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

Distances default to metres for compatibility. To compare arms by scale, set
`distance_unit: reach_fraction` and an explicit `robot_reach_m` reference from
your robot's declared/measured reach. Optional `translation_ranges`, such as
`{near: [0.025, 0.075], mid: [0.1, 0.225], far: [0.25, 0.4375]}`, use the selected
unit. These default fractions reproduce the metre bins at a 0.8 m reference;
0.8 m is a conversion reference, not an inferred robot specification. Rotation
ranges remain radians. Resolved metre bounds and reference reach are persisted;
robot reach is never guessed from its name.

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

Generation defaults to `reachability_solver: finite_difference_dls`: a bounded,
multistart numerical solver that uses FK and finite-difference Jacobians, never
`robot.compute_ik` or a measured adapter. `screening_iterations` and
`screening_restarts` bound work per target. `robot_ik` explicitly restores the old
screening route. Every manifest and success summary records the solver and the
selection condition. An independent solver reduces baseline coupling but still
conditions the population on that solver's reachable subset; success on this
population is not an unconditional IK success rate.

Screening honors tighter suite thresholds and fails with `invalid_case` after `max_generation_attempts` rather
than silently dropping difficult inputs. Geometry validity also checks the
initial joint state against `protocol.start_state_tolerance_rad`. Reports show
first-sample joint-error p50/p95/max and the number of measurable rows, including
outputs rejected by that check. The measured start-valid fraction is diagnostic;
missing outputs have no first-sample measurement. The default stays 1e-3 rad.

`path_shape` describes a waypoint layout, and success checks ordered arrivals.
The optional `reference_path_deviation_mm_mean/max` measures sampled translation
distance to the polyline from the requested start through the waypoints. It is
a quality diagnostic, not a continuous path-following certificate or a success
gate. It is N/A for joint goals and pure rotation. EEF reference
joint states remain diagnostic and are not supplied as EEF goals to adapters.

`constraint_patterns` optionally projects semantic masks onto the same independently
screened `sampling_method: cartesian` base cases. Patterns contain `joint`,
`position`, `orientation`, or `pose`; they repeat to K and are eligible only when
their length is <=K. `constraint_pattern_design: full` applies every eligible
pattern. `balanced` applies every singleton plus one rotating pattern per other
length, using the declared seed order. `base_case_id` retains pairing.
Patterns that expand to identical modality sequences at a configured K are
rejected to preserve unique case identities. There is no additional FK random-walk
population. A screened witness certifies goal reachability, not a collision-free
or dynamically feasible path.

All projected cases share maximum absolute per-joint error and independent constrained-component
matching. Validation retains native states and deterministically subdivides
joint-space segments to supply at least `protocol.validation_samples` states;
strictly increasing arrivals and simultaneous pose conditions remain mandatory.
Raw `constraint_metrics` records arrivals, the `linf` success norm, sample-grid
identity and auxiliary L2 error at Joint arrivals. L2 never gates success.
Homogeneous Joint/FullPose projections retain the established classical adapter
input contract. Partial or mixed projections require `masked_constraints`;
unsupported adapters receive neither hidden joint solutions nor implicit extra
pose constraints. `NeuralPlanOptions.waypoint_modalities` passes explicit masks
to NMG; Position requires coincident runtime/policy TCP origins. The common
Joint/FullPose comparison has its own canonical table. Legacy populations without
`constraint_patterns` retain their existing scoring contract.

`run_benchmark(..., case_manifest=path)` replays a hashed manifest after checking
schema, case hashes, shapes, tracks, version and robot. Replay requires the
host-written `resolved_embodiment` identity: `embodiment_id`, `config_hash` and
provider source. Changed robot configuration or embodiment overrides are rejected
before simulation even when the public name and DOF match. Declared performance
subsets and batch sizes retain that identity without regenerating tasks; keep
timing replicas separate from quality denominators.

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
produce N/A. A separate workload summary pools the balanced shape/count and
requested-start allocations while retaining robot, batch size, modality, batch
mode, distance and motion component. Its interval describes its own case-macro
success rate, not the finer rows. Fewer than ten seed clusters are explicitly
marked `few_seed_clusters`; small percentile bootstraps are coarse and unstable.
Unique fingerprints do not turn repeats into independent Bernoulli samples.
The measured-difficulty table classifies each environment row independently on
conditioning (`near_singular_threshold`, default 0.02) and joint margin
(`near_limit_threshold`, default 0.04); near-limit and near-singular can overlap.
Missing repeats and unsupported populations follow the same rate policy.
Declared `held_out` domains do not verify a model's training split.

### Atomic tasks

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

With the Atomic control period, TOPPRA enables
`ToppraPlanOptions.stop_at_waypoints`: adjacent joint-goal pairs stop at rest and
are conservatively extended onto the exact control grid before composition.
Shared goals remain explicit, row prefixes merge before terminal padding, and
`MotionGenerator` preserves solved native samples. This retains stops for gripper
holds. Non-Atomic calls keep the default continuous spline; NMG's request and
rollout budget are unchanged. Phase sample counts remain lower bounds, with
physical execution and independent compiled timing checks determining the result.

The physical standard is `atomic-skill-v2`. Pour checks peak rotation, lift and
retention, plus return to the initial **full orientation** during terminal settle
within the existing rotation tolerance. Catalogues intending upright-object tilt
must choose a horizontal object-local axis. Preserve v1 artifacts and use new
protocol identities for this changed success definition.

Reference preparation first uses exact sequential IK, then may use bounded
search inside the original position/rotation arrival region for rejected rows.
FK and the original joint-bound tolerance check each recovered state against the
original target; probe poses never replace requests. Successful rows stay frozen,
while unresolved rows stop preparation with an error. Endpoint evidence does not
certify a trajectory, execution or physical success; failed search does not prove
impossibility.

The following optional selection rules change frozen populations and require
reviewed identities before comparison. Keep rejected proposals and control
artifacts, and replay identical selected cases for every method:

- **AxisAlign:** `AxisAlignOptions.alignment_twist_rad` selects the free rotation
  about the target axis, as a scalar or one value per row; zero keeps the
  historical shortest arc. `alignment_offset_m` supplies a world translation
  from the lifted object centre, as a vector or one vector per row; zero keeps
  the fixed-centre target. Instead of an offset, `alignment_inward_candidates_m`
  accepts finite, nonnegative, strictly increasing distances starting at zero.
  Preparation samples one original batched grasp pool and tries shifts toward
  the control-part base in world XY. Each row freezes its first complete
  independent IK chain in offset and existing grasp/half-turn order. Accepted
  grasps/offsets stay fixed while unresolved rows continue; resetting the sampler
  seed separately per row would change the original batched stream.
- **Pour:** `grasp_approach_candidates` declares ordered
  `{yaw_offset_rad, max_candidates}` entries. World-Z offsets preserve approach
  elevation, begin at zero and share the original `grasp_max_candidates` budget.
  Each direction uses one original batched pool; each row freezes its first
  complete IK chain and the actual PickUp prefix receives that direction. The
  relative approach bound is unchanged.
- **Pour/AxisAlign:** `grasp_filter_before_ranking` applies the hard TCP-z
  approach bound before NMS, collision, cost ranking and truncation. Omission
  keeps legacy pool construction. Public
  `AntipodalGraspPoseGeneratorCfg.approach_alignment_angle` uses radians and is
  distinct from ray/approach-variant proposal cones. Filtering can change retained
  grasps even when the final bound is unchanged.
- **MoveHeldObject/Place:** `target_candidate_count` defaults to 1 and allows up
  to 32. One retains the original generation path. Larger values try that target
  first, then vary only goal-local position/rotation streams, retaining the root
  seed, task frame, object, grasp, start, offset, randomization amplitudes and
  workspace-shell row radii. Each row freezes its first independently qualified
  complete endpoint chain; manifests record attempted goals and selected indices.
  Exhaustion is an error. This conditioning can change target angle/yaw marginals.

Frozen manifests retain selected values and rejected searches; backend errors
propagate. Selection rules do not certify path feasibility or physical success.

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
- Collision-aware comparison, latency-budget sweeps and recovery benchmarks
  remain future work; process isolation and seed-cluster intervals are available.
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
