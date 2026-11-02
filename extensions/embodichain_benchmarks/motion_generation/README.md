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
The Python API's optional `quality_devices=["cuda:0", "cuda:1"]` assigns shards
round-robin across explicit CUDA ordinals visible to the process. `workers`
remains the total process count; a single worker uses the first listed device.
The original case partition, ordering and raw merge are preserved. Parallel
input preparation retains `device`; performance replay rejects `quality_devices`.
Measure throughput before increasing workers: startup, duplicated initialization
and contention can outweigh gains. Supply `case_manifest` to reuse frozen inputs,
or call `prepare_case_manifest(suite, output, device=...)` at a new output path
without running a planner.

Parallel execution requires installed adapters, headless mode and no video.
Each worker retains its inputs, raw trials, console log and process outcome;
the parent checks exact case/method/repeat coverage and recomputes aggregates
from the complete raw population. Explicit unsupported and lifecycle-error
records remain in the evidence. Worker failures raise and retain diagnostics.
`execution.json` records actual concurrency, device assignments, CPU threads and
elapsed wall time. Each shard and raw trial retains its actual runtime device.

Parallel wall times and memory remain in raw records but are excluded from
report timing aggregates. Performance replay requires `workers=1`. ONNX Runtime
uses the host Torch intra-op thread budget and one inter-op thread. Serial calls
retain caller thread settings; `worker_threads` only controls spawned workers.

For frozen B=1 Free-space cases and NMG-only planners, the Python API also
accepts `quality_batch_size=4`. This groups matching K/modality requests into
physical execution slots and composes with spawned workers. A model with a fixed
batch dimension must match this capacity; incompatible settings fail before
planning. The case manifest,
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

### Native Panda hand

`robot.provider: franka_panda` uses the stock `PandaWithHand.urdf`, including
two prismatic finger joints and `fr3_hand_tcp`. For the same hand on both arms,
select `provider: dual_arm` with `robot.config.robot_type: franka` and
`robot.config.gripper_type: panda_hand`. This duplicates the complete native
asset, preserves the arm mount transforms and exposes `left_arm` / `right_arm`
and two-joint `left_eef` / `right_eef`. The native URDF owns the TCP; omit
`tcp_z`. Each arm solver ends at its prefixed `fr3_hand_tcp` with an identity
additional TCP transform. Give the assembly its own `uid` and `urdf_name`.

Atomic tracks using this hardware declare `config.grasp_model: franka_panda_hand`
and explicit per-hand commands `open_qpos: [0.04, 0.04]` and
`grasp_qpos: [0.0, 0.0]`. Preparation and self-sampling replay both use that
hand's geometry, including its 0.08 m maximum opening. Single-arm declarations
use `gripper.control_part: eef`; dual declarations use the `grippers` list.
Omitting these new selectors retains the existing PGI fixture behavior.

GraspKit returns sampled poses already converted once to the declared runtime
TCP; see its [frame contract](../../../embodichain/toolkits/graspkit/README.md#frames-and-collision-scope).
The Placement provider converts its explicitly authored canonical top/handle
poses to that TCP before freezing them. Frozen replay consumes these runtime-TCP
poses directly, without another hand-specific conversion. This boundary is
separate from the policy/runtime TCP map.

Changing hands changes case generation and requires fresh frozen Atomic inputs.
Explicit Placement top/handle grasps still need native-hand input and physical
qualification; matching TCP kinematics alone does not certify a successful grasp
or task. Existing PGI suites and their generic services remain available.

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
| [`atomic_dual_franka_pgi_hand_over`](./suites/atomic_dual_franka_pgi_hand_over.yaml), [`atomic_dual_franka_pgi_coordinated_placement`](./suites/atomic_dual_franka_pgi_coordinated_placement.yaml) | Nominal B=1 dual Franka + PGI development fixtures with TOPPRA, cuRobo and paired raw/retimed NMG |
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

### Dual Franka planner ownership

The two Franka fixtures evaluate HandOver and CoordinatedPlacement through
named `left_arm` and `right_arm` motion requests. The selected backend generates
each arm segment; the skill owns sequencing, synchronization and gripper commands.
This is task-level coordination with a shared single-arm policy for NMG.
CoordinatedPickment remains a separate scenario-owned IK workload.

Both NMG rows use the same 12-slot exported Franka policy. Their explicit
`policy_frame_from_world_by_control_part` maps each arm's poses into the policy's
training base. The runtime maps both goals and FK observations, including the
retimed arrival check; an unmapped arm raises an error. The fixture TCP-to-policy
mapping accounts for the PGI tool offset and Panda hand rotation. These transforms
are tied to the declared mounts and TCPs: changing either requires updating the
mapping and checking FK against the policy robot. Native cuRobo creates a backend
for each named arm and locks all non-controlled joints; this does not establish
coupled collision avoidance between simultaneously moving arms.

These suites contain one nominal case at seed 11 and keep geometry, timing,
execution and physical success separate. Input preparation freezes both arm
waypoints and sampled grasps independently of the candidate methods. Their
existence does not establish spatial generalization or physical success.
New coordinated cases also retain `case_parameters.endpoint_qualification`:
independent sequential IK witnesses and original-goal FK, finite-state and joint
limit checks for every endpoint of each arm. Both arms start from their frozen
joint slices and use named-arm arena-frame TCP transforms. The record includes
per-row residuals and the unchanged suite thresholds. A failed check stops case
preparation without deleting rows or resampling geometry; a missing IK witness
does not prove unreachability. These joint witnesses are preparation evidence,
not candidate planning seeds or targets, and certify neither paths nor physics.
Previously frozen manifests remain unchanged.
Placement additionally retains `placement_input_geometry` from the live scaled
meshes and frozen targets. It checks complete XY projected-triangle-union
containment in the basin (preserving holes and excluding handle triangles) and
vertical gaps at mesh vertices, footprint boundaries and a 5 mm interior grid.
Containment uses a 1e-12 m precision grid after the triangle unions to remove
floating-point overlay holes. Vertical-ray samples retain the original unsnapped
projections and the zero-gap threshold. The evidence records this precision;
requalifying frozen cases preserves their targets, grasps and endpoint witnesses.
The record includes mesh/topology hashes, unsupported area, boundary clearance,
sample count and minimum sampled gap. Failed rows remain in this input audit;
the experiment's preparation gate must review them before evaluating methods.
This is neither continuous 3D collision certification nor physical task success.

### Paired-arm performance replay

`BenchmarkRunner.run_paired_arm_performance()` measures complete pairs of
independent arm requests from frozen Atomic cases. It currently accepts one
Atomic track and scene batch, with exactly two named FullPose requests per
group. HandOver exposes ready transport/receiver requests and
CoordinatedPlacement exposes ready approaches. The facade executes classical
backends sequentially; an explicitly enabled NMG backend also runs a packed
dispatch for comparison. Task dependencies and physical execution are unchanged.

```python
from embodichain_benchmarks.motion_generation import load_suite
from embodichain_benchmarks.motion_generation.runner import BenchmarkRunner

suite = load_suite("resolved_dual_suite.yaml")
runner = BenchmarkRunner(
    suite,
    [spec for spec in suite.planners if spec.enabled],
    case_manifest="frozen_atomic_cases.json",
    device="cuda",
)
summary = runner.run_paired_arm_performance(
    output="paired_performance/run_001",
    request_manifest="frozen_paired_requests.json",
    warmup_trials=2,
    measured_trials=10,
    source_cases_per_suite=1,
)
```

The resolved suite must declare exactly one TOPPRA reference, and selected
methods must have their dependencies and model paths configured. Source cases
are selected by sorted case ID. Before any candidate measurement, reference
compilation captures each group's starts, goals and task options into an
immutable request cache. Its identity binds the source cases, reference
configuration, suite protocol and resolved embodiment. Changed identities or
corrupted requests raise errors; cached inputs are never silently regenerated.
Every repeated call receives its own input copy. Missing backends, invalid
preparation and a reference that never reaches a ready group are errors.

Run this entry point serially. Warmups, validation and physical execution are
outside latency measurement; the synchronized pair call includes FK, transfers
and the selected method's retiming. Sequential and packed dispatch order
alternates across repeats. Their native positions, timing, derivatives and
success masks are compared outside timing, and both arms are independently
scored. Inspect `dispatch_parity` before interpreting a speed difference as an
equivalent-behavior comparison. Repeated calls do not enlarge the task-quality
population; paired motion geometry is distinct from physical task success.

Memory is measured on a separate warmed call. CPU values are retained RSS
deltas; GPU values cover Torch allocations and exclude ONNX Runtime, cuRobo
and other native allocator memory. Small-repeat p95 values are descriptive.
The fresh output directory contains `requests.json`, `resolved_suite.yaml`,
`planners.json`, `trials.jsonl`, `aggregates.json`, `execution.json` and
`report.md`, with time/memory, success and leaderboard tables kept separate.
Actual planner metadata records resolved motion limits and model/configuration
identity; the cache remains independent of candidate checkpoint selection.

### Physical replay diagnostics

Atomic tracks may opt into bounded measured evidence with
`physics.replay_diagnostics: {enabled: true, samples_per_segment: 8, max_contacts_per_env: 64}`.
The default is disabled. Trial metadata records the resolved switch, and
`metadata.physical_replay_trace` persists initial state, selected command ends
from every compiled action segment (including prefix PickUps), and final settle.
It retains scene-batched object/TCP poses in arena coordinates, measured and
commanded gripper positions, simulation times, and contact actor identities,
impulses, signed separation and overflow counts when the backend supports them.
Unavailable contact channels remain unavailable. Sparse contact absence does
not establish absence throughout a segment. The trace supplements the unchanged
physical scoring windows; collection time is included in execution timing and
excluded from planner timing. Runs with different diagnostic settings must not
be combined to claim execution performance.

Placement target generation centers the placing mesh footprint over the support
basin and preserves its orientation and bottom-clearance rule. It uses actual
mesh bounds rather than a fixed world offset; asset origins need not be centered.
With triangle topology, the support height is the uppermost surface intersected
by a vertical ray through the basin center, so a raised rim does not set the
resting height. Missing center support raises an error; bounds-only callers
retain the upper-bound rule. The dual Franka fixture uses bread scale 1.20,
chosen from the gripper opening and object footprint clearance.
The Franka/PGI placing grasp uses zero TCP clearance above the mesh top, allowing
the authored fingertips (about 15 mm beyond its 150 mm TCP) to overlap the object
sides. Independent physical qualification is still required for each fixture.
Previously frozen cases retain their recorded grasps and targets.

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

`report.md` separates Motion Planning geometry and solved-timing rankings from
Atomic physical success. Motion, execution and end-to-end success are reported
separately. Raw NMG and fixed-clock IK interpolation have no timed rank.

New suites use report schema `3`; existing artifacts retain their recorded
versions. The [report contract](./BENCHMARK_DESIGN.md#8-artifacts-and-report-contract)
defines family tables, aggregate fields and denominator rules. See
[capability gates](./BENCHMARK_DESIGN.md#5-scenario-tracks-and-capability-gates)
for unsupported populations and eligibility.

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

The classical adapters opt into this expanded scope with
`planner.config.masked_constraints: true`; omission retains their historical
Joint/FullPose scope. IK+TOPPRA resolves partial/mixed sequences in order, using
only each active Cartesian component or explicit Joint target and the preceding
solution as its seed. Its masked DLS solve has bounded independent restarts and
never consumes the screened Cartesian joint witness. Configure its fixed budget
with `config.masked_ik`: `iterations: 96`, `restarts: 4`,
`position_tolerance: 0.005`, `rotation_tolerance: 0.05`, `damping: 0.02`.
These are solver tolerances, separate from the common evaluator thresholds.
The resulting joint knots are passed to TOPPRA; an unsolved required token fails
the row. The homogeneous Joint and FullPose routes retain the existing joint
and Cartesian-interpolation pipelines.

cuRobo receives the original ordered joint/Cartesian sequence and uses its native
terminal pose criteria for Position and Orientation. Nonterminal pose weights
remain zero; waypoint tasks impose no extra path constraint. Each subsequent
FullPose goal restores all six terminal weights. Unused payload fields are
discarded before frame conversion. A Position goal requires coincident runtime
TCP and cuRobo tool origins; auto-derived stock Franka profiles meet this condition.
Changing the supported comparison population requires a distinct reviewed
suite/method fingerprint and reevaluation; this opt-in does not relabel old results.

For selected-backend Atomic calls, adapter-injected planner options retain the
suite's full joint-budget vectors. Generic scenario defaults are cleared on
those invocations, so cuRobo prepares and plans with the same limits instead of
rebuilding each arm for the default 0.5 rad/s² acceleration. Explicit invocation
options remain unchanged. Scenario-owned IK retains its scalar motion budget.

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

Geometry and solved-timing leaderboards cover the complete selected population,
including Joint, Position, Orientation, FullPose and mixed constraints. Time and
success tables retain target modality, batch mode, distance and motion
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
