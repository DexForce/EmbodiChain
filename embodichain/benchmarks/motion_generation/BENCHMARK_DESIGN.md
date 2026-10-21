# Motion Planning & Atomic Skills Benchmark Design

This document describes the implemented comparison and evaluation contracts.
The [README](./README.md) owns commands, suite selection, configuration examples
and extension entry points. Unimplemented capabilities are collected in
[future extensions](#9-future-extensions).

## 1. Scope and comparison boundaries

The framework answers three separate questions: how much generation costs,
whether the generated motion satisfies its constraints, and whether executing
an Atomic Skill produces the intended physical result. A planner success flag
alone establishes neither trajectory validity nor physical success.

`TrackCfg.scenario` selects the report family:

- `free_space` owns Motion Planning populations and independent geometry /
  solved-timing rankings. It covers planner scaling across EEF and joint goals,
  batches, ordered waypoints, path shapes and start-state bins.
- `atomic_skill` owns skill compilation, replay and physical-stage ranking.
  The smoke suite retains MoveEndEffector and MoveJoints as integration checks;
  interaction and generalization suites exercise physical tasks.

There is no combined score across these families. The separate
`atomic-action-profile` command diagnoses individual actions and does not enter
this framework's rankings.

## 2. Ownership and fixed inputs

The execution path is:

```text
Suite YAML -> scenario cases -> frozen manifest -> planner adapter
                                              -> raw lifecycle/trial records
                                              -> external evaluation
                                              -> family aggregates and report
```

`config.py` owns configuration and domain expansion; `runner.py` owns the case
matrix, planner lifecycle and artifact writing. Scenario providers own scene
state, frozen inputs and evaluation. Planner adapters own backend construction,
preparation, capability checks and generation. `metrics/trajectory.py` validates
motion; `aggregation.py`, `generalization.py` and `reporting.py` consume records
without invoking planners or simulators. Atomic provider/replay owners are
listed in the [extension guide](./README.md#provider-and-evaluator-interfaces).

Capacity checks, preparation and planning receive untimed copies of canonical
case tensors and nested parameters. External metrics and manifests use the
canonical inputs. Every planner receives the same case's robot start state, ordered targets,
objects, grasps, seed and physics parameters. Cases record independent
reachability evidence before measured calls. A planner being evaluated cannot
select the cases that enter its own success denominator. EEF and joint goals
have distinct identities: `reference_qpos` is diagnostic for an EEF goal and the
actual ordered target for a joint goal. Adapters must not replace an EEF target
with its reference joint solution.

The manifest records full start state, case parameters, embodiment/domain
identity and required capabilities. Each track/batch population is written
before any planner evaluates it; later populations extend the manifest. The
scenario restores robot, object and simulator state before replay.

The NMG policy belongs to its planner configuration. Objects, grasp sampling,
controllers and physical evaluators must remain independent of the selected
model. Optional `nmg` and `nmg_retimed` candidates share the same checkpoint and
inputs so their comparison isolates the effect of timing the policy path.

## 3. Motion, timing and limit contracts

### Geometry and solved timing

External evaluation recomputes errors and constraints from the output:

```text
geometry_valid = requested_start_state_matched
                 and finite_positions
                 and ordered_waypoints_reached
                 and joint_position_limits_satisfied

timed_motion_valid = geometry_valid
                     and valid_solved_timing
                     and applicable_dynamic_limits_satisfied
```

Waypoint matching requires monotonically increasing arrival samples. EEF
position and orientation must satisfy their thresholds at the same sample;
joint targets use an explicit per-joint error threshold. Best hits at unrelated
samples or out-of-order arrivals do not satisfy the requested path.

`PlanResult` carries `dt` whenever positions exist, including for methods that
only provide a nominal clock. `timing_is_solved` distinguishes those outputs
from trajectories whose timing was solved under limits. Raw NMG and fixed-clock
IK interpolation participate in geometry ranking; TOPPRA, cuRobo and retimed
NMG can additionally participate in timed ranking. Nominal-clock dynamics remain
diagnostics, and unavailable timed validity/duration is N/A. Continuous path
quality is conditioned on geometry-valid outcomes consistently across methods.

Velocity, acceleration and jerk are recomputed using compact divided
differences. Central differences could hide a trajectory reversing at every
sample. Derivatives that cannot be resolved from the available samples remain
unavailable. `motion_valid` remains a raw outcome field for controller and
end-to-end evaluation; it does not replace the separate geometry/timing views.

### Shared budgets

Planning and validation use the same effective joint-speed caps: valid runtime
limits, positive URDF fallback for unavailable runtime values, and propagated
mimic constraints. An explicit suite speed request can fill a missing limit or
reduce a known cap; it cannot enlarge a usable robot limit. Acceleration and
jerk have no automatic asset fallback and are suite-owned budgets.

Timing adapters reject unavailable required limits or conflicting
adapter-local budgets. Scenario-owned IK uses scalar `MotionPolicy` limits,
so per-joint suite vectors are conservatively reduced to their minimum and
recorded in `policy_motion_limits`. Backend adapters receive the full vector.
Retiming does not guarantee jerk compliance; an explicitly configured jerk
budget is still checked independently.

### Effective motion source

Planner-backed Atomic Skills use the adapter's `MotionGenerator` with
`strategy="motion_gen"`. Exact Cartesian skills may require `ik_interp`;
coordinated skills may own their IK path. Records carry
`selected_planner_backend_used` and `effective_motion_algorithm`, so a bypassed
backend receives no credit for motion it did not produce.

CoordinatedPickment uses its own IK projection and fixed sample clock, so its
timing is nominal. Dual-arm cases freeze each arm's requested poses and require
both arms to reach them in order. Grasp sampling is case-scoped to prevent one
case's configuration from changing another's trajectory.

### Measurement boundaries

Construction, lazy backend preparation, cold calls, warmup and measured calls
are separate `TrialPhase` values. Warmup does not enter steady-state aggregates.
cuRobo preparation can include sphere fitting, backend creation and CUDA graph
capture; NMG construction includes checkpoint loading. Prepare/warm each
required input shape before measuring it.

The measured operation is `planner_call` for Motion Planning and
`atomic_action_compile` for Atomic Skills. Their report fields are
`planning_time_ms` and `compile_time_ms`. Physics execution and end-to-end time
are separate, and compile latency is never ranked against a bare planner call.
Validation, case generation, reporting and optional video capture are outside
the timed operation. CUDA wall timing synchronizes around the operation.

Native generation preserves planner samples where configured. Derived path
metrics may resample copies. Packaged Atomic suites select `physics.clock:
compiled`, compile on the declared command period (`physics_dt` times
`steps_per_waypoint`) and replay compiled arrival intervals on unchanged
physics ticks. Every sample uses the maximum interval across rows; shorter
rows hold while other rows advance. Cumulative quantization adds at most one
physics tick to the shared duration and preserves compiled segment indices.
Completion time counts scheduled ticks plus terminal settling. Legacy
`fixed_samples` replay retains its explicit per-sample stepping, and trial
metadata identifies both its clock and command period. Motion limits may extend the requested
sample count. Native timing and replay timing must remain identifiable.
Multi-waypoint latency measures the complete request, including
cuRobo's sequential segments; a per-segment diagnostic is not a replacement
for that cost.

CPU memory is retained RSS delta. GPU allocation and peak fields cover the
Torch allocator, not all native/CUDA-graph allocations. Persistent backend
caches and process ordering limit cross-run memory/cold-start comparisons;
subprocess isolation is not yet implemented.

## 4. Physical evaluation

All built-in skills use `PhysicalStageEvaluator` over measured replay
observations. The evaluator receives frozen case inputs, `ExecutionObservation`
and motion outcomes. It does not receive the scenario/engine,
`CompiledTrajectory` or projected symbolic context, so predicted effects cannot
substitute for physical evidence.

Providers declare `replay_observation_windows()` while the compiled segment
map is available. Each window selects raw TCP, active-object, scene-object,
articulation-joint or gripper-joint channels and can include terminal settling.
Replay also records configured gripper commands. Missing optional observations
remain unavailable evidence. Push contact uses a labeled object-motion proxy;
it is not contact-sensor evidence.

Contact replay retains target-joint traces and peak signed displacement through
release/retract/hold as diagnostics. Evaluators use their declared stage windows:
Press permits spring rebound after a successful press; other articulated skills
also require their grasp, motion and release stages. A diagnostic peak over the
whole replay cannot replace those stage criteria. The actual skill stages and
thresholds are owned by `evaluation.py` and the provider evaluators.

Return one `PhysicalEvaluation` per environment with the same ordered stage
IDs. Stage status is `passed`, `failed`, `not_reached`, `not_applicable` or
`unavailable`. Failed and unavailable stages require failure codes; measurements
use finite numeric values with units in their names. A failed, unavailable or
unreached stage prevents subsequent stages from being reached. Duplicate/missing
rows, duplicate stages and
inconsistent ordering are evaluator errors, recorded as
`metric_evaluation_error`.

Physical success requires at least one applicable stage and every applicable
stage to pass. An incomplete or all-inapplicable result cannot establish
success. `physical_skill_success` remains independent of motion diagnostics;
`end_to_end_success` additionally requires planning, valid motion and successful
execution. The raw `task_success` field retains that overall meaning. Custom
providers without a physical evaluator can still implement `task_result()`;
the built-ins do not maintain a second scoring path.

The Atomic Action runtime's `PhysicalEvidenceRequest`, `PhysicalEvidenceBatch`
and `PhysicalEvidenceFrame` provide a separate evidence port for execution
verifiers, phase gates and held-object guards. `ExecutionRunner` can collect an
aligned frame after a fresh observation. Primitives declare evidence
requirements rather than reading or mutating simulator objects. Runtime
verifiers and benchmark evaluators can adapt the same evidence source without
sharing their orchestration.

The supplied cuRobo comparisons use an empty external collision world even
when physical articulations exist in the simulator. Replay establishes measured
task outcomes, not collision safety against non-target links or environment
geometry. Held-object geometry is not dynamically attached to the planner;
this constraint-information gap must remain visible in result interpretation.

## 5. Scenario tracks and capability gates

The implemented scenarios are `free_space` and `atomic_skill`. Free-space
requires `batched` and `empty_world` capabilities, then checks EEF or joint
support per case. Atomic Skills also check provider-specific requirements.

- `supported`: run the mandatory case normally.
- `unsupported`: record the reason and exclude it from the success-rate
  denominator; it still lowers coverage.
- Runtime error or failed execution after support is declared: count as failure.

A population with no applicable cases has N/A success, not zero or perfect
success. Missing supported cases also count as failures. Aggregation balances
cases so a large environment batch cannot dominate a single-environment case.
Track eligibility requires complete mandatory coverage. Ineligible algorithms
remain visible with `eligible=false`; omitting difficult cases cannot improve
eligibility.

## 6. Domain populations and generalization

Packaged free-space v3 suites use Cartesian translation, rotation and combined
requests, near/mid/far motion ranges, and explicit homogeneous/heterogeneous
batches. Seeded start populations include measured joint margins and scaled
finite-difference Jacobian conditioning. A requested near-singular start is
screened from a bounded candidate pool and carries its measured classification.
The bounded independent IK/FK screen must satisfy stricter suite thresholds;
failed generation raises instead of reducing a candidate's denominator.
Single-target requests cannot claim a turn; pure rotation has no translation
path shape. Coverage profiles balance shape/count pairs across seeds. The
README owns configuration and the sampler owns exact ranges.

Success summaries bootstrap complete seed clusters at the configured confidence
level using 1000 deterministic draws, preserving case weights and supported
missing-outcome penalties. Environment rows and repeated calls are dependent
within a cluster. Fewer than two clusters have unavailable intervals. Unique
input fingerprints distinguish independent tasks from repeated throughput
requests. These intervals characterize the declared populations; they do not
verify training-data exclusion or justify strong claims from tiny seed counts.

Domains have versioned identities and declared `nominal`, `robustness` or
`held_out` kinds. Expansion preserves a shared `track_group`, applies overrides
to separate effective tracks and freezes the resulting cases. Undeclared
domains stay null; randomization alone does not imply a nominal baseline.

Spatial suites group skills by their physical contract. Isolated factors keep
primitive parameters, physics, robot and task semantics fixed. Object-only
skills are not labeled as target-shift evaluations. The shipped populations
include:

| Population | What changes |
| --- | --- |
| Object/target pose | Subject or Cartesian target translation/orientation; articulations may also vary root height and roll/pitch |
| Workspace/relative shells | Offset radius toward a configured workspace boundary, recorded as near/mid/far buckets |
| Task frame | Shared planar translation/yaw of the scene while preserving relative layout |
| Composed pose | Skill-appropriate subject, target, support and task-frame factors varied together |
| Primitive geometry | Cube `size` only: uniform scales and axis-elongated cuboids, with separate nominal populations |

Pose domains use deterministic stratification including interval bounds.
Primitive domains retain the logical object ID, physical parameters, skill
settings, seeds and nominal pose; YAML anchors keep runtime parameter overrides
and provenance identical. Primitive populations remain separate from pose
shifts. Dual-arm placement additionally varies support pose and rebuilds grasp
and target transforms per environment row. Exact bounds and sample counts live
in suite YAML rather than a second documentation copy.

Domain summaries report coverage, nominal success, non-nominal mean/worst
success, population variance and retention, plus spatial-bucket counts and
failure stages where present. Generalization eligibility requires a nominal
baseline and at least one non-nominal domain, all with complete measured
coverage within the same track and attributed algorithm. Retention is
worst-domain success divided by nominal success; it is N/A when the nominal
population is incomplete or has zero success. Unsupported populations follow
section 5's denominator rules.

## 7. Success semantics and failure taxonomy

Keep planning, geometry, solved timing, execution and physical stages separate.
Planner-internal stopping thresholds do not define benchmark success. External
motion thresholds are suite-owned; physical criteria are versioned in
`evaluation.py`. Both remain identical across compared algorithms. Sequence
success comes from one replayed episode, never a product of independent action
rates.

Failure records preserve the failed layer and reason. The vocabulary includes:

- Input/capability: `invalid_case`, `unsupported_capability`,
  `unsupported_capacity`, `runtime_unavailable`.
- Planning/evaluation: `checkpoint_load_failure`, `planner_exception`,
  `planner_reported_failure`, `timeout`, `scenario_prepare_error`,
  `metric_evaluation_error`.
- Motion: `non_finite_trajectory`, `waypoint_miss`, `joint_limit_violation`,
  `dynamic_limit_violation`, `controller_tracking_failure`.
- Physical task: `object_not_grasped`, `object_dropped`, `release_failure`,
  `task_goal_miss`, plus skill-stage-specific reasons.

A future qualified collision validator may add `self_collision` and
`environment_collision`; empty-world trajectory validity does not establish
those checks. Raw `stages` and `failure_stage` retain physical attribution even
when a separate motion or execution failure also exists.

## 8. Artifacts and report contract

`resolved_suite.yaml`, `environment.json`, `case_manifest.json` and
`trials.jsonl` preserve the configuration, runtime environment, frozen inputs
and raw lifecycle/outcome records. Numeric values stay numeric until Markdown
rendering. Case manifest schema `3` includes optional domain, track-group,
embodiment and capability metadata.

Report schema `3` uses explicit family views. `aggregate_results()` receives
`track_scenarios` from resolved configuration; friendly track/skill names do not
select the family. Each represented family has three tables:

1. **Time & Memory:** lifecycle and steady-state timing with the family's timed
   operation, execution timing where applicable, and memory measurements.
2. **Success & Other Metrics:** coverage, applicable success views, stage rates
   and continuous quality diagnostics.
3. **Leaderboard:** all evaluated algorithms within each track and ranking,
   including unsupported or ineligible rows. Geometry, timed-motion and
   physical-skill success are never combined into one rank.

`report_views` is the family index; canonical aggregate tables use
`motion_planning_*` and `atomic_skills_*` names. The additional
`motion_timing_leaderboard` exposes solved timing, and `generalization` contains
domain statistics. Markdown consumes these views directly, without an
unprefixed legacy leaderboard fallback. Raw trial records remain shared.

Unmeasured and inapplicable values render as N/A. Supported failures remain in
success denominators; continuous geometry metrics use the declared valid
population. Report success and coverage alongside conditional quality to keep
missing or failed cases visible. No algorithm disappears merely because it
could not run.

Suite populations/thresholds, case schema and report schema are versioned
independently. Combine results only under matching suite/protocol definitions;
latency comparisons additionally need matching hardware, device, dtype and
batch protocol. NMG CPU characterization is separate from cuRobo CUDA latency.
Old artifacts must not be relabeled as a new protocol.

## 9. Future extensions

These are not current benchmark capabilities:

- Qualified collision worlds and independent collision validation. A deployment
  comparison must expose information asymmetry when cuRobo receives obstacles
  and an NMG policy does not; equal-input obstacle comparison requires a policy
  that accepts the same constraints.
- Latency-budget/Pareto sweeps, per-process isolation and
  broader memory accounting. `max_planning_time` checked after a plan is not a
  preemptive deadline.
- Multimodal top-k feasibility, diversity and sampling cost; paired physics/APG
  refinement with both quality gain and added cost reported.
- Observation-driven recovery under moving targets, noise, disturbances and
  slip. FK rollout alone is not closed-loop recovery.
- General per-row antipodal grasp sampling and release-scale profiles.
  Multiple configured robots alone do not establish unseen-embodiment
  generalization.
