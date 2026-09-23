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
- Default matrix: cuRobo (`primary_baseline`); IK / TOPPRA optional diagnostics
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
  artifact schema, aggregation, and report stay unchanged.

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
- Dual-arm actions and multi-action chains
- Latency-budget Pareto sweeps, confidence intervals, subprocess isolation


## Atomic Skill contract foundation (#669)

The first migration layer provides shared contracts while preserving existing
skill scoring. It does not yet implement official embodiment component loading,
new physical success criteria, segment/contact sampling, domain generation, or
domain-level aggregation. The three existing report tables remain unchanged.

### Domain provenance

Add `domain` alongside `scenario` in an existing track. This YAML fragment
shows the new fields; retain the track's existing `config`:

```yaml
tracks:
  - id: atomic-nominal
    scenario: atomic_task
    domain:
      id: nominal
      version: v1
      kind: nominal
```

`domain` accepts exactly `id`, `version`, and `kind`. The kind is `nominal`,
`robustness`, or `held_out`; IDs and versions must be nonempty strings. This
field declares provenance only: it does not perturb scenes, select held-out
objects, or establish training/evaluation separation. Use separate tracks for
separate populations until domain generators are added. Existing suites without
this declaration retain `domain: null`; randomized suites are not silently
classified as nominal.

The runner snapshots this identity onto generated cases and carries it into all
associated lifecycle/trial records. `case_manifest.json` now uses
`case_schema_version: 3` and contains the optional `domain` on each case. Each
track/batch population is written before any planner evaluates that population;
later populations extend the manifest. Planner configuration hashes stay in
trial records, separate from planner-independent case content.

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
contains the existing measured replay summary. Missing optional measurements
mean unavailable evidence; the follow-up evaluation work must extend observation
collection before claiming grasp/release/retention or intermediate-stage success.

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
stages. Stage/domain aggregation and reporting are separate follow-up work.

### Follow-up ownership

- **Embodiment and case integration:** build against `AtomicSkillCaseProvider`,
  reuse official component loading, and keep case generation independent of
  planner outcomes. Do not duplicate the runner lifecycle or artifact writers.
- **Physical evaluation:** extend `ExecutionObservation` with measured segment
  and contact evidence, implement evaluators and counterexample tests, then opt
  providers into `physical_evaluator()` individually.
- **Framework and statistics:** add versioned domain generators and aggregation
  after agreeing on capability eligibility, stage denominators, and retention.

Contract tests can run without the simulator engine:

```bash
python -m pytest -q tests/benchmark/motion_generation/test_contracts.py
```

Runner/evaluator integration and existing benchmark regressions use the normal
EmbodiChain environment:

```bash
python -m pytest -q tests/benchmark/motion_generation
```
