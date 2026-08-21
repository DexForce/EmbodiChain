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
  --suite atomic_franka_pgi_curobo --device cuda --record-video
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
- Initial Atomic Task slice: Franka + PGI, cuRobo, `MoveEndEffector`, and
  antipodal-grasp `PickUp` on a declarative cube
- Default matrix: cuRobo (`primary_baseline`); IK / TOPPRA optional diagnostics
- NMG adapter stub (`candidate`, disabled until a checkpoint is ready)
- Lifecycle timing: construct / prepare / cold / warm
- Distinct planning, motion-valid, execution, and physical task-success stages
- Planning latency, execution wall time, end-to-end time, nominal trajectory
  duration, simulated task-completion time, controller tracking RMSE, and
  task-specific object lift
- One Markdown report: Time & Memory, Success & Other Metrics, Leaderboard
- Optional Atomic Task headless replay videos after measured evaluation
  (`--record-video`, `--record-failed-video`)

## Extend

- Planner: register a `PlannerAdapter`, expose its `MotionGenerator`, and
  declare the `atomic_action` capability.
- Robot: register a `RobotProvider` and select it under `robot`; PickUp suites
  also declare the gripper control part and open/grasp qpos under `gripper`.
- Object: add another `objects` entry for built-in `cube`/`mesh`, or register a
  new object-kind factory.
- Atomic skill: implement and register an `AtomicSkillCaseProvider`; the runner,
  artifact schema, aggregation, and report stay unchanged.

## Current limits

- Real NMG checkpoint adapter
- `collision-deployment` and obstacle-aware common-input tracks
- Atomic Task execution is currently `B=1`; the supplied suite covers only
  Franka + PGI and cuRobo
- Remaining skills: `MoveHeldObject`, `Place`, `Press`, and action chains
- Latency-budget Pareto sweeps, confidence intervals, subprocess isolation
