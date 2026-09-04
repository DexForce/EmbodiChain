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
- Physical `task_success` for those contact skills requires the configured
  peak signed target-joint displacement from the effect segment through replay
  hold; this also
  accepts a spring-loaded button that rebounds before the final hold state
- Deterministic 16-seed generalization sweep for the original six-skill subset,
  with bounded robot-start, target, held-object, and object-pose randomization
- Default matrix: cuRobo (`primary_baseline`); IK / TOPPRA optional diagnostics
- Direct, batched NMG ONNX adapter (`candidate`, enabled when a model path is supplied)
- Lifecycle timing: construct / prepare / cold / warm
- Distinct planning, motion-valid, execution, and physical task-success stages
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
- Atomic Task execution is currently `B=1`; the supplied suite covers only
  Franka + PGI and cuRobo
- The v3 suite's Microwave and Drawer are physical simulation articulations,
  but cuRobo still receives an empty external collision world. Contact replay
  validates target-joint actuation; it does not guarantee collision safety
  against non-target appliance links or other environment geometry
- Dual-arm actions and multi-action chains
- Latency-budget Pareto sweeps, confidence intervals, subprocess isolation
