# Motion Generation Benchmark

Free-space motion-generation suite with cuRobo as the default primary baseline.

Design background and roadmap: see [`BENCHMARK_DESIGN.md`](./BENCHMARK_DESIGN.md).

## Run

```bash
embodichain benchmark motion-generation --suite smoke
embodichain benchmark motion-generation --suite coverage
embodichain benchmark motion-generation --extra-baselines ik_interpolate toppra
embodichain benchmark motion-generation --path-shapes direct l_turn --start-state-bins nominal near_singularity
```

Artifacts land under `outputs/benchmarks/motion_generation/<timestamp>/`
(`resolved_suite.yaml`, `case_manifest.json`, `trials.jsonl`, `aggregates.json`,
`report.md` with exactly three tables).

## Implemented

- Extensible planner/scenario registries and track-based suite YAML
- `free-space-common` track with fixed manifests and start-state bins
- Default matrix: cuRobo (`primary_baseline`); IK / TOPPRA optional diagnostics
- NMG adapter stub (`candidate`, disabled until a checkpoint is ready)
- Lifecycle timing: construct / prepare / cold / warm
- Ordered waypoint matching and external `motion_valid` (separate from
  `PlanResult.success`)
- One Markdown report: Time & Memory, Success & Other Metrics, Leaderboard

## Not implemented yet

- Real NMG checkpoint adapter
- `collision-deployment` and `atomic-task` tracks
- Physics execution / task-success metrics
- Latency-budget Pareto sweeps, confidence intervals, subprocess isolation
