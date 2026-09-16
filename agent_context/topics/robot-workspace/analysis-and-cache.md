# Workspace analysis and cache details

[Topic overview](robot-workspace.md). Read this for the selected detailed flow.

### Offline analysis, cache, and runtime path

1. `WorkspaceAnalyzerConfig` chooses `JOINT_SPACE`, `CARTESIAN_SPACE`, or `PLANE_SAMPLING`,
   plus sampler seed/count/batch size, cache, constraints, visualization, metrics, IK
   seeds/reference pose, control part, and plane settings.
2. Analysis is designed for one simulation environment; with multiple environments it warns
   and uses environment zero. Control part resolution prefers an explicit name, then
   `left_arm`, `right_arm`, then the first available part, and finally the robot default
   solver.
3. Joint-space mode samples within selected joint limits and applies FK; stored
   `workspace_points` and `joint_configurations` contain only valid aligned rows.
4. Cartesian mode samples requested XYZ bounds, or infers bounds from 1,000 random FK samples
   plus margin/fallback using one batched FK call, then runs seeded IK. It stores all sampled points and reachability
   outputs plus best joint configurations aligned to reachable points.
5. Plane mode creates samples on the configured plane and follows the same IK/result alignment path.
6. `WorkspaceAnalyzer.analyze()` checks `ResultsCache` before computation unless force is set,
   computes metrics after analysis, then saves (`analyzer.py`).
7. Cache defaults to `~/.cache/embodichain_data/robot_workspace` in the CLI. A raw analyzer
   config with cache enabled but no `cache_dir` has no persistent result cache; only its
   legacy/in-memory sampling cache remains.
8. `ResultsCache` hashes canonical metadata into a readable key plus 12-hex digest. Key
   material includes analyzer version, robot identity, control part/joints/limits,
   mode/sampler/seed/batch settings, constraints, IK seeds/reference hash/plane, and
   robot/solver asset absolute paths with size and modification time.
9. A cache entry is complete only when both `results.npz` and `meta.json` exist. Load returns
   a miss for missing, unreadable, or invalid files.
10. `RobotWorkspace` requires nonempty `joint_configurations`. It selects `reachable_points`
    when aligned, otherwise aligned `workspace_points`; it accepts compatible success scores
    and validates positions as `N x 3` and qpos as `N x D`.
11. Runtime sampling is point-uniform or voxel-uniform, optionally score-weighted and with
    replacement. Voxel-uniform first chooses an occupied voxel, then a point inside it,
    reducing bias toward densely sampled areas.
12. `Robot.sample_reachable_pose()` resolves the named control part, samples cached qpos,
    computes batch FK against each environment's current robot base, and filters requested
    **local-arena-frame** position bounds. `WorkspaceSample.eef_pose` is in the local arena
    frame, not world coordinates. Unfilled rows are padded with identity pose/zero qpos, index
    `-1`, and `valid=False` (`runtime.py`; `robot.py`).

### Invariants to preserve

- Cached Cartesian points, joint configurations, and optional scores must stay row-aligned.
  Runtime deliberately drops a point/score source that cannot be aligned.
- Cache identity must change for solver/robot asset changes and every parameter that changes
  the sampled distribution or reachability test.
- Runtime qpos is the durable sampling representation. Local-arena-frame poses are recomputed
  through each environment's current robot/base transform; cached Cartesian points alone are
  insufficient for multi-environment placement.
- A cache describes kinematic reachability. It does not prove collision-free motion,
  trajectory feasibility, grasp validity, or task success.
- Ambiguous control-part selection must raise unless a default or exactly one workspace is
  available.
- Bound-filter sampling is attempt-limited and may return invalid padded rows; callers must
  consume the validity mask.
- `analyze-workspace` result cache and `workspace-cache` legacy session commands are different
  cache surfaces. The latter dispatches from `embodichain/cli/workspace_cache.py` to
  `workspace/caches/cache_utils.py` for session-style sampling cache entries.

### Common failures and recommended change sites

| Symptom | Likely cause / change site |
|---|---|
| Analysis uses the wrong arm | Set `control_part` explicitly; inspect analyzer fallback and `RobotCfg.workspace_cfg` keys. |
| Every run recomputes | CLI/analyzer has no persistent `cache_dir`, force is enabled, or key metadata changed. Preview the key and metadata. |
| Cache is present but treated as miss | One file is missing/corrupt, or asset path size/mtime changed. Check both entry files. |
| Runtime rejects cache alignment | Exported `joint_configurations` count differs from reachable/workspace point count; repair analyzer/export, not runtime validation. |
| Runtime cannot choose a workspace | Multiple control-part caches exist without an explicit name/default. |
| Requested bounded poses include invalid padding | Expected after exhausting attempts; filter with `valid`, never treat padded identity as reachable. |
| Sampled pose is reachable but motion collides | Workspace is kinematic only; add motion planning/collision validation at the consuming layer. |
| `workspace-cache list/clean` cannot see ResultsCache entries | It targets legacy session caches; use analyzer preview/cache directory semantics for result entries. |
| Concurrent analyzers expose corrupt/incomplete cache | `ResultsCache.save()` writes result then metadata directly with no observed lock or temp-file replacement. Add atomic write/lock coverage before relying on shared concurrent writers. |

### Manipulability scores

- When the metric config enables manipulability (default `ALL`), the analyzer
  computes per-configuration Yoshikawa scores `w = sqrt(det(J J^T))` from the
  active solver's Jacobian after analysis. `manipulability_scores` is
  row-aligned with `joint_configurations` (and therefore with reachable points
  in Cartesian/plane modes), stored in `results.npz`, and restored on cache
  hits; aggregates land in `metrics["manipulability"]`.
- `ManipulabilityMetric` produces statistics only from Jacobians or
  precomputed scores. The former centroid-distance placeholder was measured
  to be negatively correlated with true manipulability (corr ≈ −0.37 on
  Franka) and now yields a warning plus empty results instead.
- Cost is negligible: batched Jacobian + determinant is ~10 ms per 470
  configurations on GPU.

### Seed selection for Cartesian/plane IK

Cartesian and plane analysis verify reachability through the solver's
multi-start `get_ik`, so `PytorchSolverCfg.enable_seed_selection` applies
without analyzer changes. Measured on Franka (4000 identical points, warm):
enabling it at unchanged `num_samples=30` detected slightly more reachable
points at 3.3× lower wall time; `num_samples=8` reached 4.1× with <1%
detection loss. Analytic solvers (OPW/SRS/UR) are unaffected.

### Sampling and allocation controls

- `SamplingConfig` and `analyze-workspace --sampler` default to scrambled Sobol.
  Explicit random/grid selections remain available. Sobol draws continue a
  dimension-specific sequence; recreate the sampler to replay it. Samplers use
  private RNGs and do not reset the application's Torch random state.
- `sample_within_constraints=True` (`--sample-within-constraints`) refills random,
  Sobol or LHS proposals inside the permitted domain. Cartesian box/ground bounds
  are intersected before generation; sphere proposals use a volume-uniform
  transform; plane proposals are filtered after projection. Exclusion zones are
  respected. Exhausting `max_sampling_rounds` raises instead of returning an
  incomplete cache. Reachability is conditional on this permitted domain, so its
  percentage is not interchangeable with the default full-domain statistic.
- `retain_diagnostics=False` (`--compact-results`) retains reachable positions,
  aligned qpos and scores, while dropping all-target points and masks from the
  returned result and archive. Counts and metrics remain available. Runtime and
  preview accept both formats. This reduces retained/output storage; analysis
  still materializes the original sampled domain before compaction.
- IK batches allocate target poses once across point/seed axes. One seed bypasses
  seed-axis reductions and selection indexing. The analyzer never changes the
  process-wide logging level. Analyzer FK/IK explicitly selects environment zero
  and defaults to the robot device when no simulation manager is supplied.
- Cache keys include a sampling revision, compact/domain settings, geometry and
  excluded zones. These changes intentionally invalidate older result entries.
- Focused logic tests: `tests/sim/motion/workspace/test_analysis_batching.py`.
  Real analytic solver tests (including CUDA stream reuse) are in
  `tests/sim/motion/solvers/test_analytic_batching.py`. The renderer-free benchmark
  is `scripts/benchmark/workspace_analyzer/benchmark_robot_workspace.py`; it uses
  real asset chains and Robot methods with fixed root snapshots and eager FK.
