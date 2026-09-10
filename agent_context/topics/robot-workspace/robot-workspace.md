# Robot workspace

Workspace analysis caches kinematic reachability; runtime sampling reuses cached
joint configurations with the current robot base. This does not establish
collision-free motion, grasp validity or task success.

## Entry points

Paths below are relative to `embodichain/lab/sim/` unless qualified.

| Request | Owner |
|---|---|
| Offline analysis | `workspace/analyzer.py`: `WorkspaceAnalyzer`, `WorkspaceAnalyzerConfig` |
| Result-cache identity and persistence | `workspace/caches/results_cache.py`: `ResultsCache` |
| Cache loading / point or voxel sampling | `workspace/runtime.py`: `RobotWorkspace`, `WorkspaceSample` |
| Runtime binding config | `workspace/cfg.py`: `RobotWorkspaceCfg`; `cfg.py`: `RobotCfg.workspace_cfg` |
| Robot-facing sampling and FK | `objects/robot.py`: `sample_reachable_pose()` |
| Analyze CLI | `embodichain/lab/scripts/analyze_workspace.py` |
| Legacy session-cache CLI | `embodichain/cli/workspace_cache.py` → `workspace/caches/cache_utils.py` |

## Resolution and frames

Offline: robot preset/asset → control part and joint limits → joint/Cartesian/
plane sampling → FK/IK → metrics → result cache. Runtime: `workspace_cfg`
→ lazy per-part cache load → cached qpos selection → current-base batch FK
→ bounded-position filtering.

`Robot.get_workspace()` loads `RobotWorkspace.from_cache(cache_path, ...)` on
first use. `cache_path` may be a cache-entry directory (selecting `results.npz`)
or a direct archive path. Runtime loading uses `allow_pickle=False`; its
`meta.json` is optional, unlike a complete analyzer result-cache entry.

`WorkspaceSample.eef_pose` is in the **local arena frame**, shape `(B, K, 4, 4)`;
`qpos` is `(B, K, D)`, with `(B, K)` indices and validity mask. Invalid entries
have index `-1`, identity pose/zero qpos padding and `valid=False`. Consumers must
apply the validity mask; attempt-limited filtering may return fewer valid samples.

The solver's chain-root frame and the robot's local-arena FK frame are different
boundaries. Recompute poses through `Robot` for the target environments instead
of treating cached Cartesian points as current world poses.

## Cache and sampling invariants

- Analysis results use `results.npz` and `meta.json` together under a metadata-derived
  cache key. Missing/corrupt entries are cache misses.
- Identity includes robot/solver assets and parameters affecting sampling or IK;
  preserve invalidation when adding configuration fields.
- The CLI defaults result storage to `~/.cache/embodichain_data/robot_workspace`.
  An analyzer config without a persistent cache directory need not store results.
- Runtime positions `(N, 3)`, qpos `(N, D)` and optional scores must align.
  Point-uniform and voxel-uniform sampling define different distributions.
- Multi-part runtime binding requires an explicit name/default unless only one
  workspace is available. Set the control part explicitly for reproducible analysis.
- `workspace-cache` manages legacy session caches; it is not a browser for
  analyzer `ResultsCache` entries.

Read [analysis and cache details](analysis-and-cache.md) for analysis modes,
cache-key inputs, runtime selection and cache failure diagnosis.

## Change sites and validation

| Change | Tests |
|---|---|
| Sampling/FK/IK analysis | `tests/sim/motion/workspace/test_analyzer.py` |
| Cache identity and CLI cache behavior | `tests/sim/motion/workspace/test_cache.py` |
| Runtime alignment, base pose, bounds and invalid padding | `tests/sim/motion/workspace/test_runtime.py` |
| Workspace-aware event sampling | `tests/gym/envs/managers/test_workspace_randomization.py` |

Use [robot-system](../robot-system/robot-system.md) for robot config/kinematic
wiring and [motion-planning](../motion-planning/motion-planning.md) for collision
and trajectory feasibility. An empty/misaligned cache requires repairing the
producer or binding rather than relaxing runtime validation.
