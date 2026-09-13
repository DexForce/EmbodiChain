# Environment profiling

Read this when the request needs these details. [Topic overview](env-framework.md).

## Profiling

`BaseEnv` carries an `EnvProfiler` (`self._profiler`) that records per-section
**wall time** of the reset/step pipeline. It is a no-op (zero overhead) when
disabled, so the instrumentation stays in place unconditionally.

.. note::
   Only wall time is profiled. GPU-memory profiling has been temporarily
   removed (entry parameter, recording, and report output).

### Config

`EnvCfg.profiler: EnvProfilerCfg | None = None` (None = off). Fields:

| Field | Default | Purpose |
|---|---|---|
| `enable_time` | `True` | Per-section mean/min/max/std wall time |
| `sync_cuda` | `False` | `torch.cuda.synchronize()` at section boundaries for accurate GPU time (higher overhead) |
| `warmup_steps` | `5` | Discard first N top-level step/reset samples |
| `nvtx` | `False` | Also push NVTX ranges (named in `nsys` timelines) |
| `output_path` | `None` | Dump JSON report to this path on `report()` |

CLI: `run-env --profile [--profile_output prof.json]`. In code:
`cfg.profiler = EnvProfilerCfg(enable_time=True, ...)`.

### Instrumented sections

Section names are hierarchical (built from the active call stack); a parent's
time includes its children.

- **step** → `preprocess_action`, `step_action`, `sim_update`, `update_sim_state`
  (`event_interval`), `get_obs` (`proprio`, `sensor` (`render_camera_group`,
  `sensor_fetch`), `extend` (`obs_compute`)), `get_info`, `reward`
  (`reward_compute`), `postprocess_action`, `hook_after` (`rollout_write`,
  `trajectory_write`), `auto_reset`
- **reset** → `is_task_success`, `reset_objects_state`, `initialize_episode`
  (`dataset_save`, `record_camera_save`, `trajectory_save`, `event_reset`,
  `obs_reset`, `reward_reset`, `dataset_reset`), `get_obs`

`reset()` called during step's auto-reset does **not** open a duplicate root;
its children attribute to `step.auto_reset.*`.

### Per-functor breakdown

Every registered **event** and **observation** functor is additionally timed by
name via `ManagerBase._call_functor` (centralized in the base manager; the
event/obs `apply`/`compute` loops call through it). Each functor's section is
its config attribute name, nesting under the manager call site:

- `step.update_sim_state.event_interval.<functor>` (e.g. `record_camera`)
- `reset.initialize_episode.event_reset.<functor>` (e.g. `init_bottle_pose`)
- `step.get_obs.extend.obs_compute.<functor>` (e.g. `norm_robot_eef_joint`)

`calls` reflects the firing count -- interval event functors fire every
`interval_step` (so a `interval_step=10` functor shows `calls = num_steps / 10`).
The same obs functor appears under both `step.get_obs.*` and `reset.get_obs.*`.
Zero overhead when profiling is disabled. Reward/action/dataset functors are not
yet wired through the helper.

### Report

`env._profiler.report()` prints a tree (calls / mean / min / max / std / total /
`%par` of parent, sorted by total within each parent; `(other)` = parent total
minus measured children). It is also flushed automatically in `close()` **before
`sim.destroy()`** (which exits the process). `%par` is relative to the immediate
parent; the first `warmup_steps` samples are discarded.

---

## Startup information table

`BaseEnv._setup_scene()` constructs `SimulationManager` with
`defer_startup_summary=True`, preserving the requested headless state while the
scene is assembled. `_log_initialization_summary()` reuses the shared
`sim/_startup_summary.py` simulation and scene rows, then adds seed, control
timing, episode limit, robot identity, metadata, and manager counts.
`EmbodiedEnv` keeps its later initialization-complete boundary so all managers
are available. Each environment emits the tables once and consumes the
simulation-owned startup and scene snapshots.

`SimulationManagerCfg.startup_summary` accepts `compact`, `full`, and `off`.
Compact and full keep manager status/counts in the main table and append a
separate **Functor Details** table in configured execution order. Full also
shows qualified callable paths and parameters; large containers, tensors, and
arbitrary objects get bounded descriptions without evaluating functors or
transferring tensor data. `off` disables both tables. Gym JSON/YAML accepts
top-level `startup_summary` and `dexsim_startup_info` and forwards them through
`config_to_cfg()`.
