# Motion Planning

## Entry Points

| What | Path |
|---|---|
| Planner registry | `embodichain/lab/sim/planners/__init__.py` |
| Base planner class & config | `embodichain/lab/sim/planners/base_planner.py` → `BasePlanner`, `BasePlannerCfg`, `CollisionWorldInfo`, `PlanOptions`, `validate_plan_options` |
| TOPPRA planner | `embodichain/lab/sim/planners/toppra_planner.py` → `ToppraPlanner`, `ToppraPlannerCfg`, `ToppraPlanOptions` |
| Trapezoidal planner | `embodichain/lab/sim/planners/trapezoidal_planner.py` → `TrapezoidalPlanner`, `TrapezoidalPlannerCfg`, `TrapezoidalPlanOptions` |
| Neural planner | `embodichain/lab/sim/planners/neural_planner.py` → `NeuralPlanner`, `NeuralPlannerCfg`, `NeuralPlanOptions` |
| cuRobo planner | `embodichain/lab/sim/planners/curobo/curobo_planner.py` → `CuroboPlanner`, `CuroboPlannerCfg`, `CuroboWorldCfg`, `CuroboPlanOptions` |
| Planner assets | `embodichain/data/assets/planner_assets.py` → `download_neural_planner_checkpoint()` |
| Motion generator | `embodichain/lab/sim/planners/motion_generator.py` → `MotionGenerator`, `MotionGenCfg`, `MotionGenOptions` |
| Planner utilities & data types | `embodichain/lab/sim/planners/utils.py` → `PlanState`, `PlanResult`, `MoveType`, `MovePart`, `TrajectorySampleMethod`, `interpolate_xpos_batched` |

## Overview

The planning stack has two layers:
1. **BasePlanner** — low-level trajectory planner that takes a list of `PlanState` waypoints and produces a `PlanResult` with joint trajectories.
2. **MotionGenerator** — the single stateful planning facade that composes a
   planner with strategy selection, interpolation, IK resolution, result
   normalization, and multi-part coordination.

All planners resolve their robot at init via `SimulationManager.get_instance().get_robot(cfg.robot_uid)`.

The entire stack is **env-batched** (`B = num_envs`). `PlanState` / `PlanResult` tensors carry a leading `B` dimension; `BasePlanner.plan()` and `MotionGenerator.generate()` operate on `B` environments in one call.

## Planner Hierarchy

```
BasePlanner (ABC)
  ├─ ToppraPlanner        Time-optimal path parameterization (fork-pool fan-out)
  ├─ TrapezoidalPlanner   Batched trapezoidal or jerk-limited Double-S timing
  ├─ NeuralPlanner (experimental)   APG waypoint rollout (native batching)
  └─ CuroboPlanner        CUDA collision-aware planning (native batching)

MotionGenerator           Wraps any BasePlanner; adds interpolation and multi-part support
```

Config hierarchy:
```
BasePlannerCfg            robot_uid (MISSING), planner_type
  ├─ ToppraPlannerCfg     planner_type = "toppra", max_workers, mp_context
  ├─ TrapezoidalPlannerCfg planner_type = "trapezoidal"
  └─ NeuralPlannerCfg     planner_type = "neural", checkpoint_path (MISSING)

MotionGenCfg              planner_cfg (MISSING — must be a BasePlannerCfg subclass)

PlanOptions               (empty base)
  ├─ ToppraPlanOptions    constraints, sample_method, sample_interval
  ├─ TrapezoidalPlanOptions profile, constraints, sample_method, sample_interval
  └─ NeuralPlanOptions    control_part, start_qpos, max_steps

MotionGenOptions          strategy, sample_count, velocity/acceleration limits,
                          start_qpos (B, DOF), control_part, plan_opts,
                          is_interpolate, interpolate_nums, is_linear,
                          interpolate_position_step, interpolate_angle_step
```

## Available Planners

### ToppraPlanner

Time-optimal path parameterization using the [toppra](https://github.com/hungpham2511/toppra) library.

- **Dependency**: `pip install toppra==0.6.3` (import-time error if missing).
- **Batched**: accepts `target_states` whose tensor fields carry leading batch dim `B`. Internally fans out `B` independent single-env TOPPRA solves across a `ProcessPoolExecutor`.
- **Method**: `plan(target_states, options=ToppraPlanOptions()) -> PlanResult`

`ToppraPlannerCfg` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `max_workers` | `int \| None` | `None` | Worker process count. `None` → `min(cpu_count() // 2, B)`. |
| `mp_context` | `str \| None` | `None` | Multiprocessing start method. `None` auto-selects `fork` on CPU and `spawn` on GPU; can be set to `"fork"` or `"spawn"`. |

`ToppraPlanOptions` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `constraints` | `dict` | `{"velocity": 0.2, "acceleration": 0.5}` | Per-joint or scalar limits |
| `sample_method` | `TrajectorySampleMethod` | `QUANTITY` | `TIME`, `QUANTITY`, or `DISTANCE` |
| `sample_interval` | `float \| int` | `0.01` | Time interval (seconds) or sample count depending on method |

Worker details:
- The pure-numpy module-level worker `_toppra_solve_one_env` is picklable and never touches CUDA/Warp/sim state.
- `B == 1` or `max_workers == 1` uses an inline fallback (no IPC).
- `TIME` sampling can produce per-env waypoint counts; shorter trajectories are tail-padded by repeating the final waypoint and `duration` records the real endpoint per env.
- Per-env failures set `success[b] = False` and fill the env's trajectory with its start qpos; other envs continue. `BrokenProcessPool` tears the pool down and rebuilds it on the next call.

### TrapezoidalPlanner

The trapezoidal planner is a dependency-free, batched Torch backend for
piecewise-linear joint paths. Each input waypoint is a rest point. The default
profile is acceleration-limited trapezoidal timing (with triangular fallback
for short moves); ``profile="double_s"`` selects the rest-to-rest linear-path
subset of HolisticMotion's seven-phase Double-S time law. It preserves that
implementation's discrete ``amax *= 0.9`` feasibility search for moves without
a cruise phase and its 1% ``EnforceJointLimits`` margin. Scalar or per-joint
velocity, acceleration, and jerk limits are projected onto each linear path
segment. Golden tests cover durations and sampled position, velocity, and
acceleration against the HolisticMotion Python binding. It supports fixed
quantity and approximate fixed-time sampling and returns explicit ``dt``.
``minimum_duration`` applies one uniform per-environment time scale, preserving
the path while reducing velocity, acceleration, and jerk. The minimal
``scripts/tutorials/sim/trapezoidal_profile.py`` example plots scalar position,
velocity, acceleration, and jerk without starting simulation. A runnable
batched robot example lives in ``scripts/tutorials/sim/trapezoidal_planner.py``.
The tutorial names the two diagnostics ``velocity_trapezoidal`` (the
``trapezoidal`` backend profile) and ``acceleration_trapezoidal`` (the
jerk-limited ``double_s`` backend profile). Its default ``--profile both`` run
writes a separate PNG for each profile. Every figure contains EEF XYZ, EEF XYZ
Euler orientation, joint position, joint velocity, and joint acceleration for
the selected ``--plot-env`` batch row. Figures are shown interactively by
default and are saved only when ``--plot-output`` is supplied; use
``--no-show-plot`` for headless runs. FK is evaluated in one call across all
trajectory samples for only the selected row rather than repeatedly per sample
or across the full environment batch, and plotted Euler angles are
unwrapped across ±π to suppress representation-only discontinuities. The demo
uses a stable writable Matplotlib cache under ``/tmp`` for headless execution.
Each result uses a compact 3-by-2 dashboard: a 3D FK path with start, goal, and
endpoint-line reference plus XYZ, RPY, joint position, joint velocity, and
joint acceleration time plots. Its header reports duration, derivative peaks,
and line deviation. The 3D path uses equal physical scaling on X, Y, and Z so
small cross-axis IK errors are not visually magnified, while the XYZ time plot
overlays dashed desired Cartesian coordinates against the FK result.
Each trajectory tutorial keeps a small local CJK-capable font fallback and
readable title/legend defaults; ASCII minus rendering avoids missing-glyph
boxes.
Multi-path/profile runs finish planning before showing all figures together.
``--path joint`` generates synchronized, limit-clamped motion on every arm
joint. ``--path cartesian`` is Cartesian-first rather than joint-first: the
trapezoidal backend time-parameterizes metric line arclength ``s(t)`` under
``--cartesian-velocity``, ``--cartesian-acceleration``, and
``--cartesian-jerk``; every resulting sample becomes an exact
fixed-orientation point on the line before continuous-seed IK. The joint
trajectory is never resampled afterward, so its desired EEF path retains the
planned Cartesian geometry and time law. ``--path both`` runs both diagnostics
and reports the scalar Cartesian derivative peaks plus maximum FK line error.
``--cartesian-distance`` controls line length and ``--cartesian-step`` sets a
minimum IK sample density. For a six-axis arm the tutorial uses the same
non-singular seed as the TOPPRA demo before solving the default 0.10 m downward
line. Joint derivatives used by the diagnostic and replay come from
second-order differential kinematics. The solver Jacobian gives ``dq/ds`` for
the fixed-orientation Cartesian tangent, its path derivative gives
``d²q/ds²``, and the Double-S outputs are composed by the chain rule as
``dq/dt = dq/ds * ds/dt`` and
``d²q/dt² = d²q/ds² * (ds/dt)² + dq/ds * d²s/dt²``. The implementation does
not numerically differentiate sampled IK positions in time. Cartesian derivative constraints
apply to ``s(t)`` and do not imply identical joint-space jerk bounds after the
nonlinear IK mapping. No display filtering is applied. Before derivative
evaluation,
each analytic IK sample is replaced only by the joint-limit-valid ``2π``
equivalent nearest to the previous seed, removing representation wrap jumps
without changing the physical configuration or Cartesian path.
For OPW robots the Cartesian tutorial submits the complete pose path through
``Robot.compute_ik_path``. OPW evaluates all pose candidates in one Warp launch
and performs temporally ordered branch selection in a second launch with one
thread per environment; other solver types fail explicitly rather than being
silently treated as continuous path solvers.
Interactive replay consumes ``PlanResult.dt`` rather than submitting all
samples as fast as Python can loop: every command advances enough physics steps
for its scaled interval and windowed runs are wall-clock paced. The
``--replay-speed`` multiplier controls playback speed; headless runs retain
unthrottled wall-clock execution while preserving physics-step timing. Before
each run, both current joint state and drive target are reset to the planned
start pose.
By default every input waypoint remains a rest point. Set
``stop_at_waypoints=False`` to remove duplicate and straight, same-direction
interior points before timing; genuine direction changes remain explicit rest
points, and batch rows with fewer retained points are final-pose padded.
``backend="auto"`` uses Warp profile construction and sampling for CUDA float32
inputs and the Torch reference path otherwise. ``backend="warp"`` also permits
explicit CPU Warp execution but requires float32. Path compression and limit
projection remain shared Torch tensor operations. Warp then constructs each
trapezoidal or Double-S scalar segment profile in parallel, including the
Double-S no-cruise acceleration reduction and phase integration. Shared
post-processing applies minimum-duration scaling and the Double-S duration
margin consistently across backends.
Torch uses batched ``searchsorted`` for segment lookup without materializing a
``(B, N, segments)`` comparison tensor. Warp uses binary segment search once
per ``(B, N)`` sample, then a separate ``(B, N, DOF)`` composition kernel so
joint dimensions do not repeat profile lookup work.
Torch phase lookup also uses ``searchsorted`` and expands only selected phase
durations. Position, velocity, acceleration, and jerk coefficients are gathered
directly by ``(batch, segment, phase)``, avoiding four additional
``(B, N, phases)`` temporary tensors.
An all-stationary batch takes a dedicated hold fast path: it generates only
sample times and zero-derivative outputs, skipping constraint projection,
profile construction, segment lookup, and Warp dispatch.
The reproducible microbenchmark at
``scripts/benchmark/motion_generation/trapezoidal_planner.py`` measures Torch
and Warp time, CPU/GPU memory, endpoint success, and cross-backend error, then
writes the required three-table Markdown report under ``outputs/benchmarks``.

### NeuralPlanner (experimental)

Learning-based EEF waypoint planner. Franka Panda only.

- Checkpoint: `download_neural_planner_checkpoint()` from HuggingFace (gated, needs `HF_TOKEN`)
- Use via `MotionGenerator` with `planner_type="neural"` and `plan_opts=NeuralPlanOptions(...)`
- Input: `EEF_MOVE` `PlanState` list with batched `xpos:(B, 4, 4)`
- Key cfg: `checkpoint_path` (from download), `control_part`
- Natively batched: transformer forward, reach checks, and convergence holds all operate on `(B, ...)`.

### CuroboPlanner collision worlds

`CuroboWorldCfg.rigid_objects` accepts either a mapping or a sequence. Use
`Mapping[registry_id, RigidObject]` for a registry-backed integration. The
mapping key is the authoritative logical/source obstacle ID used by the
content-cache key, `collision_world_entity_ids`, and registry validation. For
`cuboid` and `mesh`, it is also the physical YAML obstacle name and dynamic
update key. For `sphere`, one static source expands to physical YAML names such
as `registry_id_0`; dynamic sphere configuration is rejected, while cache and
full-world identity remain keyed by `registry_id`. A registry mapping whose
source lacks mesh geometry required by the selected representation fails fast
instead of silently dropping the source. The sequence form is an advanced
direct-core path that derives names from each object's `uid` or an
`obstacle_<index>` fallback.

`CuroboWorldCfg.multi_env` controls collision-world batching, not whether robot
states or goals are batched:

- `multi_env=False` (default) shares one world. Use it when obstacle poses are
  equal after each simulator-world pose is rebased into its environment's robot
  base. Replicated arenas may have different world-frame offsets and still
  safely share a world when their robot-relative layouts are identical.
- `multi_env=True` allocates one world per batch row. Use it when obstacles have
  different poses relative to their local robot bases, such as per-env pose
  randomization.

The multi-env scene is cloned from the YAML generated using env 0; enabling the
flag does not load distinct initial simulator poses for other rows. Per-env
differences require `"cuboid"` or `"mesh"` representation, registration in
`dynamic_obstacle_names`, and current `(B, 4, 4)` world poses in
`CuroboPlanOptions.dynamic_obstacle_poses`. Independent worlds replicate scene
data and collision caches, so retain the shared default for identical rebased
layouts.

`BasePlanner.collision_world_info` and
`with_collision_world(options, obstacle_poses=...)` form the generic per-plan
dynamic-world bridge. The base property returns `None` and the base hook leaves
options unchanged. `CuroboPlanner` returns an immutable `CollisionWorldInfo`
with updates enabled, clones the supplied pose tensors, and merges them into
`CuroboPlanOptions.dynamic_obstacle_poses`.
`MotionGenerator.supports_dynamic_collision_world` exposes the capability and
`MotionGenerator.bind_collision_world()` owns option copying before forwarding
to the backend hook. Atomic actions use that facade from their framework-owned
`plan()` template when a `SceneSnapshot` declares collision entities;
individual skills must not construct backend obstacle options themselves.
`CollisionWorldInfo` carries the complete canonical world, its dynamic subset,
the `"shared"` / `"per_env"` mode, and update capability as one validated
contract. It requires unique canonical IDs and requires the dynamic subset to
belong to the complete world. `MotionGenerator.collision_world_info` forwards
that contract and retains derived ID/mode properties for callers. For cuRobo,
the complete set is every mapping key (or inferred sequence name), while the
dynamic set is exactly `CuroboWorldCfg.dynamic_obstacle_names`. Sphere-expanded
physical YAML names are not part of either logical ID declaration.
`CuroboWorldCfg` rejects duplicate obstacle names and requires every
`dynamic_obstacle_name` to match an object registered in `rigid_objects`, so a
planner-local mismatch fails before backend construction.

For the canonical path, pass `SceneRegistry.collision_geometry_by_id()` into
`CuroboWorldCfg.rigid_objects`, derive dynamic names from the registry, and call
`SceneRegistry.make_planning_scene_provider(motion_generator, batch_size=...)`
before execution. The geometry mapping excludes `NONE` registrations. The
factory first requires the registry's complete `STATIC ∪ DYNAMIC` set to
equal `MotionGenerator.collision_world_entity_ids`, then requires exact
registry/derived-provider/planner dynamic-subset agreement. It also checks
update capability for a non-empty dynamic set and the same collision-world
batch mode. An external perception/hardware provider instead uses
`validate_collision_integration(..., scene_provider=provider)`.

One environment may infer `SHARED`; a multi-environment registry with dynamic
entities must explicitly choose `SHARED` or `PER_ENV`. Alias normalization
happens before planner construction, so planner IDs must never be simulator
UIDs unless that string is also the chosen canonical registry ID.

`MotionGenerator.resolve_plan_options()` is the corresponding option-ownership
boundary. It copies caller-supplied typed options, otherwise obtains backend
defaults; for TOPPRA it maps the requested sample count and generic
velocity/acceleration limits into `ToppraPlanOptions`. Atomic actions do not
import or branch on concrete planner option types.

### MotionGenerator

Unified interface for trajectory planning with optional pre-interpolation.

- Wraps a `BasePlanner` instance (resolved from `planner_cfg.planner_type`).
- Supported planner types: TOPPRA, NeuralPlanner, and cuRobo.
- `MotionGenCfg.planner_cfg` is **MISSING** — must be provided.
- `generate()` and `interpolate_trajectory()` are env-batched (`B, N, DOF`).
- `generate()` always returns a normalized `PlanResult`; failed rows hold the
  supplied `start_qpos`, and every returned trajectory has explicit `dt` and
  a `duration` derived from it.

Grasp-pose generation is a sibling planning service, not a
`MotionGenerator` feature. `embodichain.toolkits.graspkit` owns its standalone
contract and the `pg_grasp` implementation without importing `embodichain.lab`.
Atomic actions and Expert Program install a generator instance in
`AtomicActionEngine`. The three Expert Program references declare their scenes,
robot profiles, trajectory policies, and grasp-generator parameters in Gym
JSON. `configured_runtime.py` decodes the shared composition schema and lazily
creates the production simulation adapter; the config loader registers the
existing `EmbodiedEnv` under the config-selected ID. There are no task-specific
environment subclasses or direct-planning paths.

`MotionGenOptions` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `strategy` | `"motion_gen" \| "ik_interp"` | `"motion_gen"` | Use the configured backend or deterministic waypoint IK/joint interpolation |
| `sample_count` | `int \| None` | `None` | Requested normalized output count; backend default when omitted |
| `velocity_limit` | `float \| None` | `None` | Optional backend-neutral velocity limit |
| `acceleration_limit` | `float \| None` | `None` | Optional backend-neutral acceleration limit |
| `start_qpos` | `torch.Tensor \| None` | `None` | Optional backend context, shape `(B, DOF)`; required by `strategy="ik_interp"` |
| `control_part` | `str \| None` | `None` | Robot control part name (must match `RobotCfg.control_parts` key) |
| `plan_opts` | `PlanOptions \| None` | `None` | Passed to the underlying planner |
| `is_interpolate` | `bool` | `False` | Pre-interpolate waypoints before planning |
| `interpolation_dt` | `float \| None` | `None` | Required explicit waypoint interval for `strategy="ik_interp"` and automatic joint interpolation fallback |
| `interpolate_nums` | `int \| list[int]` | `10` | Points per segment (scalar or per-segment list) |
| `is_linear` | `bool` | `False` | `True` = Cartesian linear interpolation; `False` = joint-space |
| `interpolate_position_step` | `float` | `0.002` | Cartesian step size (meters) or joint step size (radians) |
| `interpolate_angle_step` | `float` | `π/90` | Angular step in joint space (radians); only if `is_linear=False` |

## Planner Interface

### PlanState (input)

Describes one waypoint or action. Tensor fields carry a leading batch dim `B`; enum/scalar fields are shared across `B`.

| Field | Type | Notes |
|---|---|---|
| `move_type` | `MoveType` | `TOOL`, `EEF_MOVE`, `JOINT_MOVE`, `SYNC`, `PAUSE` |
| `move_part` | `MovePart` | `LEFT`, `RIGHT`, `BOTH`, `TORSO`, `ALL` |
| `xpos` | `torch.Tensor \| None` | Target TCP pose `(B, 4, 4)` for `EEF_MOVE` |
| `qpos` | `torch.Tensor \| None` | Target joint angles `(B, DOF)` for `JOINT_MOVE` |
| `qvel` / `qacc` | `torch.Tensor \| None` | Target joint velocities / accelerations `(B, DOF)` |
| `is_open` | `bool` | Tool open/close (for `TOOL`) |
| `is_world_coordinate` | `bool` | `True` = world frame; `False` = relative |
| `pause_seconds` | `float` | Duration for `PAUSE` move type |

Convenience constructors:
- `PlanState.from_qpos(qpos:(B,DOF), move_type=JOINT_MOVE, ...) -> PlanState`
- `PlanState.from_xpos(xpos:(B,4,4), move_type=EEF_MOVE, ...) -> PlanState`
- `PlanState.single(qpos=(DOF,)\|None, xpos=(4,4)\|None, ...) -> PlanState` — unsqueezes single-env tensors to `B=1` (idempotent on already-batched tensors).

### PlanResult (output)

| Field | Type | Notes |
|---|---|---|
| `success` | `bool \| torch.Tensor` | Per-env success `(B,)` bool tensor (or scalar bool) |
| `xpos_list` | `torch.Tensor \| None` | EEF poses `(B, N, 4, 4)` |
| `positions` | `torch.Tensor \| None` | Joint positions `(B, N, DOF)` |
| `velocities` | `torch.Tensor \| None` | Joint velocities `(B, N, DOF)` |
| `accelerations` | `torch.Tensor \| None` | Joint accelerations `(B, N, DOF)` |
| `dt` | `torch.Tensor \| None` | Per-step arrival intervals `(B, N)`; required whenever `positions` is present |
| `duration` | `torch.Tensor \| None` | Read-only total trajectory time `(B,)`, derived as `dt.sum(dim=1)` |

Helper: `PlanResult.is_all_success() -> bool` returns `True` only when every env succeeded.
`PlanResult` rejects positions with missing, malformed, or inconsistent timing.
A failed result may omit the trajectory entirely by leaving `positions=None`.
When `MotionGenerator` resamples a fully timed result, it preserves each row's
total duration and emits new explicit arrival intervals.

### MoveType enum

| Value | Meaning |
|---|---|
| `TOOL` | Tool open or close command |
| `EEF_MOVE` | End-effector Cartesian move (IK + trajectory) |
| `JOINT_MOVE` | Joint-space move (trajectory planning only) |
| `SYNC` | Synchronized dual-arm movement |
| `PAUSE` | Pause for `pause_seconds` |

### MovePart enum

| Value | Meaning |
|---|---|
| `LEFT` | Left arm/EEF |
| `RIGHT` | Right arm/EEF |
| `BOTH` | Both arms/EEFs |
| `TORSO` | Torso (humanoid) |
| `ALL` | All joints |

## Configuration

### Registering a new planner

1. Create a `BasePlanner` subclass with a `plan()` method decorated with `@validate_plan_options`; every result containing positions must include `dt`, from which `duration` is derived.
2. Create a `BasePlannerCfg` subclass with a unique `planner_type` string.
3. Optionally create a `PlanOptions` subclass for planner-specific options.
4. For a planner that accepts live obstacles, override `collision_world_info`
   with a `CollisionWorldInfo` whose `supports_updates=True`, and implement
   `with_collision_world()` without mutating caller-owned reusable options.
5. Register in `MotionGenerator._support_planner_dict`:
   ```python
   _support_planner_dict = {
       "toppra": (ToppraPlanner, ToppraPlannerCfg),
       "neural": (NeuralPlanner, NeuralPlannerCfg),
   }
   ```
6. Export from `embodichain/lab/sim/planners/__init__.py`.

### validate_plan_options decorator

Applied to `plan()` methods to type-check the `options` argument at runtime and enforce batch consistency. Supports three styles:
- `@validate_plan_options` — bare; validates against base `PlanOptions`.
- `@validate_plan_options()` — called with no args; same as above.
- `@validate_plan_options(options_cls=MyPlanOptions)` — custom options class.

The decorator checks that every `PlanState` in `target_states` shares the same leading batch dim `B` and that `B` matches `robot.num_instances` (or is `1`).

### Constraint checking

`BasePlanner.is_satisfied_constraint(vels, accs, constraints)` verifies trajectory outputs stay within limits. Tolerance: 10% for velocity, 25% for acceleration. Supports batch dimensions `(B, N, DOF)`.

## Common Failure Modes

- **`robot_uid` is MISSING** — `BasePlannerCfg.robot_uid` defaults to `MISSING`. Forgetting to set it raises `ValueError` at planner init.
- **Robot not found** — planner init calls `SimulationManager.get_instance().get_robot(uid)`. If the robot hasn't been added to the sim yet, this returns `None` and raises `ValueError`.
- **toppra not installed** — `ToppraPlanner` import fails with `ImportError` at module load time if `toppra==0.6.3` is not installed.
- **Batch dim mismatch** — `@validate_plan_options` raises `ValueError` if `PlanState` entries have inconsistent `B` or if `B` does not equal `robot.num_instances`.
- **Single-env caller shape mismatch** — legacy callers passing `(DOF,)` qpos or `(4,4)` xpos must wrap with `PlanState.single(...)` or call `from_qpos`/`from_xpos` with a leading `B=1` dim.
- **MotionGenerator planner_type not registered** — if `planner_cfg.planner_type` is not in `_support_planner_dict`, `MotionGenerator.__init__` fails. Register new planners there first.
- **IK interpolation with unsupported MoveType** — `strategy="ik_interp"`
  accepts only `EEF_MOVE` and `JOINT_MOVE` and raises for other target types.
- **Missing interpolation inputs** — `strategy="ik_interp"` requires explicit
  `start_qpos`, `sample_count`, and `interpolation_dt`; it never reads live robot
  state or guesses a command period implicitly.
- **Missing planner timing** — constructing a `PlanResult` with positions but
  without `dt` raises immediately; `duration` is derived from `dt`.
- **CUDA requested on a CPU-only runtime** — planner success-mask normalization
  raises a direct `ValueError` before querying the active CUDA device. It never
  silently falls back to CPU.
- **Constraint tolerance** — `is_satisfied_constraint` allows 10% velocity / 25% acceleration overshoot. Dense waypoint trajectories may appear to violate constraints but pass validation.
- **Fork safety with GPU sim** — `ToppraPlannerCfg.mp_context=None` defaults to `spawn` on GPU to avoid fork-after-CUDA-init hazards. Force `fork` only when the sim device is CPU or you have verified it is safe.
- **cuRobo shared-world mismatch** — World-frame poses may differ solely because replicated arenas are offset. Compare poses after robot-base rebasing: keep `multi_env=False` if they match, and enable it only when robot-relative layouts differ.
- **Dynamic obstacles silently stale** — A planner participates in atomic-action collision revision recovery only when `collision_world_info.supports_updates=True`; its hook must bind every `collision_entity_id` pose into the current planning attempt.
- **Registry/planner identity drift** — Registry-backed cuRobo worlds must use a
  canonical-ID mapping, not a list whose names are inferred from UIDs. Validate
  exact full registry/planner collision-world agreement, dynamic
  registry/provider/planner agreement, and batch-mode agreement through
  `SceneRegistry` before starting execution.
