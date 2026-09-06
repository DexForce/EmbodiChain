# Motion Planning

## Entry Points

| What | Path |
|---|---|
| Planner registry | `embodichain/lab/sim/motion/planners/__init__.py` |
| Base planner class & config | `embodichain/lab/sim/motion/planners/base_planner.py` → `BasePlanner`, `BasePlannerCfg`, `CollisionWorldInfo`, `PlanOptions`, `validate_plan_options` |
| TOPPRA planner | `embodichain/lab/sim/motion/planners/toppra_planner.py` → `ToppraPlanner`, `ToppraPlannerCfg`, `ToppraPlanOptions` |
| Neural planner | `embodichain/lab/sim/motion/planners/neural_planner.py` → `NeuralPlanner`, `NeuralPlannerCfg`, `NeuralPlanOptions` |
| cuRobo planner | `embodichain/lab/sim/motion/planners/curobo/curobo_planner.py` → `CuroboPlanner`, `CuroboPlannerCfg`, `CuroboWorldCfg`, `CuroboPlanOptions` |
| Planner assets | `embodichain/data/assets/planner_assets.py` → `download_neural_planner_checkpoint()` |
| Motion generator | `embodichain/lab/sim/motion/planners/motion_generator.py` → `MotionGenerator`, `MotionGenCfg`, `MotionGenOptions` |
| Planner utilities & data types | `embodichain/lab/sim/motion/planners/utils.py` → `PlanState`, `PlanResult`, `MoveType`, `MovePart`, `TrajectorySampleMethod`, `interpolate_xpos_batched` |
| Trajectory augmentation | `embodichain/lab/sim/motion/trajectory_augmentation/` → contracts, configs, operators, coverage, `GenerationSession` |

## Overview

Planners, solvers, workspace analysis, and trajectory augmentation are sibling
packages under `embodichain.lab.sim.motion`. Import planner APIs from
`embodichain.lab.sim.motion.planners`; the parent namespace resolves subpackages
lazily and must not eagerly load planners during Robot initialization. Atomic
Actions consume these motion capabilities from `sim/atomic_actions/`.
Focused tests and examples live under `tests/sim/motion/` and
`examples/sim/motion/`.

The planning stack has two layers:
1. **BasePlanner** — low-level trajectory planner that takes a list of `PlanState` waypoints and produces a `PlanResult` with joint trajectories.
2. **MotionGenerator** — the single stateful planning facade that composes a
   planner with strategy selection, interpolation, IK resolution, result
   normalization, and multi-part coordination.

All planners resolve their robot at init via `SimulationManager.get_instance().get_robot(cfg.robot_uid)`.

The entire stack is **env-batched** (`B = num_envs`). `PlanState` / `PlanResult` tensors carry a leading `B` dimension; `BasePlanner.plan()` and `MotionGenerator.generate()` operate on `B` environments in one call.

## Trajectory Augmentation Boundary

`motion/trajectory_augmentation/` owns immutable trajectory/candidate contracts,
strict configuration decoding, phase-authorized joint residuals and retiming,
sampled motion-limit checks, geometric/timing coverage, and bounded generation
session bookkeeping. Candidates are logical rows; their identities and local
random seeds do not derive from physical environment slots.

`GenerationSession` accounts for ready candidates, rollout and pending-write
budgets, accepted-episode coverage reservations, and idempotent commit receipts.
It never steps or resets an environment or writes a dataset. Execution,
initial-state restoration, physical/task validation, and durable sinks belong
to host integrations. Keep algorithm modules free of direct Gym imports while
recognizing that the public package follows normal `lab/sim` initialization.
Motion-limit validation alone does not establish collision freedom or task
success.

`rotate_grasp_about_object_axis` rotates a reference TCP pose about a fixed
object-local axis through the object origin. The caller chooses geometry-valid
angles and replans the resulting pose candidates; the operator neither moves
the object nor certifies the grasp. A cube's quarter-turn symmetry is used by
`examples/sim/motion/trajectory_generation/cube_grasp_parallel.py` to compare
0/90-degree grasp orientations crossed with reference/residual transit paths.
Four physics rows execute together using existing MoveEndEffector/PickUp plans;
only the pre-grasp transit allows `joint_residual`. The example saves a tiled
camera MP4 with measured TCP trails, `rollout.npz`, and `report.json`. Physical
acceptance checks sustained lift and TCP-relative position stability. It uses
IK interpolation and does not claim contact-aware collision validation or
LeRobot expert qualification; the separate `cube_pickup_collection.py` performs
that bounded collection workflow.

`lab/trajectory_generation/runner.py` owns the synchronous qpos job for
handwritten free motion and atomic PickUp references with offline or observed
runtime execution. It resolves supplied host/planner/executor/sink policy identities,
reserves bounded rollout capacity through `GenerationSession`, and restores the
full batch only when ready candidates need another round. The concrete
`QposRolloutExecutor` uses normal Gym demo/controller ports or one pure-sim
step owner, captures actual submitted targets and T+1 observations, and checks
the manager's actual control-time increments. Measured qpos is rechecked for
collision, dynamics, reference-relative length/duration, and task success.
Only `LeRobotEpisodeSink` receipts confirmed after sealed readback update
committed counts and coverage; the runner closes its host/sink and writes
`generation_report.json` on completion or failure.

`integrations/planning.py` exposes `EEFPath` and `EnvRowMotionPlanner`: logical
candidate count C is scheduled over real physical B rows, without changing the
backend's batch size. It supports fully annotated free paths, exact qpos checks,
EEF IK/FK conversion and explicitly supplied solved branches. Contact, held
objects, changing locked joints, and unavailable backend checks cannot pass.
Joint-path collision checks use bounded sample densification, not a continuous
collision guarantee. The cuRobo lock model cache includes actual locked-joint
initial values and rejects runtime configuration drift until explicitly closed.

Generation currently requires every physical rigid UID in both the collision
world and its pose-update IDs, so each initial-state snapshot supplies the
correct per-row geometry pose. Different relative layouts require a per-env
world. The executor certifies fixed root/rigid poses at observation boundaries;
implicit floor geometry and other fixed conditions remain the trusted profile's
responsibility. This free-motion integration does not qualify atomic PickUp,
attachment geometry, or the planned four source/host combinations.

Focused integration tests are in `tests/lab/trajectory_generation/`; use the
CPU tests for orchestration and resource limits, and opt into the serialized
real cuRobo/physics smoke tests when changing those adapters.

`examples/sim/motion/trajectory_generation/free_motion.py` runs the complete
normal-gravity pure-arm UR5 pipeline and emits a LeRobot shard plus report.
`--record-video --duration 5 --joint-displacement 0.4` also streams a 640x480,
20 fps offscreen camera preview to `preview.mp4`. Rendering observes the actual
rollout without stepping physics; video includes both initial and terminal
observations and stays separate from the numeric LeRobot schema.
`--robot panda` is a real rejection case: motion-induced hand/mimic displacement
exceeds the locked-joint model tolerance, so actual collision verification is
unavailable and no expert episode is committed. These two real cases are in
`tests/lab/trajectory_generation/test_runner_real.py`; they do not replace the
planned PickUp source/host qualification matrix.

## PickUp Collection Integration

`integrations/atomic.py::export_pickup_templates` accepts a successful offline
MoveEndEffector → PickUp compilation, preserves its explicit phase boundaries,
expands mimic coordinates, and adds real terminal hold commands. Only transit
allows residuals. It exports qpos references, without committing projected
symbolic effects or executing AtomicActionRuntime recovery/tracking.

`integrations/contact.py::PickUpMotionValidator` is the shared Runner planner
and executor contact validator. It uses conservative URDF collision hulls,
full-joint FK, cuboid world geometry, an implicit floor, phase/contact-entry
permissions, and planned held-object geometry. Actual collision checks use
measured fingers/object poses. Native contacts are checked after every 5 ms
CPU physics step; forbidden/unknown/cross-row/deep contacts, buffer overflow,
missing finger contact, slipping and dropping cannot qualify. Mounting links
must be fixed to the robot root. Two-hop self-pair exclusions match cuRobo.
The executor exempts only this target and declared mimic joints from its
free-motion immobility rules; all remaining root/world checks remain active.

`examples/sim/motion/trajectory_generation/cube_pickup_collection.py` runs four
rows, repeatedly restores the full initial state, and collects confirmed
LeRobot shards with optional synchronized video. Its 0/90-degree targets use
independent residual transit paths. Controller stiffness/opening margin are
set before capture; collision and tracking thresholds are not relaxed per row.
The sink flattens numeric matrices in C order, records `observation_shapes`,
and preserves terminal matrices. `commanded_joint_indices` distinguishes active
controller labels from full-joint passive target columns.

This is CPU-physics / unscaled fixed-base URDF / cuboid-rigid support. The
optional `trajectory-generation` dependencies supply FCL/trimesh/yourdfpy.
It is sampled validation, not continuous collision detection. Contact-aware
Gym, general via points and a registry/CLI remain work.
`integrations/atomic_runtime.py::PickUpRuntimeSource` enables the optional
`QposRolloutExecutor(runtime_source=...)` path. A fresh invocation/current context
materializes the selected five-phase candidate through the existing PickUp skill;
ExecutionRunner checks arm feedback while native contact and measured transforms
verify the held-object effect before commit. The initial sample is an observation,
not a command. Clock, command or recovery-generation mismatches cancel/hold and
reject the complete active batch; recovery demonstrations are not collected.
Use `cube_pickup_collection.py --runtime --record-video` for this pure-sim path.
Focused tests: `test_contact.py`, `test_atomic_source.py`, `test_episode_sinks.py`,
`test_atomic_runtime.py`, `test_pickup_collection.py` under
`tests/lab/trajectory_generation/`.

## Planner Hierarchy

```
BasePlanner (ABC)
  ├─ ToppraPlanner        Time-optimal path parameterization (fork-pool fan-out)
  ├─ NeuralPlanner (experimental)   APG waypoint rollout (native batching)
  └─ CuroboPlanner        CUDA collision-aware planning (native batching)

MotionGenerator           Wraps any BasePlanner; adds interpolation and multi-part support
```

Config hierarchy:
```
BasePlannerCfg            robot_uid (MISSING), planner_type
  ├─ ToppraPlannerCfg     planner_type = "toppra", max_workers, mp_context
  └─ NeuralPlannerCfg     planner_type = "neural", checkpoint_path (MISSING)

MotionGenCfg              planner_cfg (MISSING — must be a BasePlannerCfg subclass)

PlanOptions               (empty base)
  ├─ ToppraPlanOptions    constraints, sample_method, sample_interval
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
Atomic Skills and Task Program install a generator instance in
`AtomicActionEngine`. Reference deployments select execution policies under
`configs/components/execution_policies/`, while reusable embodiment components
own the simulation robot and sensors; their optional `skill_profile` owns
grasp-generator geometry/tuning. Task-local integrations retain only semantic
action options and typed task-specific overrides.
`task_program/integrations/_configured_composition.py` checks the component
contracts and lowers the result through the strict decoder in `configured.py`;
the config loader registers the existing `EmbodiedEnv` under the selected ID.
There are no task-specific environment subclasses or direct-planning paths.

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
6. Export from `embodichain/lab/sim/motion/planners/__init__.py`.

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
