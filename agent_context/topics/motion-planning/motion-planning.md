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
| Motion generator | `embodichain/lab/sim/motion/motion_generator.py` → `MotionGenerator`, `MotionGenCfg`, `MotionGenOptions` |
| Planner utilities & data types | `embodichain/lab/sim/motion/planners/utils.py` → `PlanState`, `PlanResult`, `MoveType`, `MovePart`, `TrajectorySampleMethod`, `interpolate_xpos_batched` |
| Trajectory augmentation | `embodichain/lab/sim/motion/expansion/` → contracts, configs, operators, coverage, `GenerationSession` |

## Overview

Planners, solvers, workspace analysis, and trajectory augmentation are sibling
packages under `embodichain.lab.sim.motion`. Import planner APIs from
`embodichain.lab.sim.motion.planners`; the parent namespace resolves subpackages
lazily and must not eagerly load planners during Robot initialization. Import the coordinating `MotionGenerator`, `MotionGenCfg` and `MotionGenOptions`
from `embodichain.lab.sim.motion.motion_generator`. Planners do not re-export
this facade; solver and planner imports must not load it eagerly. Atomic
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

`motion/expansion/` owns immutable trajectory/candidate contracts,
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
the object nor certifies the grasp.

Focused augmentation tests live under `tests/sim/motion/expansion/`.

Read [fixed-scene trajectory generation](trajectory-generation.md) for host restoration, qpos rollout, offline PickUp sources, contact validation, and confirmed dataset persistence.

## Choose the owning layer

- `BasePlanner` and `PlanState` / `PlanResult` define planning interfaces.
- `ToppraPlanner` owns time parameterization; `CuroboPlanner` owns collision-aware planning.
- `MotionGenerator` composes motion commands and trajectory helpers; `NeuralPlanner` is experimental.
- [Planner details](planner-details.md) cover process/memory behavior, registration and validation.
- [Collision worlds](collision-worlds.md) cover snapshots, pose updates, provenance and cache boundaries.

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

## Shared trajectory computations

`embodichain.compute.trajectory` owns pure interpolation, path resampling,
and keyframe-based warping. `interpolate_with_distance` retains keyframes;
`resample_with_distance` treats interior points as optional path samples.
MotionGenerator and atomic trajectory helpers import the compute API directly.
`lab.sim.utility.action_utils` retains solver-dependent pose/IK adaptation and
re-exports pure functions for compatibility. Warp implementations live in
`compute/trajectory/_warp/`; tests belong to `tests/compute/test_trajectory.py`.
