# CuRobo V2 Motion-Planning Integration Design

**Status:** Approved for implementation on 2026-07-11

## Goal

Integrate NVIDIA cuRobo V2 as an optional, collision-aware motion-planning
backend for EmbodiChain.  The integration must preserve EmbodiChain's existing
planner, motion-generator, and atomic-action contracts; provide a runnable
single-arm simulation demonstration; and remain importable and testable on
installations that do not have cuRobo or CUDA.

## Context and Constraints

EmbodiChain separates kinematics from planning:

```text
RobotCfg / Robot
    └── control parts and IK solvers
BasePlanner
    └── MotionGenerator
            └── AtomicActionEngine -> AtomicAction -> ActionResult
```

`BasePlanner.plan()` consumes env-batched `PlanState` waypoints and returns a
`PlanResult`.  `MotionGenerator` owns the selected planner.  Atomic actions use
`TrajectoryBuilder` to turn that result into a full-robot trajectory.  The new
backend must fit this chain; it must not invoke `env.step()` or create an
alternative action-execution API.

The integration targets **cuRobo V2 only**.  IsaacLab's implementation is a
useful reference for backend boundaries, named-joint remapping, and collision
world lifecycle, but it uses the incompatible cuRobo V1 `MotionGen` API and is
not copied.

cuRobo requires a CUDA-capable NVIDIA GPU.  It remains an optional runtime
dependency because its CUDA extra must match the user's installed PyTorch and
driver.  EmbodiChain will not add cuRobo to core dependencies or import it at
module import time.

## Scope

### Included

- A cuRobo V2 planner backend registered as `planner_type="curobo"`.
- Cartesian pose planning and joint-space planning through the existing
  `PlanState` / `PlanResult` interface.
- Single-arm control-part planning, including robust named-joint ordering,
  locked non-controlled joints, TCP, root-frame, and joint-limit handling.
- Batch-aware planning: the adapter accepts EmbodiChain's leading batch
  dimension and uses cuRobo's V2 batch planner where multiple environments are
  requested.  The CUDA integration test establishes the single-environment
  path; batch behavior is covered by adapter-level tests and runtime validation.
- Explicit static collision-world profiles and an explicit API to update named
  dynamic obstacle poses.  The first demo uses one static obstacle represented
  in both DexSim and cuRobo.
- `MotionGenerator` propagation of planner-specific execution context
  (`start_qpos` and `control_part`) without planner class special-cases.
- Atomic-action routing for single-arm Cartesian actions and opt-in
  collision-aware `MoveJoints`.
- Unit, optional CUDA, and real-simulation end-to-end tests.
- A runnable demo at `examples/sim/planners/curobo_planner.py`.

### Explicitly Excluded from This Change

- cuRobo V1 compatibility or a hidden V1 fallback.
- Automatic conversion of every DexSim entity into a cuRobo collision object.
- Attached-object collision geometry and automatic attachment/detachment while
  picking.  The public context boundary is designed so this can be added later.
- Coordinated dual-arm planning and the current `CoordinatedPickment` path.
- The legacy Gym ActionBank system, which has a separate NumPy trajectory
  representation and execution lifecycle.
- CPU fallback.  A cuRobo planner construction request without CUDA fails with
  an actionable error.

## Architecture

### Configuration and Public API

The planner package adds focused, serializable config objects:

```text
CuroboRobotProfileCfg
    robot_config_path
    active_joint_names
    base_link_name
    sim_base_link_name
    sim_base_to_curobo_base
    tool_frame_name
    tool_frame_to_tcp

CuroboWorldCfg
    world_config_path
    dynamic_obstacle_names

CuroboPlannerCfg(BasePlannerCfg)
    planner_type = "curobo"
    robot_profiles: dict[str, CuroboRobotProfileCfg]
    world: CuroboWorldCfg
    warmup: bool
    collision_activation_distance
    max_attempts
    max_planning_time
    use_cuda_graph

CuroboPlanOptions(PlanOptions)
    start_qpos
    control_part
    dynamic_obstacle_poses: dict[str, Tensor[B, 4, 4]]
    velocity_scale
    acceleration_scale
    max_attempts
```

Each `robot_profiles` key is an EmbodiChain control-part name.  The selected
profile explicitly maps between simulator joint names and cuRobo joint names;
the backend never assumes that simulator indices and cuRobo indices are equal.
Any non-controlled joints, including gripper joints, must be locked in the
cuRobo V2 robot profile so they do not appear in the backend's active joint
list. Their preserved simulator values must equal the V2 profile's
`lock_joints` values during planning and atomic-action playback; the first
release documents this cross-model lock contract but does not automatically
validate joint-name/value equivalence. The backend rejects an active-joint/
profile mismatch rather than planning collision geometry that EmbodiChain will
not execute.

`CuroboPlanner` is a `BasePlanner` implementation.  It lazily imports V2
types, creates and warms a `MotionPlanner` for a one-environment request and a
V2 `BatchMotionPlanner` for a multi-environment request, then returns the
standard result fields:

```text
success:       bool tensor (B,)
positions:     float tensor (B, N, controlled_dof)
velocities:    float tensor (B, N, controlled_dof) or None
accelerations: float tensor (B, N, controlled_dof) or None
dt:            float tensor (B, N)
duration:      float tensor (B,)
xpos_list:     float tensor (B, N, 4, 4) when FK is available
```

The planner supports `EEF_MOVE` by converting EmbodiChain world-frame matrices
to cuRobo target poses, including the selected robot base transform and TCP.
It supports `JOINT_MOVE` with cuRobo's joint-space planning API.  Unsupported
move types fail before planning with a clear `ValueError`.

### Motion-Generator Integration

`MotionGenerator` registers `"curobo"` alongside existing planner types.
`BasePlanner` gains two explicit extension points:

- `preinterpolate_targets: bool`, which defaults to `True`;
- `supports_joint_move: bool`, which defaults to `False` and keeps
  Cartesian-only backends on local joint interpolation for atomic phases;
- `with_motion_context(options, *, start_qpos, control_part)`, whose base
  implementation returns the supplied options unchanged.

Planner capabilities replace the current `isinstance(NeuralPlanner)` decision:

- planners declare whether EmbodiChain pre-interpolation is appropriate;
- CuRobo declares it is not appropriate for Cartesian targets, so it receives
  the original `EEF_MOVE` targets and performs its own collision-aware IK and
  trajectory optimization;
- planners receive the same runtime context through
  `with_motion_context(...)`, rather than `MotionGenerator` copying fields
  only for one concrete planner.

This prevents a Cartesian action from being silently converted through
EmbodiChain IK before CuRobo sees it.  It also ensures the start position and
the requested control part reach `CuroboPlanOptions` on every call.

### Collision-World Boundary

The initial collision world is explicit and deterministic.  `CuroboWorldCfg`
points to a cuRobo V2 scene profile containing mesh and primitive obstacles.
At initialization the adapter loads this static scene once and warms the
planner.  Before a plan, callers may supply poses keyed by the configured
dynamic obstacle names; the adapter verifies every name and updates only those
poses.  Adding or removing geometry requires an explicit world reload rather
than a fragile implicit scene scan.

All obstacle poses, goal poses, robot base poses, and tool poses use a single
documented world coordinate convention.  The adapter converts EmbodiChain
4x4 matrices to cuRobo's position/quaternion representation at this boundary
and tests the quaternion ordering directly. Static collision YAML is authored
in the cuRobo base frame and therefore applies to fixed-base scenes; a moving
base must publish relevant obstacles through named dynamic updates.

### Atomic-Action Integration

The existing data flow remains intact:

```text
AtomicActionEngine
  -> TrajectoryBuilder.plan_arm_traj
  -> MotionGenerator.generate
  -> CuroboPlanner.plan
  -> PlanResult
  -> full-DoF ActionResult trajectory
```

`ActionCfg.planner_type` documents and validates `"curobo"`.  When the action
uses `motion_source="motion_gen"` and a CuRobo generator, the builder creates
`CuroboPlanOptions`, disables EmbodiChain Cartesian pre-interpolation, and
embeds the controlled joint plan into full robot DoF exactly as it does for
other planners.

The supported first-release atomic surface is:

- `MoveEndEffector`;
- movement phases of `PickUp`, `Place`, `Press`, and `MoveHeldObject` in a
  static collision scene;
- `MoveJoints` when explicitly configured with `motion_source="motion_gen"`.

Default action behavior remains unchanged (`motion_source="ik_interp"`).
Coordinated two-arm actions reject the CuRobo backend until they have a
separate multi-arm planning design.  Because attached-object collision modeling
is excluded, users must not claim collision-aware carrying for a held object
until that later extension is implemented.

### Optional Dependency Behavior

The public planner package exports its configuration types without importing
cuRobo.  Constructing or calling `CuroboPlanner` performs the dependency and
CUDA checks.  A missing dependency raises an error that names the official V2
installation command family (`.[cu12]` or `.[cu13]`) and links to the project
installation instructions.  Tests that require the real library use
`pytest.importorskip("curobo")`.

## Error Handling and Correctness Rules

- Missing `robot_uid`, unknown control part, absent profile, joint-name
  mismatch, invalid robot/world profile, or incompatible batch size fail before
  a CUDA plan is launched.
- A failed cuRobo solution becomes a per-environment `success` tensor.  Its
  trajectory is held at `start_qpos`, matching the existing
  `AtomicActionEngine` failed-environment behavior.
- Planner output is reordered from cuRobo named joints back to EmbodiChain
  control-part order before it reaches `PlanResult`.
- Output tensors are moved to the robot's device and have the exact batch and
  DoF shapes expected by `BasePlanner`.
- The implementation validates that an action's requested `planner_type`
  matches the already constructed `MotionGenerator` planner; it never silently
  plans with TOPPRA or NeuralPlanner under a CuRobo label.

## Verification Strategy

1. **Pure unit tests, no cuRobo required**
   - config validation, registry/export behavior, optional-import errors;
   - planner capability propagation through `MotionGenerator`;
   - named-joint reorder and matrix/pose conversion helpers;
   - `TrajectoryBuilder` and `MoveJoints` routing with a fake CuRobo backend.

2. **Optional CUDA/cuRobo integration tests**
   - create a V2 planner with a Panda profile and static obstacle;
   - plan a collision-free pose trajectory and assert success, finite values,
     start state, endpoint tolerance, joint ordering, and time fields;
   - update a configured obstacle pose and verify the update reaches the
     backend.

3. **DexSim end-to-end test**
   - create a single-arm robot and a matching static obstacle;
   - execute `AtomicActionEngine + MoveEndEffector` with
     `planner_type="curobo"`;
   - assert a full-DoF trajectory, per-environment success, and target-pose
     tolerance after playback.  Mark this test `requires_sim` and `slow`.

4. **Runnable demo**
   - instantiate the same robot, obstacle, profile, planner, and atomic
     action;
   - plan, print success/shape/endpoint error, and replay the returned full-DoF
     trajectory;
   - fail early with installation or profile guidance when CuRobo/CUDA is not
     available.

## Documentation and Installation

The integration documentation will identify CuRobo V2 as an optional CUDA
dependency and send users to NVIDIA's official installation flow.  It will
explain how to generate a robot profile with `RobotBuilder`, how to configure
a static world, how to select `planner_type="curobo"`, and the first-release
limits on dynamic geometry, attachments, coordinated arms, and ActionBank.

## Acceptance Criteria

- `import embodichain.lab.sim.planners` works when cuRobo is absent.
- An installed V2/cuRobo CUDA environment plans a collision-free Panda path
  through the EmbodiChain `MotionGenerator` API.
- `MoveEndEffector` reaches a target through `AtomicActionEngine` with a
  full-DoF result trajectory and without EmbodiChain pre-IK.
- Existing TOPPRA, NeuralPlanner, default atomic actions, and ActionBank
  behavior continue to pass their existing tests.
- `examples/sim/planners/curobo_planner.py` runs with documented V2 profiles
  and demonstrates obstacle-aware planning and playback.
