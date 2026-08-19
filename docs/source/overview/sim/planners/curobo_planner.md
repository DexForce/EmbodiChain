# cuRobo V2 Planner

CuroboPlanner is EmbodiChain's optional, CUDA-accelerated and collision-aware
motion-planning backend. It implements the normal MotionGenerator and
atomic-action interfaces while cuRobo performs collision-aware inverse
kinematics and trajectory optimization. It supports Cartesian EEF_MOVE and
joint-space JOINT_MOVE requests for one configured control part at a time.

planner_type="curobo" selects this backend. cuRobo V2 is deliberately not an
EmbodiChain core dependency: importing EmbodiChain planners does not import
cuRobo, and constructing this planner requires a CUDA-capable NVIDIA GPU.

## Install cuRobo V2

cuRobo V2 is installed separately from EmbodiChain because public package
indexes do not accept Git dependencies in published package metadata. Select
exactly one CUDA-matched source requirement:

~~~bash
# Recommended for the normal EmbodiChain environment, where PyTorch is present.
uv pip install "nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo.git@v0.8.0"
uv pip install "nvidia-curobo[cu13] @ git+https://github.com/NVlabs/curobo.git@v0.8.0"

# For a fresh environment that also needs PyTorch.
uv pip install "nvidia-curobo[cu12-torch] @ git+https://github.com/NVlabs/curobo.git@v0.8.0"
uv pip install "nvidia-curobo[cu13-torch] @ git+https://github.com/NVlabs/curobo.git@v0.8.0"

python -c "import curobo; print(curobo.__version__)"
pytest --pyargs curobo.tests
~~~

These commands follow [NVIDIA's official cuRobo installation
guide](https://nvlabs.github.io/curobo/latest/getting-started/installation.html)
and pin the source dependency to the cuRobo V2 `v0.8.0` release. Use a Python
3.10--3.13 environment on Linux with a supported NVIDIA GPU and driver. The
non-`torch` variants are preferred for EmbodiChain because the simulation
environment normally already provides PyTorch; the `-torch` variants delegate
the PyTorch version requirement to cuRobo. Keep cuRobo in the same Python
environment that runs the simulator.

## Configure a control part

The cuRobo robot model and the per-control-part profile are both auto-generated
internally - no external cuRobo robot YAML (e.g. `franka.yml`) and no
`robot_profiles` config are needed. On the first plan, the adapter fits collision
spheres to each link of the robot's URDF and writes a cuRobo V2 robot YAML (see
[Auto-generated robot YAML](curobo-auto-generated-robot-yaml). The tool frame, TCP
offset, and base link are read from the control part's IK solver, and the
simulator->cuRobo joint mapping is identity (the generated YAML reuses the
URDF's own joint names). The control part is selected at plan time through
`CuroboPlanOptions.control_part` and validated against `robot.control_parts`.

Lock non-controlled joints (for example gripper joints) in the cuRobo robot
profile so they are not exposed as active planner joints. The simulator values of
those joints must remain equal to the V2 profile's `lock_joints` values while a
plan is executed; the adapter intentionally preserves non-control simulator
joints in the full-DoF atomic-action output. For example, the Panda V2 profile
locks both fingers at `0.04`, so use the same simulated finger state or include
the fingers in the planned control part. A mismatch means cuRobo validates a
different collision geometry from the one replayed in DexSim.

Assuming the scene has been registered as shown in
{doc}`../scene_registry`, construct the planner world from that catalog:

~~~python
from embodichain.lab.sim.planners import (
    CuroboPlannerCfg,
    CuroboWorldCfg,
    MotionGenCfg,
    MotionGenerator,
)
from embodichain.lab.sim.skills import SceneCollisionWorldMode

collision_mode = registry.resolve_collision_world_mode(
    batch_size=robot.num_instances,
)

planner_cfg = CuroboPlannerCfg(
    robot_uid="my_franka",
    planner_type="curobo",
    world=CuroboWorldCfg(
        rigid_objects=registry.collision_geometry_by_id(),
        obstacle_representation="cuboid",
        dynamic_obstacle_names=list(registry.dynamic_collision_entity_ids),
        multi_env=collision_mode is SceneCollisionWorldMode.PER_ENV,
    ),
)
motion_generator = MotionGenerator(MotionGenCfg(planner_cfg=planner_cfg))
scene_provider = registry.make_planning_scene_provider(
    motion_generator,
    batch_size=robot.num_instances,
)
~~~

cuRobo's Python logger defaults to error-only output. Set
`CuroboPlannerCfg.log_level` to `"debug"`, `"info"`, `"warning"`, or `"error"`
to change its verbosity. This setting does not affect EmbodiChain's own logs.

The physics and planner devices are independent.
`SimulationManagerCfg(sim_device="cpu")` keeps robot state, targets, and
returned trajectories on CPU,
while cuRobo still performs all model generation and planning on CUDA. By
default a CPU simulation uses PyTorch's current CUDA device; set
`CuroboPlannerCfg.cuda_device="cuda:1"` (or an integer GPU index) to select a
different planning GPU. A CPU value is rejected because cuRobo itself has no
CPU backend.

The robot configuration must be a cuRobo V2 robot profile with collision
spheres and self-collision data; the adapter generates this from the robot's
URDF automatically. A plain URDF alone is not sufficient for collision planning
without that sphere-fitting step.

The adapter automatically rebases simulator-world Cartesian goals and dynamic
obstacle poses through the live simulator control-part base, so parallel arena
offsets and a moved robot base are handled. If the simulator and cuRobo base
frames use different fixed conventions, set
`CuroboPlannerCfg.sim_base_to_curobo_base` to the transform from the simulator
base to the cuRobo base. Collision-world poses are authored in the cuRobo
base/world frame. `tool_frame_to_tcp` (read from `solver.tcp_xpos`) converts an
EmbodiChain TCP goal into the chosen cuRobo tool frame when the solver's end link
is not itself the TCP. By convention, the adapter uses
`T_curobo,X = T_curobo,sim_base @ inv(T_world,sim_base) @ T_world,X`. It obtains
the simulator base from the control part's IK solver root.

`CuroboPlannerCfg.use_cuda_graph` defaults to `True`. The planner runs in the
simulator process and reuses its CUDA context; it does not launch a persistent
`spawn` worker or copy planning tensors through multiprocessing queues. Set
`use_cuda_graph=False` when lower one-time initialization cost and lower
graph-resident memory are more important than hot planning latency.

In graph mode, planner initialization uses the same per-device
`CaptureCoordinator` as DexSim's Newton backend and synchronizes the device
before and after cuRobo warmup. EmbodiChain also forces cuRobo's PyTorch graphs
to use `cuda_graph_capture_error_mode="thread_local"` (the default).
Unlike PyTorch's strict `"global"` mode, this allows DexSim's Vulkan render
thread to continue making CUDA calls without invalidating capture on the
planner thread.

If coordinator acquisition times out before recording begins,
`cuda_graph_fallback=True` waits for the active capture to finish and builds a
non-graph backend. An exception after graph recording starts is deliberately
not downgraded: CUDA may have invalidated the process context, so the planner
raises an error and requires a simulator-process restart. The `"global"` and
`"relaxed"` modes remain available for diagnosis, but `"thread_local"` is the
supported renderer-compatible setting.

cuRobo cannot reset a captured trajectory-optimizer graph when switching
between Cartesian pose goals and joint-space goals. EmbodiChain therefore
caches those two goal types separately and initializes each lazily. Applications
that use only one move type retain one planner backend; using both incurs a
second one-time warmup and its graph-resident memory, but still no subprocess or
second CUDA context.

The collision world is always auto-generated from live `RigidObject` meshes via
`CuroboWorldCfg.rigid_objects`. The canonical, registry-backed form is a mapping
from authoritative registry ID to live object; the adapter reads each object's
mesh (`get_vertices` / `get_triangles`) and world pose (`get_local_pose`) and
writes a cached cuRobo scene YAML on the first plan, using
`CuroboWorldCfg.obstacle_representation` (`"sphere"` by default for fast
collision queries; use `"cuboid"` for a local-frame AABB placed as an OBB via
the object pose, or `"mesh"` for the exact triangle mesh).
Generated poses are authored in the cuRobo base/world frame, so this is exact
when the robot base sits at the simulator world origin. The mapping key, rather
than `RigidObject.uid`, is the canonical logical/source ID used by cache
identity and collision-world validation. For `"cuboid"` and `"mesh"`, that ID
is also used unchanged as the physical YAML obstacle name and runtime update
key. For obstacles that move or live in an offset base frame, also declare their
canonical IDs in
`CuroboWorldCfg.dynamic_obstacle_names` and update poses at plan time through
`CuroboPlanOptions.dynamic_obstacle_poses` (provision
`CuroboWorldCfg.collision_cache` before planning). Dynamic updates require the
`"cuboid"` or `"mesh"` representation because sphere fitting expands one object
into physical YAML obstacles named `<canonical_id>_0`, `<canonical_id>_1`, and
so on; dynamic sphere configuration is rejected. These derived names are
backend details. The cache and registry/planner full-world contract continue to
use the unexpanded canonical source ID.

Registry-backed mappings fail fast if a selected source has no mesh geometry
required by the chosen representation. This prevents a canonical collision ID
from being silently skipped during YAML generation. The advanced sequence form
retains its lower-level behavior independently of this registry contract.

`CuroboPlanner.collision_world_entity_ids` reports every configured logical
source ID: each mapping key on the registry path, or each inferred name on the
advanced sequence path. It deliberately does not expose sphere-expanded
physical YAML names. `dynamic_collision_entity_ids` reports exactly the
configured dynamic subset. Static entries therefore participate in
construction-time identity validation even though they do not receive per-plan
pose updates.

`CuroboWorldCfg` validates this planner-local registration at construction:
obstacle IDs must be unique, and every dynamic obstacle ID must match an entry
in `rigid_objects`. A sequence of objects is retained only as an advanced
direct-core path; it derives names from each `uid` or an `obstacle_<index>`
fallback. Do not use that form for a registry-backed world.

The {doc}`../scene_registry` integration performs two higher-level checks before
execution. First, all registry `STATIC ∪ DYNAMIC` IDs must exactly equal
`MotionGenerator.collision_world_entity_ids`. Second, registry, derived scene
provider, and planner dynamic-ID subsets must exactly agree. The planner must
also support pose updates and its shared/per-environment batch mode must agree
with the registry. Aliases are normalized at the registry boundary; cuRobo
never translates a canonical ID back to a simulator UID.

### Shared and per-environment collision worlds

`CuroboWorldCfg.multi_env` controls collision-world batching only. Robot start
states and planning goals remain batched regardless of this setting.

Choose the setting based on obstacle poses after EmbodiChain rebases them from
the simulator world frame into each environment's robot-base frame:

| Environment layout | Recommended setting |
|---|---|
| Replicated arenas have different simulator-world offsets, but each obstacle has the same pose relative to its local robot base | `multi_env=False` (default) |
| Obstacles have different poses relative to their respective robot bases, for example due to per-environment pose randomization | `multi_env=True` |

With `multi_env=False`, all batch rows share one collision world. Raw
simulator-world poses may differ—for example, because env 1 is translated from
env 0—but the shared world remains correct when rebasing removes the arena
offset and the resulting robot-relative poses are equal. If the rebased poses
differ, the adapter rejects the update and instructs the caller to enable
`multi_env`.

With `multi_env=True`, cuRobo allocates one collision world per batch row and
EmbodiChain sends row `i` of each dynamic obstacle pose to world `i`. The
auto-generated YAML still reads the static scene from env 0 and clones that
scene for every row; setting `multi_env=True` does not by itself discover each
environment's distinct initial object poses. Any object whose robot-relative
pose differs by environment must also:

1. Use `obstacle_representation="cuboid"` or `"mesh"`.
2. Be listed in `CuroboWorldCfg.dynamic_obstacle_names`.
3. Have its current `(B, 4, 4)` simulator-world poses passed through
   `CuroboPlanOptions.dynamic_obstacle_poses` when planning.

For a registry-backed world, derive both the geometry mapping and dynamic ID
list from the same catalog:

```python
world_cfg = CuroboWorldCfg(
    rigid_objects=registry.collision_geometry_by_id(),
    obstacle_representation="cuboid",
    dynamic_obstacle_names=list(registry.dynamic_collision_entity_ids),
    multi_env=True,
)

current_snapshot = scene_provider.snapshot(timestamp=now, env_ids=env_ids)
plan_options = CuroboPlanOptions(
    control_part="arm",
    dynamic_obstacle_poses=current_snapshot.collision_obstacle_poses(
        batch_size=robot.num_instances,
        device=robot.device,
        dtype=robot.get_qpos().dtype,
    ),
)
```

An empty world (`rigid_objects=None`) is likewise materialized once per row in
multi-env mode so its per-environment cache is allocated. Dynamic pose updates
still require the named geometry to already exist in every scene; the adapter
does not insert new geometry at runtime. Independent worlds replicate scene
data and collision caches across the batch, so retain the shared default when
the rebased layouts are identical.

For a registry-backed integration, a single-environment dynamic world may infer
the registry's shared mode. A multi-environment registry with dynamic collision
entities must explicitly choose shared or per-environment semantics, then set
`multi_env=False` or `True` to match. The registry validator rejects a mismatch
before planning.

(curobo-auto-generated-robot-yaml)=
## Auto-generated robot YAML

On the first plan, the adapter auto-derives the cuRobo robot profile from the
robot's URDF and solver, so nothing robot-specific needs to be hardcoded:

- `robot_config_path` is produced by `generate_curobo_robot_yaml`, which fits
  collision spheres to each link mesh and writes a cuRobo V2 robot YAML.
- The TCP, tool frame, and base link are read from the robot's solver
  (`robot._solvers[control_part]`): `tool_frame_name` <- `solver.end_link_name`,
  `tool_frame_to_tcp` <- `solver.tcp_xpos`, `base_link_name` <-
  `solver.root_link_name`.
- `sim_to_curobo_joint_names` is the identity mapping, since the generated YAML
  reuses the simulator's own URDF joint names.

The generated YAML is cached on disk (default `$XDG_CACHE_HOME/embodichain_curobo`
or `~/.cache/embodichain_curobo`) keyed by the URDF path, URDF content, control
part, tool frame, and fit parameters, so editing the URDF or changing the fit
settings regenerates automatically and subsequent inits reuse the cache. Tune the
fit with `CuroboPlannerCfg.auto_gen` (`fit_type="voxel"` by default for fast
first-generation; `"morphit"` for best quality; `force=True` to bypass the cache).
The default `sphere_density=0.1` keeps the per-link sphere count low (~80 for a
Panda) so planning stays fast; raise it for tighter collision coverage.

## Generate a motion

MotionGenerator passes start_qpos and control_part to the cuRobo backend. For
Cartesian goals, leave EmbodiChain pre-interpolation disabled: cuRobo must
receive the original pose. By default the returned collision-checked samples are
arc-length resampled to the invocation's `MotionPolicy.sample_count` waypoint
count (so the same runtime policy controls trajectory length across planners);
set `CuroboPlannerCfg.preserve_plan_samples=True` to keep
cuRobo's own samples (whose count is derived from `interpolation_dt` and the
trajectory duration).

~~~python
import torch

from embodichain.lab.sim.planners import (
    CuroboPlanOptions,
    MotionGenOptions,
    PlanState,
)

goal_pose = torch.eye(4, device=robot.device).unsqueeze(0)
goal_pose[:, :3, 3] = torch.tensor(
    [[0.55, 0.30, 0.45]], device=robot.device
)
result = motion_generator.generate(
    [PlanState.from_xpos(goal_pose)],
    MotionGenOptions(
        start_qpos=robot.get_qpos(name="arm"),
        control_part="arm",
        plan_opts=CuroboPlanOptions(),
    ),
)
assert result.success.all()
~~~

## Atomic actions and supported scope

Single-arm MoveEndEffector is supported through the normal
`strategy="motion_gen"` route. MoveJoints can opt in to collision-aware
joint-space planning with `strategy="motion_gen"`; the action uses the planner
already owned by its MotionGenerator. Movement phases of PickUp, Place,
and MoveHeldObject can use the same single-arm static-world route.

This first release intentionally has the following limits:

- Only one configured control part is planned per request; coordinated dual-arm
  planning and CoordinatedPickment are unsupported.
- Collision worlds are generated from `RigidObject` meshes (cuboid/mesh/sphere)
  plus named dynamic pose updates. Arbitrary geometry insertion and removal at
  runtime are unsupported.
- The generated collision world assumes a fixed-base robot at the simulator
  origin. With a moving base, publish each relevant world obstacle as a named
  dynamic pose for every plan; automatic reprojection of static obstacles is
  unsupported.
- attached-object collision geometry, automatic attachment/detachment, and
  collision-aware carrying of a held object are unsupported.
- Non-control joints must remain at the matching cuRobo V2 `lock_joints`
  values. The adapter does not yet validate cross-model locked-joint name/value
  equivalence automatically.
- The legacy Gym ActionBank path is unsupported.
- CPU execution of cuRobo itself and cuRobo V1 compatibility are unsupported.
  CPU physics is supported because tensors are transferred to CUDA only for
  planning and the resulting trajectory is copied back to the simulation
  device.

## Demo

After installing cuRobo V2 and configuring a CUDA simulation environment, run
the Panda obstacle-avoidance demo from the repository root:

~~~bash
python examples/sim/planners/curobo_planner.py --headless --hold-steps 1 --step-repeat 1

# CPU physics with CUDA planning
python examples/sim/planners/curobo_planner.py --headless --sim-device cpu
~~~

The demo exports the DexSim `demo_block` into the cuRobo collision world via
`CuroboWorldCfg.rigid_objects` (the robot and world YAMLs are both
auto-generated), prints the result status and trajectory shape, then replays the
returned full-DoF trajectory. CUDA graph capture is enabled by default with the
renderer-compatible `"thread_local"` mode; pass `--no-cuda-graph` to disable it.
Headless runs
automatically record this fixed offscreen camera view to an MP4. Set an explicit
destination with `--record-save-path outputs/videos/curobo_demo.mp4`, adjust
the rate with `--record-fps`, or pass `--disable-record` to skip recording. See
[MotionGenerator](motion_generator.md) for the common planner interface.
