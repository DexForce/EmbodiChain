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

EmbodiChain exposes cuRobo V2 as CUDA-matched optional dependencies. From the
EmbodiChain repository root, select exactly one extra:

~~~bash
# Recommended for the normal EmbodiChain environment, where PyTorch is present.
uv pip install ".[curobo-cu12]"  # CUDA 12.x
uv pip install ".[curobo-cu13]"  # CUDA 13.x

# For a fresh environment that also needs PyTorch.
uv pip install ".[curobo-cu12-torch]"  # CUDA 12.x
uv pip install ".[curobo-cu13-torch]"  # CUDA 13.x

python -c "import curobo; print(curobo.__version__)"
pytest --pyargs curobo.tests
~~~

The extras follow [NVIDIA's official cuRobo installation
guide](https://nvlabs.github.io/curobo/latest/getting-started/installation.html)
and pin the source dependency to the cuRobo V2 `v0.8.0` release. Use a Python
3.10--3.13 environment on Linux with a supported NVIDIA GPU and driver. The
non-`torch` extras are preferred for EmbodiChain because the simulation
environment normally already provides PyTorch; the `-torch` variants delegate
the PyTorch version requirement to cuRobo. Keep cuRobo in the same Python
environment that runs the simulator.

## Configure a control part

The cuRobo robot model and the per-control-part profile are both auto-generated
internally - no external cuRobo robot YAML (e.g. `franka.yml`) and no
`robot_profiles` config are needed. On the first plan, the adapter fits collision
spheres to each link of the robot's URDF and writes a cuRobo V2 robot YAML (see
[Auto-generated robot YAML](#auto-generated-robot-yaml)). The tool frame, TCP
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

~~~python
from embodichain.lab.sim.planners import (
    CuroboPlannerCfg,
    CuroboWorldCfg,
    MotionGenCfg,
    MotionGenerator,
)

planner_cfg = CuroboPlannerCfg(
    robot_uid="my_franka",
    planner_type="curobo",
    world=CuroboWorldCfg(rigid_objects=[demo_block]),
)
motion_generator = MotionGenerator(MotionGenCfg(planner_cfg=planner_cfg))
~~~

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

`CuroboPlannerCfg.use_cuda_graph` defaults to `False` for the same DexSim GPU
stream-safety reason. Enable it explicitly only after validating the local
simulation stack.

The collision world is always auto-generated from live `RigidObject` meshes via
`CuroboWorldCfg.rigid_objects`: the adapter reads each object's mesh
(`get_vertices` / `get_triangles`) and world pose (`get_local_pose`) and writes a
cached cuRobo scene YAML on the first plan, using
`CuroboWorldCfg.obstacle_representation` (`"cuboid"` by default - a local-frame
AABB placed as an OBB via the object pose; also `"mesh"` for the exact triangle
mesh, or `"sphere"` to fit spheres with cuRobo's `fit_spheres_to_mesh`).
Generated poses are authored in the cuRobo base/world frame, so this is exact
when the robot base sits at the simulator world origin. For obstacles that move
or live in an offset base frame, also declare their names in
`CuroboWorldCfg.dynamic_obstacle_names` and update poses at plan time through
`CuroboPlanOptions.dynamic_obstacle_poses` (provision
`CuroboWorldCfg.collision_cache` before planning). With the default shared world
(`multi_env=False`), all batch rows must provide the same obstacle pose; set
`multi_env=True` when each environment needs its own collision-world instance
(for example, different dynamic obstacle poses). In that mode the generated world
YAML is cloned into one V2 scene per batch row. An empty world (`rigid_objects`
left `None`) is likewise materialized once per row so its per-environment cache
is allocated. Dynamic pose updates still require the named geometry to already
exist in every scene; the adapter does not insert new geometry at runtime.

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
receive the original pose and preserves its own collision-checked samples.

~~~python
import torch

from embodichain.lab.sim.planners import MotionGenOptions, PlanState
from embodichain.lab.sim.planners.curobo_planner import CuroboPlanOptions

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
motion_source="motion_gen" route. MoveJoints can opt in to collision-aware
joint-space planning with motion_source="motion_gen" and
planner_type="curobo". Movement phases of PickUp, Place, Press, and
MoveHeldObject can use the same single-arm static-world route.

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
- CPU execution and cuRobo V1 compatibility are unsupported.

## Demo

After installing cuRobo V2 and configuring a CUDA simulation environment, run
the Panda obstacle-avoidance demo from the repository root:

~~~bash
python examples/sim/planners/curobo_planner.py --headless --hold-steps 1 --step-repeat 1
~~~

The demo exports the DexSim `demo_block` into the cuRobo collision world via
`CuroboWorldCfg.rigid_objects` (the robot and world YAMLs are both
auto-generated), prints the result status and trajectory shape, then replays the
returned full-DoF trajectory. It disables cuRobo CUDA graph capture by default because graph
capture can conflict with DexSim GPU physics; pass `--cuda-graph` only after
validating that the local simulator stream setup supports it. Headless runs
automatically record this fixed offscreen camera view to an MP4. Set an explicit
destination with `--record-save-path outputs/videos/curobo_demo.mp4`, adjust
the rate with `--record-fps`, or pass `--disable-record` to skip recording. See
[MotionGenerator](motion_generator.md) for the common planner interface.
