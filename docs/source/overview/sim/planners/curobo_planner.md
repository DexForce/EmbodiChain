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

Follow [NVIDIA's official cuRobo installation
guide](https://nvlabs.github.io/curobo/latest/getting-started/installation.html)
to install the V2 release that matches the CUDA driver and PyTorch environment.
The official flow clones cuRobo and uses a CUDA-matched extra:

~~~bash
git clone https://github.com/NVlabs/curobo.git
cd curobo
uv venv --python 3.11
source .venv/bin/activate

# Choose exactly one command for the installed CUDA/PyTorch environment.
uv pip install .[cu12]        # CUDA 12.x when PyTorch is already installed
uv pip install .[cu12-torch]  # CUDA 12.x fresh environment, installs PyTorch
uv pip install .[cu13]        # CUDA 13.x when PyTorch is already installed
uv pip install .[cu13-torch]  # CUDA 13.x fresh environment, installs PyTorch

python -c "import curobo; print(curobo.__version__)"
~~~

EmbodiChain does not install cuRobo transitively. Keep the cuRobo installation
in the same Python environment that runs the simulator, and use NVIDIA's
instructions when the CUDA or PyTorch version differs from this example.

## Configure a control part

Create one CuroboRobotProfileCfg for each EmbodiChain control part that may use
cuRobo. sim_to_curobo_joint_names is required for an explicit `robot_config_path`:
it maps the simulator's joint names to the names in the cuRobo V2 robot profile,
so no numeric joint ordering is assumed. Omit both `robot_config_path` and
`sim_to_curobo_joint_names` to auto-derive the whole profile from the robot's URDF
and solver (see [Auto-generated robot YAML](#auto-generated-robot-yaml) below).
Lock non-controlled joints in the cuRobo robot profile itself so
they are not exposed in the loaded planner's active joint list. To plan a
gripper or another extra active joint, define a control part that includes it.
The retained simulator value of every such joint must equal the corresponding
cuRobo V2 `lock_joints` value throughout planning and playback. Atomic actions
preserve non-control joints in their full-DoF trajectory; they do not infer or
drive the profile's locked joints. For example, the stock Panda V2 profile
locks both fingers at `0.04`, so use the same simulated finger state or include
the fingers in the planned control part. A mismatch means cuRobo validates a
different collision geometry from the one replayed in DexSim.

~~~python
from embodichain.lab.sim.planners import (
    CuroboPlannerCfg,
    CuroboRobotProfileCfg,
    CuroboWorldCfg,
    MotionGenCfg,
    MotionGenerator,
)

franka_profile = CuroboRobotProfileCfg(
    robot_config_path="franka.yml",
    sim_to_curobo_joint_names={
        f"fr3_joint{index}": f"panda_joint{index}" for index in range(1, 8)
    },
    base_link_name="panda_link0",
    tool_frame_name="panda_hand",
    # DexSim's Panda TCP is 103.4 mm along +Z from cuRobo's panda_hand.
    tool_frame_to_tcp=[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1034],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

planner_cfg = CuroboPlannerCfg(
    robot_uid="my_franka",
    planner_type="curobo",
    robot_profiles={"arm": franka_profile},
    world=CuroboWorldCfg(world_config_path="path/to/collision_world.yml"),
)
motion_generator = MotionGenerator(MotionGenCfg(planner_cfg=planner_cfg))
~~~

The robot configuration must be a cuRobo V2 robot profile with collision
spheres and self-collision data. Generate or update that profile with V2
RobotBuilder; a plain URDF alone is not sufficient for collision planning. The
mapped simulator joints must match the selected control part, and
tool_frame_name must name the cuRobo end-effector frame.

`base_link_name`, when supplied, is checked against the loaded cuRobo model.
The adapter automatically rebases simulator-world Cartesian goals and dynamic
obstacle poses through the live simulator control-part base, so parallel arena
offsets and a moved robot base are handled. If the simulator and cuRobo base
frames use different fixed conventions, set `sim_base_to_curobo_base` to the
transform from the simulator base to the cuRobo base. Static collision YAML is
always authored in the cuRobo base/world frame. `tool_frame_to_tcp` is
different: it converts an EmbodiChain TCP goal into the chosen cuRobo tool
frame. Omit it only when both frames are identical. By convention, the adapter
uses `T_curobo,X = T_curobo,sim_base @ inv(T_world,sim_base) @ T_world,X`.
It obtains the simulator base from the control part's IK solver root; if that
part intentionally has no local solver, provide `sim_base_link_name` in the
profile instead.

`CuroboPlannerCfg.use_cuda_graph` defaults to `False` for the same DexSim GPU
stream-safety reason. Enable it explicitly only after validating the local
simulation stack.

CuroboWorldCfg.world_config_path names an explicit collision world. The initial
release accepts cuRobo cuboid, mesh, and voxel geometry. If obstacle poses
change at runtime, declare their names in
CuroboWorldCfg.dynamic_obstacle_names, provision
CuroboWorldCfg.collision_cache before planning, and pass their batched
(B, 4, 4) poses through CuroboPlanOptions.dynamic_obstacle_poses. Geometry is
not extracted automatically from DexSim. With the default shared world
(`multi_env=False`), all batch rows must provide the same obstacle pose; set
`multi_env=True` when each environment needs its own collision-world instance
(for example, different dynamic obstacle poses). In that mode a single mapping
YAML (including the supplied demo scene) is cloned into one V2 scene per batch
row. A top-level YAML list may instead define one mapping per row; it must have
either one entry (cloned) or exactly the active batch size. An empty configured
world is likewise materialized once per row so its per-environment cache is
allocated. Dynamic pose updates still require the named geometry to already
exist in every scene; the adapter does not insert new geometry at runtime.

## Auto-generated robot YAML

A profile with the default `CuroboRobotProfileCfg()` (no `robot_config_path`, no
`sim_to_curobo_joint_names`) is fully auto-derived from the robot's URDF and
solver on the first plan, so nothing robot-specific needs to be hardcoded:

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

~~~python
planner_cfg = CuroboPlannerCfg(
    robot_uid="my_franka",
    robot_profiles={"arm": CuroboRobotProfileCfg()},  # auto-derived
    world=CuroboWorldCfg(world_config_path="path/to/collision_world.yml"),
)
~~~

Explicit profiles (`robot_config_path="franka.yml"` + `sim_to_curobo_joint_names`)
bypass auto-generation entirely and remain fully supported.

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
- Static cuboid, mesh, and voxel worlds plus named dynamic pose updates are
  supported. Automatic scene extraction, arbitrary geometry insertion, and
  removal are unsupported.
- A static collision YAML is for a fixed-base robot. With a moving base,
  publish each relevant world obstacle as a named dynamic pose for every plan;
  automatic reprojection of static YAML obstacles is unsupported.
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

The demo mirrors a cuboid in DexSim and the cuRobo collision world, prints the
result status and trajectory shape, then replays the returned full-DoF
trajectory. It disables cuRobo CUDA graph capture by default because graph
capture can conflict with DexSim GPU physics; pass `--cuda-graph` only after
validating that the local simulator stream setup supports it. Headless runs
automatically record this fixed offscreen camera view to an MP4. Set an explicit
destination with `--record-save-path outputs/videos/curobo_demo.mp4`, adjust
the rate with `--record-fps`, or pass `--disable-record` to skip recording. See
[MotionGenerator](motion_generator.md) for the common planner interface.
