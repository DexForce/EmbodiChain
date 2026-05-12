# MotionGenerator

`MotionGenerator` provides a unified interface for robot trajectory planning, supporting both joint space and Cartesian space interpolation. It is designed to work with different planners (such as ToppraPlanner) and can be extended to support collision checking in the future.

## Features

* **Unified planning interface**: Supports trajectory planning with or without collision checking (collision checking is reserved for future implementation).
* **Flexible planner selection**: Allows selection of different planners (currently supports TOPPRA for time-optimal planning).
* **Automatic constraint handling**: Retrieves velocity and acceleration limits from the robot or uses user-specified/default values.
* **Supports both joint and Cartesian interpolation**: Generates discrete trajectories using either joint space or Cartesian space interpolation.
* **Convenient sampling**: Supports various sampling strategies via `TrajectorySampleMethod`.

## Usage

### Initialization

```python
from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
)

from embodichain.lab.sim.planners.motion_generator import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
from embodichain.lab.sim.objects.robot import Robot
from embodichain.lab.sim.solvers.pink_solver import PinkSolverCfg
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod, PlanState, MoveType
from embodichain.lab.sim.planners.motion_generator import MotionGenOptions

# Configure the simulation
sim_cfg = SimulationManagerCfg(
    width=1920,
    height=1080,
    physics_dt=1.0 / 100.0,
    sim_device="cpu",
)

sim = SimulationManager(sim_cfg)

# Get UR10 URDF path
urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")

# Create UR10 robot
robot_cfg = RobotCfg(
    uid="UR10_test",
    urdf_cfg=URDFCfg(
        components=[{"component_type": "arm", "urdf_path": urdf_path}]
    ),
    control_parts={"arm": ["Joint[1-6]"]},
    solver_cfg={
        "arm": PinkSolverCfg(
            urdf_path=urdf_path,
            end_link_name="ee_link",
            root_link_name="base_link",
            pos_eps=1e-2,
            rot_eps=5e-2,
            max_iterations=300,
            dt=0.1,
        )
    },
    drive_pros=JointDrivePropertiesCfg(
        stiffness={"Joint[1-6]": 1e4},
        damping={"Joint[1-6]": 1e3},
    ),
)
robot = sim.add_robot(cfg=robot_cfg)

# Constraints are now specified in ToppraPlanOptions, not in ToppraPlannerCfg
motion_gen = MotionGenerator(
    cfg=MotionGenCfg(
        planner_cfg=ToppraPlannerCfg(
            robot_uid="UR10_test",
        )
    )
)
```

### Trajectory Planning

#### Joint Space Planning

```python
# Create options with constraints and planning parameters
plan_opts = ToppraPlanOptions(
    constraints={
        "velocity": 0.2,
        "acceleration": 0.5,
    },
    sample_method=TrajectorySampleMethod.TIME,
    sample_interval=0.01
)

# Create motion generation options
motion_opts = MotionGenOptions(
    plan_opts=plan_opts,
    control_part="arm",
    is_interpolate=False,
)

# Use generate() method instead of plan()
target_states = [
    PlanState(move_type=MoveType.JOINT_MOVE, qpos=torch.tensor([1, 1, 1, 1, 1, 1]))
]
result = motion_gen.generate(
    target_states=target_states,
    options=motion_opts
)
```

#### Cartesian Space Planning

```python
import torch
import numpy as np

# Create options with constraints
plan_opts = ToppraPlanOptions(
    constraints={
        "velocity": 0.2,
        "acceleration": 0.5,
    },
    sample_method=TrajectorySampleMethod.TIME,
    sample_interval=0.01
)

# Create motion generation options with interpolation for smoother Cartesian motion
motion_opts = MotionGenOptions(
    plan_opts=plan_opts,
    control_part="arm",
    is_interpolate=True,  # Enable pre-interpolation for Cartesian moves
    interpolate_nums=10,   # Number of points between each waypoint
    is_linear=True,        # Linear interpolation in Cartesian space
)

# Define target poses as 4x4 transformation matrices
# Each matrix is [position(3), orientation(3x3)] in row-major order
target_pose_1 = torch.eye(4)
target_pose_1[:3, 3] = torch.tensor([0.5, 0.3, 0.4])  # position

target_pose_2 = torch.eye(4)
target_pose_2[:3, 3] = torch.tensor([0.6, 0.4, 0.3])  # another position

# Use EEF_MOVE for Cartesian space planning
target_states = [
    PlanState(move_type=MoveType.EEF_MOVE, xpos=target_pose_1),
    PlanState(move_type=MoveType.EEF_MOVE, xpos=target_pose_2),
]

result = motion_gen.generate(
    target_states=target_states,
    options=motion_opts
)
```


### Estimating Trajectory Sample Count

You can estimate the number of sampling points required for a trajectory before generating it:

```python
# Estimate based on joint configurations (qpos_list)
qpos_list = torch.as_tensor([
    [0, 0, 0, 0, 0, 0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [1, 1, 1, 1, 1, 1]
])
sample_count = motion_gen.estimate_trajectory_sample_count(
    qpos_list=qpos_list,  # List of joint positions
    step_size=0.01, # unit: m
    angle_step=0.05, # unit: rad
    control_part="arm",
)
print(f"Estimated sample count: {sample_count}")
```

## Notes

* The planner type can be specified as a string or `PlannerType` enum.
* If the robot provides its own joint limits, those will be used; otherwise, default or user-specified limits are applied.
* For Cartesian interpolation, inverse kinematics (IK) is used to compute joint configurations for each interpolated pose.
* The class is designed to be extensible for additional planners and collision checking in the future.
* The sample count estimation is useful for predicting computational load and memory requirements.
