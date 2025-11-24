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

from embodichain.lab.sim.planners.motion_generator import MotionGenerator
from embodichain.lab.sim.objects.robot import Robot
from embodichain.lab.sim.solvers.pink_solver import PinkSolverCfg
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod

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

motion_gen = MotionGenerator(
    robot=robot,
    uid="arm",
    planner_type="toppra",
    default_velocity=0.2,
    default_acceleration=0.5
)

```

### Trajectory Planning

#### Joint Space Planning

```python
current_state = {
    "position": [0, 0, 0, 0, 0, 0],
    "velocity": [0, 0, 0, 0, 0, 0],
    "acceleration": [0, 0, 0, 0, 0, 0]
}
target_states = [
    {"position": [1, 1, 1, 1, 1, 1]}
]
success, positions, velocities, accelerations, times, duration = motion_gen.plan(
    current_state=current_state,
    target_states=target_states,
    sample_method=TrajectorySampleMethod.TIME,
    sample_interval=0.01
)
```

#### Cartesian or Joint Interpolation

```python
# Using joint configurations (qpos_list)
qpos_list = [
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1]
]
out_qpos_list, out_xpos_list = motion_gen.create_discrete_trajectory(
    qpos_list=qpos_list,
    is_linear=False,
    sample_method=TrajectorySampleMethod.QUANTITY,
    sample_num=20
)
```

### Estimating Trajectory Sample Count

You can estimate the number of sampling points required for a trajectory before generating it:

```python
# Estimate based on joint configurations (qpos_list)
qpos_list = [
    [0, 0, 0, 0, 0, 0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [1, 1, 1, 1, 1, 1]
]
sample_count = motion_gen.estimate_trajectory_sample_count(
    qpos_list=qpos_list,  # List of joint positions
    step_size=0.01, # unit: m
    angle_step=0.05, # unit: rad
)
print(f"Estimated sample count: {sample_count}")
```

## Notes

* The planner type can be specified as a string or `PlannerType` enum.
* If the robot provides its own joint limits, those will be used; otherwise, default or user-specified limits are applied.
* For Cartesian interpolation, inverse kinematics (IK) is used to compute joint configurations for each interpolated pose.
* The class is designed to be extensible for additional planners and collision checking in the future.
* The sample count estimation is useful for predicting computational load and memory requirements.
