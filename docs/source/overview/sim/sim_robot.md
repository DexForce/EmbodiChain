# Robot

```{currentmodule} embodichain.lab.sim
```

The {class}`~objects.Robot` class extends {class}`~objects.Articulation` to support advanced robotics features such as kinematic solvers (IK/FK), motion planners, and part-based control (e.g., controlling "arm" and "gripper" separately).

## Configuration

Robots are configured using {class}`~cfg.RobotCfg`.
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `control_parts` | `Dict[str, List[str]]` | `None` | Defines groups of joints (e.g., `{"arm": ["joint1", ...], "hand": ["finger1", ...]}`). |
| `solver_cfg` | `SolverCfg` | `None` | Configuration for kinematic solvers (IK/FK). |
| `urdf_cfg` | `URDFCfg` | `None` | Advanced configuration for assembling a robot from multiple URDF components. |

### Setup & Initialization

A `Robot` must be spawned within a `SimulationManager`.

```python
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RobotCfg
from embodichain.lab.sim.solvers import SolverCfg

# 1. Initialize Simulation Environment
# Note: Use 'device' to specify device (e.g., "cuda:0" or "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
sim_cfg = SimulationManagerCfg(device=device, physics_dt=0.01)
sim = SimulationManager(sim_config=sim_cfg)

# 2. Configure Robot
robot_cfg = RobotCfg(
    fpath="assets/robots/franka/franka.urdf",
    control_parts={
        "arm": ["panda_joint[1-7]"],
        "gripper": ["panda_finger_joint[1-2]"]
    },
    solver_cfg=SolverCfg() 
)

# 3. Spawn Robot
# Note: The method is 'add_robot', and it takes 'cfg' as argument
robot: Robot = sim.add_robot(cfg=robot_cfg)

# 4. Reset Simulation
# This performs a global reset of the simulation state
sim.reset_objects_state()
```

## Robot Class

### Control Parts

A unique feature of the `Robot` class is **Control Parts**. Instead of controlling the entire DoF vector at once, you can target specific body parts by name.

```python
# Get joint IDs for a specific part
arm_ids = robot.get_joint_ids(name="arm")

# Control only the arm
# Note: Ensure 'sim.update()' is called in your loop to apply these actions
target_qpos = torch.zeros((robot.num_instances, len(arm_ids)), device=device)
robot.set_qpos(target_qpos, name="arm", target=True)
```

### Kinematics (Solvers)
The robot class integrates solvers to perform differentiable Forward Kinematics (FK) and Inverse Kinematics (IK).
#### Forward Kinematics (FK)
Compute the pose of a link (e.g., end-effector) given joint positions.
```python
# Compute FK for a specific part (uses the part's configured solver)
current_qpos = robot.get_qpos()
ee_pose = robot.compute_fk(qpos=current_qpos, name="arm")
print(f"EE Pose: {ee_pose}")
```
#### Inverse Kinematics (IK)
Compute the required joint positions to reach a target pose.
```python 
# Compute IK
# pose: Target pose (N, 7) or (N, 4, 4)
target_pose = ee_pose.clone() # Example target
target_pose[:, 2] += 0.1      # Move up 10cm

success, solved_qpos = robot.compute_ik(
    pose=target_pose,
    name="arm",
    joint_seed=current_qpos
)
```
### Proprioception
Get standard proprioceptive observation data for learning agents.
```python
# Returns a dict containing 'qpos', 'qvel', and 'qf'
obs = robot.get_proprioception()
```
### Advanced API
The Robot class overrides standard Articulation methods to support the name argument for part-specific operations. 
| Method | Description |
| :--- | :--- |
| `set_qpos(..., name="part")` | Set joint positions for a specific part. |
| `set_qvel(..., name="part")` | Set joint velocities for a specific part. |
| `set_qf(..., name="part")` | Set joint efforts for a specific part. |
| `get_qpos(name="part")` | Get joint positions of a specific part. |
| `get_qvel(name="part")` | Get joint velocities of a specific part. |
| `set_qpos_limits(..., name="part")` | Set physical joint position limits for a specific part. |
| `get_qpos_limits(name="part")` | Get physical joint position limits of a specific part. |

```python
# Tighten the left arm physical workspace
left_arm_limits = robot.get_qpos_limits(name="left_arm")
left_arm_limits[..., 0] += 0.05  # tighten lower bound
left_arm_limits[..., 1] -= 0.05  # tighten upper bound
robot.set_qpos_limits(left_arm_limits, name="left_arm")
```

For solver-only planning limits, configure `SolverCfg.user_qpos_limits` on the
robot's solver config. Robot solver synchronization will intersect those solver
limits with the robot's current effective `qpos_limits`.

For more API details, refer to the {class}`~objects.Robot` documentation.
