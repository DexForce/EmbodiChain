# Observation Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available observation functors that can be used with the Observation Manager. Observation functors are configured using {class}`~cfg.ObservationCfg` and can operate in two modes: ``modify`` (update existing observations) or ``add`` (add new observations).

## Pose Computations

```{list-table} Pose Computation Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``get_rigid_object_pose``
  - Get world poses of rigid objects. Returns 4x4 transformation matrices of shape (num_envs, 4, 4). If the object doesn't exist, returns a zero tensor.
* - ``get_sensor_pose_in_robot_frame``
  - Transform sensor poses to robot coordinate frame. Returns pose as [x, y, z, qw, qx, qy, qz] of shape (num_envs, 7).
```

## Sensor Information

```{list-table} Sensor Information Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``get_sensor_intrinsics``
  - Get the intrinsic matrix of a camera sensor. Returns 3x3 intrinsic matrices of shape (num_envs, 3, 3). For stereo cameras, supports selecting left or right camera intrinsics.
* - ``compute_semantic_mask``
  - Compute semantic masks from camera segmentation masks. Returns masks of shape (num_envs, height, width, 3) with channels for robot, background, and foreground objects.
```

## Keypoint Projections

```{list-table} Keypoint Projection Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``compute_exteroception``
  - Project 3D keypoints (affordance poses, robot parts) onto camera image planes. Supports multiple sources: affordance poses from objects (e.g., grasp poses, place poses) and robot control part poses (e.g., end-effector positions). Returns normalized 2D coordinates. Implemented as a Functor class.
```

## Normalization

```{list-table} Normalization Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``normalize_robot_joint_data``
  - Normalize joint positions or velocities to [0, 1] range based on joint limits. Supports both ``qpos_limits`` and ``qvel_limits``. Operates in ``modify`` mode.
```

```{currentmodule} embodichain.lab.sim.objects
```

```{note}
To get robot end-effector poses, you can use the robot's {meth}`~Robot.compute_fk()` method directly in your observation functors or task code.
```

## Usage Example

```python
from embodichain.lab.gym.envs.managers.cfg import ObservationCfg, SceneEntityCfg

# Example: Add object pose to observations
observations = {
    "object_pose": ObservationCfg(
        func="get_rigid_object_pose",
        mode="add",
        name="object/cube/pose",
        params={
            "entity_cfg": SceneEntityCfg(uid="cube"),
        },
    ),
    # Example: Normalize joint positions
    "normalized_qpos": ObservationCfg(
        func="normalize_robot_joint_data",
        mode="modify",
        name="robot/qpos",
        params={
            "joint_ids": list(range(7)),  # First 7 joints
            "limit": "qpos_limits",
        },
    ),
}
```
