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
* - ``get_object_pose``
  - Get the arena poses of objects. Returns 4x4 transformation matrices of shape (num_envs, 4, 4) by default, or (num_envs, 7) as [x, y, z, qw, qx, qy, qz] when ``to_matrix=False``. Returns zero tensor if object doesn't exist.
* - ``get_rigid_object_pose``
  - Get the arena poses of rigid objects. Returns 4x4 transformation matrices of shape (num_envs, 4, 4) by default, or (num_envs, 7) when ``to_matrix=False``. If the object doesn't exist, returns a zero tensor. (Deprecated: use ``get_object_pose`` instead.)
* - ``get_sensor_pose_in_robot_frame``
  - Transform sensor poses to robot coordinate frame. Returns 4x4 transformation matrices of shape (num_envs, 4, 4). For stereo cameras, supports selecting left or right camera pose via ``is_right`` parameter.
* - ``get_robot_eef_pose``
  - Get robot end-effector pose using forward kinematics. Returns 4x4 transformation matrices of shape (num_envs, 4, 4) by default, or (num_envs, 3) for position only when ``position_only=True``. Supports specifying ``part_name`` for different control parts.
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
  - Compute semantic masks from camera segmentation masks. Returns masks of shape (num_envs, height, width, 4) with channels for background, foreground, robot left-side, and robot right-side.
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
  - Normalize joint positions or velocities to a specified range based on joint limits. Supports both ``qpos_limits`` and ``qvel_limits``. Operates in ``modify`` mode. Default range is [0, 1].
```

## Object Properties

```{list-table} Object Properties Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``get_object_body_scale``
  - Get the body scale of objects. Returns tensor of shape (num_envs, 3). Only supports ``RigidObject``. Returns zero tensor if object doesn't exist.
* - ``get_rigid_object_velocity``
  - Get the world velocities (linear and angular) of rigid objects. Returns tensor of shape (num_envs, 6). Returns zero tensor if object doesn't exist.
* - ``get_rigid_object_physics_attributes``
  - Get physics attributes (mass, friction, damping, inertia) of rigid objects with caching. Returns a ``TensorDict`` containing: ``mass`` (num_envs, 1), ``friction`` (num_envs, 1), ``damping`` (num_envs, 1), ``inertia`` (num_envs, 3). Cache is cleared on environment reset. Implemented as a Functor class.
```

## Target / Goal

```{list-table} Target / Goal Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``target_position``
  - Get virtual target position from environment state. Reads target pose from ``env.{target_pose_key}`` (set by randomization events). Returns tensor of shape (num_envs, 3). Returns zeros if not yet initialized. Supports custom ``target_pose_key`` parameter.
```

```{currentmodule} embodichain.lab.sim.objects
```

```{note}
For custom observation needs, you can also use the robot's {meth}`~Robot.compute_fk()` method directly in your observation functors or task code.
```

## Usage Example

```python
from embodichain.lab.gym.envs.managers.cfg import ObservationCfg, SceneEntityCfg

# Example: Add object pose to observations
observations = {
    "object_pose": ObservationCfg(
        func="get_object_pose",
        mode="add",
        name="object/cube/pose",
        params={
            "entity_cfg": SceneEntityCfg(uid="cube"),
            "to_matrix": True,
        },
    ),
    # Example: Get object velocity
    "object_velocity": ObservationCfg(
        func="get_rigid_object_velocity",
        mode="add",
        name="object/cube/velocity",
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
            "range": [0.0, 1.0],
        },
    ),
    # Example: Get robot end-effector pose
    "eef_pose": ObservationCfg(
        func="get_robot_eef_pose",
        mode="add",
        name="robot/eef/pose",
        params={
            "part_name": "left_arm",
            "position_only": False,
        },
    ),
    # Example: Get object physics attributes
    "object_physics": ObservationCfg(
        func="get_rigid_object_physics_attributes",
        mode="add",
        name="object/cube/physics",
        params={
            "entity_cfg": SceneEntityCfg(uid="cube"),
        },
    ),
}
```
