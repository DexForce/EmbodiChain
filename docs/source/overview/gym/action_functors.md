# Action Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available action terms that can be used with the Action Manager. Action terms are configured using {class}`~cfg.ActionTermCfg` and are responsible for processing raw actions from the policy and converting them to the format expected by the robot (e.g., qpos, qvel, qf).

## Joint Position Control

```{list-table} Joint Position Action Terms
:header-rows: 1
:widths: 30 70

* - Action Term
  - Description
* - ``DeltaQposTerm``
  - Delta joint position action: current_qpos + scale * action -> qpos. The policy outputs position deltas relative to the current joint positions.
* - ``QposTerm``
  - Absolute joint position action: scale * action -> qpos. The policy outputs direct target joint positions.
* - ``QposNormalizedTerm``
  - Normalized action in [-1, 1] -> denormalize to joint limits -> qpos. The policy outputs normalized values that are mapped to joint limits. With scale=1.0 (default), action in [-1, 1] maps to [low, high].
```

## End-Effector Control

```{list-table} End-Effector Action Terms
:header-rows: 1
:widths: 30 70

* - Action Term
  - Description
* - ``EefPoseTerm``
  - End-effector pose (6D or 7D) -> IK -> qpos. The policy outputs target end-effector poses which are converted to joint positions via inverse kinematics. Returns ``ik_success`` in the output so reward/observation can penalize or condition on IK failures. Supports both 6D (euler angles) and 7D (quaternion) pose representations.
```

## Velocity and Force Control

```{list-table} Velocity and Force Action Terms
:header-rows: 1
:widths: 30 70

* - Action Term
  - Description
* - ``QvelTerm``
  - Joint velocity action: scale * action -> qvel. The policy outputs target joint velocities.
* - ``QfTerm``
  - Joint force/torque action: scale * action -> qf. The policy outputs target joint torques/forces.
```

## Usage Example

```python
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg

# Example: Delta joint position control
actions = {
    "joint_position": ActionTermCfg(
        func="embodichain.lab.gym.envs.managers.action_manager.DeltaQposTerm",
        params={
            "scale": 0.1,  # Scale factor for action deltas
        },
    ),
}

# Example: Normalized joint position control
actions = {
    "normalized_joint_position": ActionTermCfg(
        func="embodichain.lab.gym.envs.managers.action_manager.QposNormalizedTerm",
        params={
            "scale": 1.0,  # Full joint range utilization
        },
    ),
}

# Example: End-effector pose control
actions = {
    "eef_pose": ActionTermCfg(
        func="embodichain.lab.gym.envs.managers.action_manager.EefPoseTerm",
        params={
            "scale": 0.1,
            "pose_dim": 7,  # 7D (position + quaternion)
        },
    ),
}
```

## Action Term Properties

All action terms provide the following properties:

- ``action_dim``: The dimension of the action space (number of values the policy should output)
- ``process_action(action)``: Method to convert raw policy output to robot control format

The Action Manager also provides:

- ``total_action_dim``: Total dimension of all action terms combined
- ``action_type``: The active action type (term name) for backward compatibility
