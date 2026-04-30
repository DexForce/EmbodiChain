# Action Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available action terms that can be used with the Action Manager. Action terms are configured using {class}`~cfg.ActionTermCfg` and are responsible for processing raw actions from the policy and converting them to the format expected by the robot (e.g., qpos, qvel, qf).

````{tip}
**Using an AI coding agent?** Use the **`/add-functor`** skill to scaffold a new action term with the correct class structure, `ActionTermCfg` registration, and module placement in `actions.py`.
````

## Joint Position Control

```{list-table} Joint Position Action Terms
:header-rows: 1
:widths: 25 75

* - Action Term
  - Description
* - {class}`~actions.DeltaQposTerm`
  - Delta joint position action: current_qpos + scale * action -> qpos. The policy outputs position deltas relative to the current joint positions.

    ```json
    {"func": "DeltaQposTerm", "params": {"scale": 0.1}}
    ```
* - {class}`~actions.QposTerm`
  - Absolute joint position action: scale * action -> qpos. The policy outputs direct target joint positions.

    ```json
    {"func": "QposTerm", "params": {"scale": 1.0}}
    ```
* - {class}`~actions.QposDenormalizedTerm`
  - Normalized action in [-1, 1] -> denormalize to joint limits -> qpos. The policy outputs normalized values that are mapped to joint limits. With scale=1.0 (default), action in [-1, 1] maps to [low, high].

    ```json
    {"func": "QposDenormalizedTerm", "params": {"scale": 1.0}}
    ```
* - {class}`~actions.QposNormalizedTerm`
  - Normalize action from qpos limits -> [range[0], range[1]]. Maps joint positions to a normalized range based on joint limits. Typically used for post-processing action outputs.

    ```json
    {"func": "QposNormalizedTerm", "params": {"range": [0.0, 1.0]}}
    ```
```

## End-Effector Control

```{list-table} End-Effector Action Terms
:header-rows: 1
:widths: 25 75

* - Action Term
  - Description
* - {class}`~actions.EefPoseTerm`
  - End-effector pose (6D or 7D) -> IK -> qpos. The policy outputs target end-effector poses which are converted to joint positions via inverse kinematics. Returns ``ik_success`` in the output so reward/observation can penalize or condition on IK failures. Supports both 6D (euler angles) and 7D (quaternion) pose representations.

    ```json
    {"func": "EefPoseTerm", "params": {"scale": 0.1, "pose_dim": 7}}
    ```
```

## Velocity and Force Control

```{list-table} Velocity and Force Action Terms
:header-rows: 1
:widths: 25 75

* - Action Term
  - Description
* - {class}`~actions.QvelTerm`
  - Joint velocity action: scale * action -> qvel. The policy outputs target joint velocities.

    ```json
    {"func": "QvelTerm", "params": {"scale": 1.0}}
    ```
* - {class}`~actions.QfTerm`
  - Joint force/torque action: scale * action -> qf. The policy outputs target joint torques/forces.

    ```json
    {"func": "QfTerm", "params": {"scale": 1.0}}
    ```
```

## Usage Example

```python
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg

# Example: Delta joint position control
actions = {
    "joint_position": ActionTermCfg(
        func="DeltaQposTerm",
        params={
            "scale": 0.1,  # Scale factor for action deltas
        },
    ),
}

# Example: Normalized joint position control
actions = {
    "normalized_joint_position": ActionTermCfg(
        func="QposDenormalizedTerm",
        params={
            "scale": 1.0,  # Full joint range utilization
        },
    ),
}

# Example: Normalize qpos to [0, 1] range (for post-processing)
actions = {
    "normalize_qpos": ActionTermCfg(
        func="QposNormalizedTerm",
        params={
            "range": [0.0, 1.0],  # Normalize to [0, 1] range
        },
    ),
}

# Example: End-effector pose control
actions = {
    "eef_pose": ActionTermCfg(
        func="EefPoseTerm",
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
