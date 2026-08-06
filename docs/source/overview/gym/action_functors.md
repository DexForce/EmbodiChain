# Action Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available action terms that can be used with the Action Manager. Action terms are configured using {class}`~cfg.ActionTermCfg` and are responsible for processing raw actions from the policy and converting them to the format expected by the robot (e.g., qpos, qvel, qf).

## Quick Reference

- Use this page when defining the policy-facing action space for RL or closed-loop control tasks.
- Action terms are configured with {class}`~cfg.ActionTermCfg`.
- Action terms transform one batched policy action into robot control commands for all environments.
- Check each term's ``action_dim`` to determine how many values the policy must output.

````{tip}
**Using an AI coding agent?** Use the **`/add-functor`** skill to scaffold a new action term with the correct class structure, `ActionTermCfg` registration, and module placement in `actions.py`.
````

## Policy and Command Contract

The Action Manager exposes one flat {class}`gymnasium.spaces.Box` to the
policy. Each ``pre`` term owns a contiguous slice in configuration order. The
manager processes those slices into typed ``qpos``, ``qvel``, or ``qf``
commands and applies each command to that term's selected joints.

By default, every term's policy range is ``[-1, 1]``. Use ``scale`` to map that
normalized value to a useful physical magnitude, or use
{class}`~actions.QposDenormalizedTerm` to map the full normalized range to
joint position limits. Physical commands are clipped to the robot's qpos,
qvel, or qf limits by default.

The following parameters are common to pre-processing terms:

- ``joint_ids``: Static active-joint indices controlled by the term.
- ``control_part``: Named robot control part; use this instead of
  ``joint_ids``.
- ``action_range``: Two finite policy-space bounds. The default is
  ``[-1, 1]``. ``QposDenormalizedTerm`` uses its existing ``range`` parameter.
- ``clip``: Clip the processed physical command to robot limits. Defaults to
  ``true``.
- ``allow_overlap``: Permit two terms to address the same joint only when it
  is explicitly ``true`` on both terms. Disjoint groups are recommended
  because overlapping position, velocity, and effort semantics are otherwise
  ambiguous.

Flat tensors are the standard RL interface. A mapping may also address terms
by their configuration names, which is useful for scripted controllers:

```python
env.step({
    "arm_velocity": arm_velocity_action,
    "gripper_effort": gripper_effort_action,
})
```

Without an Action Manager, ``env.step(tensor)`` remains a qpos command for
backward compatibility. Direct typed commands are also accepted:

```python
env.step({"qvel": target_velocity})
env.step({"qf": target_effort})
```

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
    {"func": "QposNormalizedTerm", "mode": "post", "params": {"range": [0.0, 1.0]}}
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
  - Joint velocity action: scale * action -> qvel. The policy outputs target joint velocities. Configure zero position stiffness on these joints when a position drive should not oppose the velocity target.

    ```json
    {"func": "QvelTerm", "params": {"scale": 1.0}}
    ```
* - {class}`~actions.QfTerm`
  - Joint force/torque action: scale * action -> qf. The policy outputs target joint torques/forces. The command is reapplied before every physics substep so it is held across control decimation.

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
        mode="post",
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

# Example: one flat RL action controlling disjoint joint groups
actions = {
    "arm_velocity": ActionTermCfg(
        func="QvelTerm",
        params={
            "joint_ids": [0, 1, 2, 3, 4, 5],
            "scale": 1.5,
        },
    ),
    "gripper_effort": ActionTermCfg(
        func="QfTerm",
        params={
            "control_part": "gripper",
            "scale": 20.0,
        },
    ),
}
```

For the mixed example, the policy outputs
``[arm_velocity..., gripper_effort...]``. The environment retains that exact
flat action in RL and trajectory buffers while routing the processed commands
to ``set_qvel`` and ``set_qf`` respectively.

````{attention}
Velocity and effort commands do not automatically change the robot's drive
configuration. A qvel term on joints with non-zero stiffness may fight the
position drive. A qf term with non-zero stiffness or damping is additive to
the active drive rather than pure torque control. The Action Manager emits a
warning for these combinations; configure the robot drive properties to match
the intended control mode.
````

## Action Term Properties

All action terms provide the following properties:

- ``action_dim``: The dimension of the action space (number of values the policy should output)
- ``action_space``: The per-term policy-space bounds
- ``joint_ids``: The resolved robot joints controlled by the term
- ``command_key``: The physical output type (``qpos``, ``qvel``, or ``qf``)
- ``process_action(action)``: Method to convert raw policy output to robot control format
