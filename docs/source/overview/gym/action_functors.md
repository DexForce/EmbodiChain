# Action Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

Action terms define an ordered flat policy interface. `ActionManager` validates
the complete `(num_envs, action_dim)` floating tensor, slices it in configuration
order, calls each term's `process_actions()`, and later calls `apply_actions()`
once before physics. Each term writes only its selected non-mimic joints.

## Quick Reference

- Configure each term with {class}`~cfg.ActionTermCfg`; action terms have no
  `mode` field.
- The config attribute name becomes the stable term name and descriptor name.
- The policy width is the sum of term `action_dim` values in config order.
- Terms writing the same command type cannot own overlapping joint IDs.
- `ControllerAction` is a separate controller-ready path and bypasses the
  Action Manager.

````{tip}
**Using an AI coding agent?** Use the **`/add-functor`** skill to scaffold a new
action term with the project protocol and tests.
````

## Built-in Actions

```{list-table}
:header-rows: 1
:widths: 30 70

* - Action
  - Policy and command mapping
* - {class}`~actions.JointPositionAction`
  - Absolute selected-joint position targets with optional scale and offset.
* - {class}`~actions.JointPositionToLimitsAction`
  - Per-joint normalized `[-1, 1]` values mapped to selected joint limits.
* - {class}`~actions.RelativeJointPositionAction`
  - Scaled offsets added to the current selected-joint positions.
* - {class}`~actions.DefaultJointPositionAction`
  - Normalized locomotion offsets around a configured default pose, with
    current/previous raw buffers and encoder bias.
* - {class}`~actions.JointVelocityAction`
  - Scaled selected-joint velocity targets.
* - {class}`~actions.JointEffortAction`
  - Scaled selected-joint force or torque targets.
* - {class}`~actions.EefPoseAction`
  - A selected arm's absolute `xyz_rpy` or `xyz_quat_xyzw` target converted by
    IK to selected-arm joint positions. Failed rows hold measured arm qpos.
* - {class}`~actions.ParallelGripperAction`
  - One normalized scalar mapped continuously or binarily to explicit
    independent gripper-joint commands.
```

## Configuration Examples

Relative joint control:

```yaml
actions:
  arm_action:
    func: RelativeJointPositionAction
    params: {part_name: arm, scale: 0.1}
```

Composed Cartesian arm and parallel gripper control:

```yaml
control_parts: [arm, hand]
actions:
  arm_action:
    func: EefPoseAction
    params: {part_name: arm, pose_representation: xyz_rpy}
  gripper_action:
    func: ParallelGripperAction
    params:
      part_name: hand
      command_mode: continuous
      use_joint_limits: true
```

The second configuration has seven policy dimensions: the six-dimensional arm
slice followed by the one-dimensional gripper slice. A joint-position arm with
seven independent joints plus the same gripper term has eight dimensions.

## Action Term Contract

Every action term exposes:

- `action_dim` and a one-dimensional Gymnasium `Box`;
- `raw_actions` and `processed_actions` buffers;
- `command_type` and ordered `controlled_joint_ids`;
- an immutable `ActionTermDescriptor` describing names, units, normalization,
  controlled joints, and metadata;
- `process_actions(actions)`, `apply_actions()`, and selected-row `reset()`.

The manager binds term descriptors to flat slices as `ActionDescriptor` values.
Dataset policy contracts use these descriptors directly; they do not infer
semantics from class names.
