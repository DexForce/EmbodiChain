# Built-in atomic actions

All built-ins implement `plan(invocation, context) -> ActionPlan`. Motion and
recovery settings belong to the invocation rather than each action config.

| Skill ID | Goal | Semantic roles | Expected task effect |
|---|---|---|---|
| `move_end_effector` | `EndEffectorPoseGoal` | manipulator `primary` | none |
| `move_joints` | `JointPositionGoal` or `NamedJointPositionGoal` | manipulator `primary` | none |
| `pick_up` | `GraspGoal` | manipulator/end effector `primary` | attach object |
| `move_held_object` | `HeldObjectPoseGoal` | manipulator/end effector `primary` | none |
| `place` | `PlaceGoal` or `AssembleGoal` | manipulator/end effector `primary` | detach object |
| `press` | `PressGoal` | manipulator/end effector `primary` | none |
| `coordinated_pickment` | `CoordinatedPickGoal` | `left`, `right` | create coordinated attachment |
| `coordinated_placement` | `CoordinatedPlacementGoal` | `placing`, `support` | update/remove attachments |
| `hand_over` | `GraspGoal` | `source`, `destination` | transfer attachment |

## Pose goals

Pose-valued goals accept explicit tensors. Selected goals also accept a
`SceneEntityPose(entity_id, relative_pose=...)`, which is resolved from every
new `SceneSnapshot` during planning or replanning. Explicit tensors may use
`(4, 4)` or `(B, 4, 4)` shapes; waypoint-capable goals additionally accept
`(B, N, 4, 4)`.

## Configuration boundaries

Action configs contain implementation-owned behavior such as gripper open/close
positions, phase split counts, lift distance, and grasp-selection constraints.
They do not contain planner choice, motion source, trajectory sample count,
velocity limits, recovery budgets, or dynamic-goal thresholds. Those reusable
choices live in `MotionPolicy` and `RecoveryPolicy`.

`MoveJoints` and `MoveEndEffector` resolve the concrete manipulator entirely from
`ActionBinding`. Complex manipulation skills currently validate their semantic
bindings against the configured hardware resources used by their phase-specific
hand parameters.

## Planning versus physical success

`ActionPlan.plan_success` reports whether a valid motion was produced per
environment. It does not prove that a grasp, release, contact, or handover
occurred. Such actions declare a `StateDelta`; an `ExecutionSession` commits it
only after the caller supplies a successful semantic-effect verification mask.
