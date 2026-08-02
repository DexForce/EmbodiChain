Atomic actions
==============

Atomic actions are typed, side-effect-free motion planners. An action receives a
grounded :class:`~embodichain.lab.sim.atomic_actions.ActionInvocation` and the
latest :class:`~embodichain.lab.sim.atomic_actions.PlanningContext`, then returns
an :class:`~embodichain.lab.sim.atomic_actions.ActionPlan`.

The contracts deliberately separate four concerns:

* a **goal** describes what should happen;
* an **ActionBinding** maps semantic roles such as ``primary`` or ``source`` to
  robot control resources;
* a **MotionPolicy** and **RecoveryPolicy** describe reusable planning and
  bounded-recovery choices;
* a **PlanningContext** contains measured robot state, verified task state, and
  a versioned scene snapshot.

Static compilation
------------------

Use :meth:`~embodichain.lab.sim.atomic_actions.AtomicActionEngine.compile` when
the scene is treated as fixed during planning:

.. code-block:: python

   from embodichain.lab.sim.atomic_actions import (
       ActionBinding,
       ActionInvocation,
       AtomicActionEngine,
       EndEffectorPoseGoal,
       MotionPolicy,
       MoveEndEffector,
       MoveEndEffectorCfg,
   )

   engine = AtomicActionEngine(motion_generator)
   engine.register(MoveEndEffector(motion_generator, MoveEndEffectorCfg()))

   invocation = ActionInvocation(
       skill_id="move_end_effector",
       goal=EndEffectorPoseGoal(xpos=target_pose),
       binding=ActionBinding(manipulators={"primary": "left_arm"}),
       motion_policy=MotionPolicy(sample_count=80, control_dt=1.0 / 60.0),
   )
   compiled = engine.compile((invocation,))
   trajectory = compiled.trajectory.positions

``compile`` never steps the simulator. It applies each plan's expected
:class:`~embodichain.lab.sim.atomic_actions.StateDelta` only to the returned
``projected_context`` so a following action can be planned against hypothetical
state.

Dynamic goals and closed-loop execution
---------------------------------------

Use :class:`~embodichain.lab.sim.atomic_actions.SceneEntityPose` when a goal
must be resolved from the latest scene snapshot:

.. code-block:: python

   from embodichain.lab.sim.atomic_actions import (
       EndEffectorPoseGoal,
       RecoveryPolicy,
       SceneEntityPose,
   )

   invocation = ActionInvocation(
       skill_id="move_end_effector",
       goal=EndEffectorPoseGoal(
           xpos=SceneEntityPose("moving_tray", relative_pose=tray_to_tcp)
       ),
       binding=ActionBinding(manipulators={"primary": "left_arm"}),
       recovery_policy=RecoveryPolicy(
           max_replans=3,
           tracking_error_threshold=0.05,
           goal_translation_threshold=0.02,
       ),
   )

   session = engine.start((invocation,), initial_context)
   while session.status.value == "running":
       tick = session.tick(latest_context)
       if tick.command is not None:
           send_joint_command(tick.command)

The session emits one command per tick. It compares observations with the last
command, detects material motion of referenced scene entities, enforces phase
timeouts, and replans from the latest observation within the recovery budget.
It does not own the simulator or controller loop.

Task-state effects
------------------

Pick, place, handover, and coordinated skills declare attachment changes as a
:class:`~embodichain.lab.sim.atomic_actions.StateDelta`. Planning does not commit
those changes. During closed-loop execution, a non-empty effect requires an
external per-environment verification mask:

.. code-block:: python

   tick = session.tick(latest_context)
   if any(event.kind.value == "effect_verification_required" for event in tick.events):
       verified = verify_grasp_or_release()
       tick = session.tick(latest_context, effect_success=verified)

This prevents a successful trajectory plan from being mistaken for a successful
physical grasp or release.

Adding an action
----------------

Define an action-owned frozen goal dataclass with a stable ``goal_kind``. Then
implement ``plan(invocation, context)`` and declare the stable skill metadata:

.. code-block:: python

   from dataclasses import dataclass
   from typing import ClassVar

   @dataclass(frozen=True, slots=True)
   class PushGoal:
       goal_kind: ClassVar[str] = "push"
       contact_pose: torch.Tensor

   class Push(AtomicAction[PushGoal]):
       skill_id: ClassVar[str] = "push"
       GoalType: ClassVar[type] = PushGoal
       manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

       def plan(
           self,
           invocation: ActionInvocation[PushGoal],
           context: PlanningContext,
       ) -> ActionPlan:
           goal = self.require_goal(invocation)
           # Resolve the bound resource, plan from context.robot.qpos, and
           # return a full-robot TimedTrajectory or position tensor.
           return self.build_plan(
               invocation,
               context,
               success=success_mask,
               trajectory=full_robot_positions,
           )

Do not step simulation, mutate ``PlanningContext``, commit ``StateDelta``, or
expose planner-specific configuration through the goal. See the in-repository
``add-atomic-action`` skill for the complete checklist.
