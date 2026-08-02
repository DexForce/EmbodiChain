Atomic actions
==============

Atomic actions are typed, side-effect-free motion planners. The engine resolves
a grounded :class:`~embodichain.lab.sim.atomic_actions.ActionInvocation` into a
:class:`~embodichain.lab.sim.atomic_actions.ResolvedActionRequest`; an action
combines that snapshot with the latest
:class:`~embodichain.lab.sim.atomic_actions.PlanningContext` and returns an
:class:`~embodichain.lab.sim.atomic_actions.ActionPlan`.

For the complete architecture and ownership model, see
:doc:`/overview/sim/atomic_actions/index`. For the capability matrix and visual
demonstrations of every built-in skill, see
:doc:`/overview/sim/atomic_actions/builtin_actions`.

The contracts deliberately separate six concerns:

* a **goal** describes what should happen;
* an **ActionBinding** maps semantic roles such as ``primary`` or ``source`` to
  names declared in the engine robot's ``control_parts`` mapping;
* a **ControlPartCommandProfile** maps embodiment-specific meanings such as
  ``open``, ``grasp``, or ``ready`` to typed commands;
* typed **ActionOptions** contain behavior that may vary for one skill call;
* a **MotionPolicy** and **RecoveryPolicy** describe reusable planning and
  bounded-recovery choices;
* a **PlanningContext** contains measured robot state, verified task state, and
  a versioned scene snapshot.

Binding values are keys from ``RobotCfg.control_parts``. They are not joint,
link, TCP-frame, or scene-object names. The engine validates them and resolves
their full-robot joint indices before planning. The ``end_effectors`` map names
an actuated hand/tool control part rather than an IK end frame.

A role is an action-defined semantic participant slot, not a control part. In
``{"primary": "left_arm"}``, ``primary`` means the principal participant of
that single-participant action, while ``left_arm`` is the concrete control-part
key. It has no inherent left/right or default-arm meaning. Actions publish their
required slots through ``manipulator_roles`` and ``end_effector_roles``. When a
role such as ``primary`` occurs in both maps, the entries select the arm and
hand/tool serving the same functional participant, but the caller is still
responsible for choosing a physically compatible pair.

The engine exclusively owns the ``MotionGenerator``, shared trajectory builder,
and control-part profiles. Atomic action constructors accept only optional
typed default options; ``register()`` binds each action to the engine resources. Use
``engine.plan_action(action, invocation, context)`` for an unregistered,
default-option-specific action instance.

Runnable examples
-----------------

Focused examples live under ``scripts/tutorials/atomic_action``:

* ``move_end_effector.py``
* ``move_joints.py``
* ``pickup.py``
* ``move_held_object.py``
* ``place.py``
* ``assemble.py``
* ``press.py``
* ``coordinated_pickment.py``
* ``coordinated_placement.py``
* ``hand_over.py``

The scripts are interactive by default. Add ``--auto_play`` to skip prompts;
combine it with ``--headless --device cpu`` for a headless run that records
video under ``outputs/videos``:

.. code-block:: bash

   python scripts/tutorials/atomic_action/move_end_effector.py --headless --auto_play --device cpu
   python scripts/tutorials/atomic_action/pickup.py --headless --auto_play --device cpu
   python scripts/tutorials/atomic_action/assemble.py --headless --auto_play --device cpu
   python scripts/tutorials/atomic_action/hand_over.py --headless --auto_play --device cpu

The ``motion_generator`` variable in the snippets below is a configured
:class:`~embodichain.lab.sim.planners.MotionGenerator`; its robot, planner,
device, cache, and collision world become the resources owned by the engine.

Control-part commands
---------------------

Hand qpos and named robot postures are robot knowledge rather than action
configuration. Register them by concrete ``Robot.control_parts`` key when the
engine is built:

.. code-block:: python

   from embodichain.lab.sim.atomic_actions import (
       AtomicActionEngine,
       ControlPartCommandProfile,
   )

   engine = AtomicActionEngine(
       motion_generator,
       control_profiles={
           "left_hand": ControlPartCommandProfile.joint_positions(
               open=left_open_qpos,
               grasp=left_grasp_qpos,
           ),
           "left_arm": ControlPartCommandProfile.joint_positions(
               ready=left_ready_qpos,
           ),
       },
   )

``PickUp``, ``Place``, and the other manipulation skills resolve ``open`` and
``grasp`` from their bound end effector. ``MoveJoints`` resolves a string target
from its bound manipulator. Joint limits validate possible commands, but do not
define their semantic meaning; supply calibrated robot commands in production.

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
   )

   engine = AtomicActionEngine(motion_generator)
   engine.register(MoveEndEffector())

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

   latest_context = initial_context
   session = engine.start((invocation,), latest_context)
   while session.status.value == "running":
       tick = session.tick(latest_context)
       if tick.command is not None:
           send_joint_command(tick.command)
       latest_context = observe_context()

The session emits one command per tick. It compares observations with the last
command, detects material motion of referenced scene entities, enforces phase
timeouts, and replans from the latest observation within the recovery budget.
It does not own the simulator or controller loop.

Recovery replans reuse one immutable invocation-revision snapshot. If an
application intentionally changes the goal, options, policy, binding, or a
control command while the action is active, submit a strictly newer revision:

.. code-block:: python

   revised = ActionInvocation(
       skill_id=invocation.skill_id,
       goal=updated_goal,
       binding=invocation.binding,
       motion_policy=invocation.motion_policy,
       recovery_policy=invocation.recovery_policy,
       skill_options=updated_options,
       control_overrides=updated_control_commands,
       invocation_id=invocation.invocation_id,
       revision=invocation.revision + 1,
   )
   session.revise_current(revised)

The session replans from its latest context and emits an
``invocation_revised`` event. ``skill_id`` and ``invocation_id`` must still
identify the active logical call.

Only entities referenced through ``SceneEntityPose`` become automatic
scene-motion dependencies. A skill may query a simulation entity's live pose
when it plans, but that query alone does not cause an executing session to
replan when the entity moves.

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
define typed runtime options when needed, implement ``plan(request, context)``,
and declare the stable skill metadata:

.. code-block:: python

   from dataclasses import dataclass
   from typing import ClassVar

   @dataclass(frozen=True, slots=True)
   class PushGoal:
       goal_kind: ClassVar[str] = "push"
       contact_pose: torch.Tensor

   @dataclass(frozen=True, slots=True)
   class PushOptions(ActionOptions):
       retreat_distance: float = 0.1

   class Push(AtomicAction[PushGoal, PushOptions]):
       skill_id: ClassVar[str] = "push"
       GoalType: ClassVar[type] = PushGoal
       OptionsType: ClassVar[type] = PushOptions
       manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

       def __init__(self, default_options: PushOptions | None = None) -> None:
           super().__init__(default_options)

       def plan(
           self,
           request: ResolvedActionRequest[PushGoal, PushOptions],
           context: PlanningContext,
       ) -> ActionPlan:
           goal = self.require_goal(request)
           options = request.skill_options
           # Resolve the bound resource, plan from context.robot.qpos, and
           # return a full-robot TimedTrajectory or position tensor.
           return self.build_plan(
               request,
               context,
               success=success_mask,
               trajectory=full_robot_positions,
           )

Do not step simulation, mutate ``PlanningContext``, commit ``StateDelta``, or
expose planner-specific configuration through the goal. See the in-repository
``add-atomic-action`` skill for the complete checklist.
