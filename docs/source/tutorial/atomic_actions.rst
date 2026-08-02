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
       RigidObjectSceneProvider,
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

   from embodichain.lab.sim.atomic_actions import (
       ExecutionRunner,
       SimulationExecutionAdapter,
       TaskState,
   )

   scene_provider = RigidObjectSceneProvider({"moving_tray": moving_tray})
   adapter = SimulationExecutionAdapter(
       sim,
       robot,
       scene_provider=scene_provider,
   )
   task = TaskState.empty(robot.get_qpos().shape[0], robot.device)
   initial_context = adapter.observe(task)
   session = engine.start((invocation,), initial_context)
   runner = ExecutionRunner(session, adapter, adapter, clock=adapter)
   result = runner.run_until_blocked()

The session owns planning progress and bounded recovery. The runner owns the
outer lifecycle: it requests fresh observations, schedules each command from
the :class:`~embodichain.lab.sim.atomic_actions.TimedTrajectory` time deltas,
checks controller acknowledgements, and performs cancel-then-hold on failure.
The simulation adapter advances physics instead of sleeping in wall-clock time.

For an application that already owns its event loop, call the non-blocking
:meth:`~embodichain.lab.sim.atomic_actions.ExecutionRunner.step` method. A step
with ``is_waiting`` set has not consumed a new observation or effect result; use
its ``wait_duration`` to schedule the next call.

The complete simulation example deliberately changes a measured joint position,
observes ``tracking_error`` and ``replanned`` events, and finishes the regenerated
trajectory:

.. code-block:: bash

   python scripts/tutorials/atomic_action/tracking_error_recovery.py --headless

The moving-goal counterpart changes a rigid object's pose while an
``EndEffectorPoseGoal(SceneEntityPose(...))`` is executing:

.. code-block:: bash

   python scripts/tutorials/atomic_action/moving_target_recovery.py --headless

For collision-aware execution, declare tracked rigid objects as collision
entities. The provider advances a per-environment collision-world revision when
an obstacle moves; the active phase is invalidated and a supporting planner,
such as cuRobo, receives the latest obstacle poses during replanning:

.. code-block:: bash

   python scripts/tutorials/atomic_action/dynamic_obstacle_recovery.py --headless

Task-state effects
------------------

Pick, place, handover, and coordinated skills declare attachment changes as a
:class:`~embodichain.lab.sim.atomic_actions.StateDelta`. Planning does not commit
those changes. During closed-loop execution, a non-empty effect requires an
external per-environment verification mask:

.. code-block:: python

   def verify_effect(context, tick):
       return verify_grasp_or_release(context)

   result = runner.run_until_blocked(effect_verifier=verify_effect)

This prevents a successful trajectory plan from being mistaken for a successful
physical grasp or release. If verification is asynchronous, omit the callback;
``run_until_blocked`` returns at the verification boundary and the application
can later resume with ``runner.step(effect_success=verified)`` when the next
cycle is due, or call ``run_until_blocked(effect_verifier=...)`` again. The
runner remembers the pending boundary even though the session emits its event
only once.

Adding an action
----------------

Define an action-owned frozen goal dataclass with a stable ``goal_kind``. Then
implement the protected ``_plan(invocation, context)`` hook and declare the
stable skill metadata. The inherited public ``plan()`` method must not be
overridden because it binds the latest collision scene first:

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

       def _plan(
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
