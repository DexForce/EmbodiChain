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
:doc:`/overview/sim/atomic_actions/builtin_actions`. Canonical scene identity and
snapshot/provider setup are documented in :doc:`/overview/sim/scene_registry`.

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
and control-part profiles. It creates and binds all built-in actions by default;
callers select them by stable ``skill_id`` without a separate ``register()``
step. Put invocation-varying behavior in ``ActionInvocation.skill_options``.
``register()`` remains available for custom implementations, and
``load_builtins=False`` creates an isolated or fully custom engine.

Choosing an engine entry point
------------------------------

Application code normally uses one of three engine entry points:

.. list-table::
   :header-rows: 1
   :widths: 18 26 22 34

   * - API
     - Choose it when
     - Returns
     - State and observation behavior
   * - ``engine.plan()``
     - Planning or inspecting one action
     - ``ActionPlan``
     - Reads one context and does not project a next context
   * - ``engine.compile()``
     - Planning a fixed sequence whose goals are already known
     - ``CompiledTrajectory``
     - Propagates hypothetical qpos and expected effects, without observing execution
   * - ``engine.start()``
     - Executing from fresh observations with bounded recovery
     - ``ExecutionSession``
     - ``tick()`` consumes measured context, emits commands, requests effect verification, and can replan

As a short rule: use ``plan`` for one action, ``compile`` for a static action
sequence, and ``start`` followed by ``tick`` for observed execution and error
recovery. None of these APIs steps the simulator directly. The application
sends commands returned by an execution session and supplies new observations.

``AtomicAction.plan(request, context)`` is different from ``engine.plan()``.
It is the framework-owned template method called by the engine, not an
additional application execution entry point. Atomic-action authors implement
the protected ``_plan()`` hook instead. Register custom action instances with
``engine.register()`` before using the same public planning entry points.

This extension contract is intentionally strict: a subclass that defines
``plan()`` raises ``TypeError`` at class definition. There is no legacy adapter;
custom actions must rename that implementation to ``_plan()`` so the
framework-owned collision-scene preparation cannot be bypassed.

Runnable examples
-----------------

Focused examples live under ``scripts/tutorials/atomic_action``:

* ``move_end_effector.py``
* ``move_joints.py``
* ``pickup.py``
* ``move_held_object.py``
* ``place.py``
* ``assemble.py``
* ``press_button.py``
* ``turn_knob.py``
* ``coordinated_pickment.py``
* ``coordinated_placement.py``
* ``hand_over.py``
* ``moving_target_recovery.py``
* ``dynamic_obstacle_recovery.py``

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

Planning one action
-------------------

Use :meth:`~embodichain.lab.sim.atomic_actions.AtomicActionEngine.plan` when one
registered action needs to be inspected, tested, or passed through
application-owned orchestration:

.. code-block:: python

   from embodichain.lab.sim.atomic_actions import (
       ActionBinding,
       ActionInvocation,
       EndEffectorPoseGoal,
       MotionPolicy,
   )

   invocation = ActionInvocation(
       skill_id="move_end_effector",
       goal=EndEffectorPoseGoal(xpos=target_pose),
       binding=ActionBinding(manipulators={"primary": "left_arm"}),
       motion_policy=MotionPolicy(sample_count=80, control_dt=1.0 / 60.0),
   )

   plan = engine.plan(invocation, latest_context)
   if plan.plan_success.all():
       trajectory = plan.trajectory.positions
       diagnostics = plan.diagnostics
       segments = plan.segments

Named segments use half-open action-local waypoint ranges. For a compiled
sequence, call ``compiled.segment(action_index, name)`` to get the corresponding
range in concatenated-trajectory coordinates. This is preferable to repeating
a primitive's private sample-split formula in application or tutorial code.

The returned :class:`~embodichain.lab.sim.atomic_actions.ActionPlan` describes
only that invocation. Its expected effects are not committed, and ``plan`` does
not produce a projected context for a following action. Use ``compile`` when
the engine should propagate hypothetical state through a sequence.

Static compilation
------------------

Use :meth:`~embodichain.lab.sim.atomic_actions.AtomicActionEngine.compile` when
the scene is treated as fixed and all goals in a sequence are known during
planning:

.. code-block:: python

   from embodichain.lab.sim.atomic_actions import (
       ActionBinding,
       ActionInvocation,
       AtomicActionEngine,
       EndEffectorPoseGoal,
       MotionPolicy,
   )

   engine = AtomicActionEngine(motion_generator)
   binding = ActionBinding(manipulators={"primary": "left_arm"})
   motion_policy = MotionPolicy(sample_count=80, control_dt=1.0 / 60.0)

   approach = ActionInvocation(
       skill_id="move_end_effector",
       goal=EndEffectorPoseGoal(xpos=approach_pose),
       binding=binding,
       motion_policy=motion_policy,
   )
   retreat = ActionInvocation(
       skill_id="move_end_effector",
       goal=EndEffectorPoseGoal(xpos=retreat_pose),
       binding=binding,
       motion_policy=motion_policy,
   )

   initial_context = engine.initial_context()
   compiled = engine.compile((approach, retreat), initial_context)
   if compiled.plan_success.all():
       trajectory = compiled.trajectory.positions
       approach_plan, retreat_plan = compiled.action_plans
       final_context = compiled.projected_context

``compile`` never steps the simulator. It applies each plan's expected
:class:`~embodichain.lab.sim.atomic_actions.StateDelta` only to the returned
``projected_context`` so a following action can be planned against hypothetical
state. Calling it with one invocation is valid, but ``plan`` is simpler when a
projected context and sequence-shaped result are unnecessary.

Do not compile across a point where later targets depend on physical execution.
The coordinated-placement tutorial, for example, compiles both pick-ups,
executes them, rebuilds held-object state from measured poses, and then compiles
the placement stage. Use ``start`` when that observation and recovery loop
should remain active throughout execution.

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

   from embodichain.lab.sim.atomic_actions import (
       ExecutionRunner,
       SimulationExecutionAdapter,
       TaskState,
   )
   from embodichain.lab.sim.skills import SceneRegistry

   registry = SceneRegistry.from_simulation(
       sim,
       rigid_objects={"moving_tray": moving_tray.uid},
   )
   scene_provider = registry.make_planning_scene_provider(
       motion_generator,
       batch_size=robot.num_instances,
   )
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

For a lightweight scene source that does not need environment correlation IDs,
pass a ``scene_supplier(timestamp)`` callback instead. ``scene_provider`` and
``scene_supplier`` are mutually exclusive.

The session owns planning progress and bounded recovery. The runner owns the
outer lifecycle: it requests fresh observations, schedules each command from
the :class:`~embodichain.lab.sim.atomic_actions.TimedTrajectory` time deltas,
checks controller acknowledgements, and performs cancel-then-hold on failure.
The simulation adapter advances physics instead of sleeping in wall-clock time.
``ExecutionRunnerCfg`` contains runner-level transport and scheduling settings;
it is not an atomic-action option and is not replaced by invocation revision.

For an application that already owns its event loop, call the non-blocking
:meth:`~embodichain.lab.sim.atomic_actions.ExecutionRunner.step` method. A step
with ``is_waiting`` set has not consumed a new observation or effect result; use
its ``wait_duration`` to schedule the next call.

The complete simulation example starts with a visible cube directly in front of
the robot, then applies a short horizontal force pulse so physics and friction
slide it sideways during one ``PickUp`` invocation whose
``GraspGoal.grasp_xpos`` is a ``SceneEntityPose``. The session observes
``dynamic_goal_changed`` and ``replanned`` events, discards the entire stale
approach/close/lift plan, and rebuilds it from the cube's new location. The
replanned action closes the gripper, verifies the physical lift, and finishes
while holding the cube. The original and regenerated goal axes remain visible
for comparison:

.. code-block:: bash

   python scripts/tutorials/atomic_action/moving_target_recovery.py --headless --auto_play --device cpu

For collision-aware execution, register each pose-updatable obstacle with
``SceneCollisionRole.DYNAMIC`` and configure the same canonical registry IDs as
the planner's dynamic obstacle names. Derive the cuRobo object mapping with
``registry.collision_geometry_by_id()`` and construct the runtime provider with
``registry.make_planning_scene_provider(motion_generator, batch_size=...)``.
That one factory call checks that the registry's complete ``STATIC ∪
DYNAMIC`` set exactly matches the planner's complete collision world, then
checks that the registry, provider, and planner dynamic subsets exactly match.
It also checks planner capability and shared/per-environment world mode. One
environment may infer a shared world; a multi-environment dynamic registry must choose
``SceneCollisionWorldMode.SHARED`` or ``PER_ENV`` explicitly. See
:doc:`/overview/sim/scene_registry` for the complete cuRobo mapping example.

The provider advances per-environment collision-world revisions when an
obstacle moves; the session invalidates affected rows and the framework binds
the latest poses before replanning. Pose thresholds use the last materially
published pose as their baseline, so cumulative sub-threshold motion is
eventually reported:

.. code-block:: bash

   python scripts/tutorials/atomic_action/dynamic_obstacle_recovery.py --headless --auto_play

``MotionPolicy.dynamic_collision_mode`` defaults to
``DynamicCollisionMode.AUTO``. Use ``DynamicCollisionMode.REQUIRED`` when
planning must fail unless the live collision world is available, or
``DynamicCollisionMode.OFF`` to ignore snapshot collision entities and
collision-world revisions. These modes do not disable static-world or
self-collision checks configured by the selected planner.

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

Entities referenced through ``SceneEntityPose`` become automatic scene-motion
dependencies. Object-centric skills may additionally declare an explicit
``ObjectSemantics.entity_id`` when they ground an object pose from the same
scene snapshot; for example, ``PickUp`` automatically tracks that ID. The
legacy ``ObjectSemantics.entity`` live-pose fallback is deprecated and does not
create a scene dependency.

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
only once. The durable state is ``tick.pending_effect`` (an
``EffectVerificationRequest``), not the presence of that one-time event.

Adding an action
----------------

Define an action-owned frozen goal dataclass. Then define typed runtime options
when needed, implement the protected
``_plan(request, context)`` hook, and declare the stable skill metadata. Do not
override the inherited public ``plan()`` method because it binds the latest
collision scene first. Legacy custom actions that implemented ``plan()`` must
rename it to ``_plan()``; defining ``plan()`` is rejected immediately.

Return scalar or per-environment planner success through ``build_plan``. The
framework normalizes the mask and holds failed rows at the observed qpos, so a
new action should not reproduce that masking itself.

A minimal implementation looks like:

.. code-block:: python

   from dataclasses import dataclass
   from typing import ClassVar

   @dataclass(frozen=True, slots=True)
   class PushGoal:
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

       def _plan(
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
