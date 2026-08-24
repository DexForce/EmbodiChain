Semantic skills
===============

Semantic skills let task code describe object-centric intent while the scene
registry and robot profile own simulator- and embodiment-specific details. This
tutorial covers two complete examples:

* ``Pick -> Place`` with one manipulator;
* ``Pick -> RegisteredSemanticCall`` lowered to a dual-arm HandOver.

The runnable sources are:

* ``scripts/tutorials/semantic_skill/place.py``;
* ``scripts/tutorials/semantic_skill/hand_over.py``;
* ``scripts/tutorials/semantic_skill/tutorial_utils.py`` for shared setup and
  verification helpers.

Both runnable examples use the same three-part structure:

* ``create_*_application(...)`` assembles a fully bound
  ``SkillRuntime`` and installs its default physical-effect verifier;
* ``create_*_task()`` declares only robot-independent semantic calls;
* ``app.run(task, ...)`` is the application-facing execution entry point.

``app`` is still the canonical ``SkillRuntime`` rather than another wrapper class.
The factory only keeps simulator, scene-registry, robot-profile, and verifier
construction out of the task declaration.

Read :doc:`atomic_actions` first if you need the underlying planning, execution,
and effect-verification model. The complete semantic architecture is documented
in :doc:`/overview/sim/semantic_skills`; canonical scene registration is covered
by :doc:`/overview/sim/scene_registry`.

Run the examples
----------------

The examples are interactive by default:

.. code-block:: bash

   python scripts/tutorials/semantic_skill/place.py
   python scripts/tutorials/semantic_skill/hand_over.py

For an unattended simulation run, use:

.. code-block:: bash

   python scripts/tutorials/semantic_skill/place.py --headless --auto_play --device cpu
   python scripts/tutorials/semantic_skill/hand_over.py --headless --auto_play --device cpu

Both examples accept the common simulation tutorial flags. Use ``--help`` for
the complete list. ``--diagnose_plan`` takes a separate offline path that
analyzes, grounds, and statically compiles the workflow without executing
controller commands:

.. code-block:: bash

   python scripts/tutorials/semantic_skill/place.py --headless --device cpu --diagnose_plan
   python scripts/tutorials/semantic_skill/hand_over.py --headless --device cpu --diagnose_plan

.. attention::

   Diagnostic compilation projects expected attachment changes hypothetically
   between calls. It proves that the current workflow can be lowered and
   planned; it does not prove that a physical grasp, release, or transfer
   occurred. Normal execution uses ``SkillRuntime`` and an explicit
   effect verifier.

The application entry
---------------------

After constructing the simulator entities, normal task-facing code is compact:

.. code-block:: python

   app = create_place_application(
       sim,
       robot,
       workpiece,
       hand_open=hand_open,
       hand_grasp=hand_grasp,
       n_sample=args.n_sample,
       force_reannotate=args.force_reannotate,
   )

   result = app.run(
       create_place_task(),
       task_id="tutorial.semantic_pick_place",
       on_step=observe_runtime_step,
   )
   result.require_all_succeeded()

The factory installs the live effect verifier on the runtime, so it does not
appear in every ``run`` call. ``on_step`` remains explicit because it is
optional tutorial observability rather than semantic task intent.

Keep the whole known task in one tuple when possible. When a later call
genuinely depends on a new observation or an application/agent decision, wait
for the current workflow to reach a terminal ``SkillResult`` and submit the
next workflow on the same runtime.

Example 1: semantic Pick and Place
----------------------------------

The Place example demonstrates the normal built-in path. Its workflow contains
no robot control-part names:

.. code-block:: python

   from embodichain.lab.sim.skills import (
       Pick,
       Place,
       SceneObjectRef,
       SemanticPose,
   )

   def create_place_task() -> tuple[Pick, Place]:
       workpiece = SceneObjectRef("workpiece")
       return (
           Pick(object=workpiece),
           Place(
               object=workpiece,
               at=SemanticPose(
                   position=(-0.40, 0.48, 0.025),
                   quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
               ),
           ),
       )

The scene registry owns object identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simulation object is registered under the canonical semantic ID
``workpiece``. The grasp affordance is a direct child with the open
``affordance.grasp`` capability and an explicit payload revision. The object
selects it as its capability-scoped default:

.. code-block:: python

   object_ref = SceneObjectRef("workpiece")
   grasp_ref = SceneAffordanceRef("workpiece.grasp.antipodal")

   object_registration = replace(
       simulation_registry.lookup(object_ref),
       semantic_type="cube",
       default_affordances={GRASP_AFFORDANCE_CAPABILITY: grasp_ref},
   )
   registry = SceneRegistry(
       (
           object_registration,
           SceneEntityRegistration(
               ref=grasp_ref,
               parent=object_ref,
               native_name="antipodal_grasp",
               affordance=antipodal_affordance,
               affordance_capabilities=frozenset(
                   {GRASP_AFFORDANCE_CAPABILITY}
               ),
               affordance_revision="antipodal-v1",
               relative_pose=torch.eye(4),
           ),
       )
   )

``Pick(object=workpiece)`` can now omit an explicit affordance. Resolution is
deterministic because the parent owns one default for the required capability.

The robot profile owns embodiment details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tutorial profile maps one logical ``primary_manipulator`` resource onto the
robot's ``arm`` motion endpoint and ``hand`` grasp endpoint. It also owns the
semantic ``open`` and ``grasp`` joint commands, resource default, and per-skill
policy presets:

.. code-block:: python

   profile = RobotSkillProfile(
       profile_id="tutorial.single_arm",
       resources={
           "primary_manipulator": create_manipulator_resource(
               "primary_manipulator",
               motion_control_part="arm",
               grasp_control_part="hand",
           )
       },
       command_profiles={
           "hand": ControlPartCommandProfile.joint_positions(
               open=hand_open,
               grasp=hand_grasp,
           )
       },
       defaults={
           "pick_up": ResourceBinding(
               resources={"primary": "primary_manipulator"}
           ),
           "place": ResourceBinding(
               resources={"primary": "primary_manipulator"}
           ),
       },
       presets={...},
       default_preset="default",
   )

The semantic calls remain unchanged if another robot profile can satisfy the
same atomic skill contracts.

Assemble and run the application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SkillRuntime.from_simulation`` assembles the standard planning scene,
simulation ports, action engine, manifest, and compiler:

.. code-block:: python

   def create_place_application(
       simulation,
       robot,
       workpiece,
       *,
       hand_open,
       hand_grasp,
       n_sample,
       force_reannotate,
   ) -> SkillRuntime:
       registry = ...
       profile = ...
       verify_effect = ...
       return SkillRuntime.from_simulation(
           simulation=simulation,
           robot=robot,
           motion_generator=create_curobo_motion_generator(robot),
           scene_registry=registry,
           robot_profile=profile,
           effect_verifier=verify_effect,
           control_dt=4 * simulation.sim_config.physics_dt,
       )

   app = create_place_application(...)
   result = app.run(
       create_place_task(),
       task_id="tutorial.semantic_pick_place",
       on_step=observe_runtime_step,
   )
   result.require_all_succeeded()

The verifier checks the live lift, the planned object-to-EEF relation, final
object position, and open hand before accepting the symbolic effects. The
``on_step`` callback reports recovery events and does not decide whether an
effect succeeded.

Submitting both calls in one segment is important for planning quality. Static
analysis can pass the Place target to Pick as a downstream reachability target.
The runtime still grounds and executes one call at a time from fresh
observations.

Example 2: registered dual-arm HandOver
---------------------------------------

The dual-arm example demonstrates the extension path in addition to resource
disjointness. It registers ``tutorial.hand_over`` against the existing atomic
HandOver descriptor:

.. code-block:: python

   call_catalog = builtin_semantic_call_catalog().with_descriptor(
       SemanticCallDescriptor(
           call_id="tutorial.hand_over",
           spec_type=RegisteredSemanticCall,
           target_descriptor=AtomicHandOver.descriptor(),
       )
   )

   def create_handover_task() -> tuple[Pick, RegisteredSemanticCall]:
       return (
           Pick(object=workpiece),
           RegisteredSemanticCall(
               call_id="tutorial.hand_over",
               arguments={"object": workpiece},
           ),
       )

Why use a registered call here?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The public semantic :class:`~embodichain.lab.sim.skills.HandOver` path delegates
middle and final object targets to a named ``HandOverPoseProvider`` selected by
the robot profile. This tutorial instead demonstrates how an application can
publish a separate versioned call schema and explicitly lower it to tuned
``HandOverOptions``.

The lowerer is executable integration code, so it is installed on the compiler
rather than placed inside ``RegisteredSemanticCall.arguments``:

.. code-block:: python

   class TutorialHandOverLowerer(RegisteredSemanticLowerer):
       call_id = "tutorial.hand_over"
       schema_version = 1

       def lower(self, call, *, context, bound):
           semantics = registry.object_semantics(
               call.arguments["object"],
               affordance=registry.resolve_affordance(
                   call.arguments["object"],
                   capability=GRASP_AFFORDANCE_CAPABILITY,
               ),
           )
           return SemanticLowering(
               goal=GraspGoal(semantics),
               skill_options=HandOverOptions(
                   middle_object_pose=middle_pose.to(context.robot.qpos.device),
                   final_object_pose=final_pose.to(context.robot.qpos.device),
                   receive_pick_object_part="bottom",
               ),
           )

The descriptor and lowerer call IDs and schema versions must match exactly. A
registered call cannot replace the curated ``pick``, ``place``, or
``hand_over`` meanings.

The dual-arm profile declares two physically disjoint manipulator resources.
Its defaults map Pick's ``primary`` slot to the source and HandOver's ``source``
and ``destination`` slots to different resources. Binding rejects overlapping
claims before execution.

The application factory passes the extension objects and default verifier to
the runtime explicitly:

.. code-block:: python

   def create_handover_application(
       simulation,
       robot,
       workpiece,
       *,
       left_open,
       left_grasp,
       right_open,
       right_grasp,
       n_sample,
       force_reannotate,
   ) -> SkillRuntime:
       registry = ...
       profile = ...
       verify_effect = ...
       return SkillRuntime.from_simulation(
           simulation=simulation,
           robot=robot,
           motion_generator=create_toppra_motion_generator(robot),
           scene_registry=registry,
           robot_profile=profile,
           call_catalog=call_catalog,
           effect_verifier=verify_effect,
           registered_lowerers=(TutorialHandOverLowerer(registry),),
           control_dt=4 * simulation.sim_config.physics_dt,
       )

   app = create_handover_application(...)
   result = app.run(
       create_handover_task(),
       task_id="tutorial.semantic_pick_handover",
       on_step=observe_runtime_step,
   )

The effect verifier first accepts the source Pick only after observing the held
relation. At the transfer boundary it verifies source release, destination
grasp, destination ownership, and the final object target before committing the
new ``TaskState``.

Dynamic decisions between workflows
-----------------------------------

When an application or agent cannot know the whole task in advance, use a
completed workflow as the decision boundary:

.. code-block:: python

   acquire = app.run(
       (Pick(object=workpiece),),
       task_id="agent_task.acquire",
   )
   acquire.require_all_succeeded()

   # Decide only after the workflow has committed verified state.
   destination = choose_destination(acquire.task_state)
   result = app.run(
       (Place(object=workpiece, at=destination),),
       task_id="agent_task.deliver",
   )

Successful workflows retain verified symbolic state and the per-environment
eligible mask. A failed or cancelled workflow is terminal. Only one workflow
may own a runtime at a time; scheduling multiple independent tasks is an
application responsibility.

For non-blocking integration, call ``app.start(*calls,
workflow_id=...)`` and repeatedly call ``app.step()``. If the returned result
has a positive ``wait_duration``, wait on the runtime clock before the next
step. Physical effects are evaluated from the configured evidence collector and
effect monitors; application callbacks observe runner steps but do not submit a
separate boolean success mask.

Recovery boundaries
-------------------

The semantic runtime delegates call-local recovery to the atomic
``ExecutionRunner``. Tracking errors, supported scene-target movement,
collision-world revisions, timeouts, and planning failure produce structured
events and consume the selected preset's bounded recovery budget.

Keep these distinctions in mind:

* Pick monitors its object and grasp target only through the approach segment;
  contact- or lift-induced motion does not look like an external target update.
* Physical effects are never committed from planning success alone.
* Rows that exhaust recovery become ineligible and remain excluded from later
  calls and dynamic segments.
* A terminal runtime failure does not automatically choose another skill or
  reconcile uncertain physical state. Perform that task-level recovery at a
  completed-workflow boundary.

Inspect ``SkillResult.status``, ``eligible_mask``, ``calls``, ``effects``, and
aggregated ``events`` for structured feedback. Call ``require_all_succeeded``
when partial vectorized success should be treated as an application error.

Further reading
---------------

* :doc:`/overview/sim/semantic_skills` — architecture, ownership, dynamic tasks,
  and extension contracts;
* :doc:`/overview/sim/scene_registry` — canonical IDs, affordances, snapshots,
  and collision integration;
* :doc:`/overview/sim/atomic_actions/robot_skill_profiles` — resource graphs,
  policy presets, and grounding-provider selection;
* :doc:`/overview/sim/atomic_actions/builtin_actions` — behavior and recovery
  contracts of the lowered atomic primitives.
