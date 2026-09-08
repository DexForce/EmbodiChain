Configure and Run an Embodied Task Program
===========================================

This tutorial deploys Task Program's built-in Semantic Calls through explicit
environment, integration, embodiment, and execution-policy configuration. The
calls are declarative language instructions, not an independent execution API.

For a smaller introduction to the language itself, first complete
:doc:`task_program_python`, which constructs and compiles the same typed schema
in Python without starting a Gym environment.

Use :doc:`atomic_actions` instead when Python code should construct and execute
typed ``ActionInvocation`` values directly. The architecture and ownership
rules are described in :doc:`/overview/task_program/index`.

What you will configure
-----------------------

A supported configuration-defined Task Program has six explicit owners:

1. ``program.yaml`` owns the embodiment-independent task flow and targets;
2. ``integration.yaml`` owns task-specific semantic requirements, its nested
   ``scene_binding``, resource defaults, action options, effect monitors, and
   runtime services;
3. ``env.yaml`` owns the physical scene and ordinary Gym environment values;
4. an embodiment component owns one robot, its sensors, and optional semantic
   skill profile;
5. an execution-policy component owns motion, recovery, runner, and effect
   assurance settings; and
6. ``task.<embodiment>.yaml`` selects those reusable owners for one runnable
   Gym ID.

There is no separate physical ``scene.yaml`` or task component. In particular,
``env.yaml`` contains no ``id`` or ``task_program`` field, so the same physical
environment can also be selected by a handwritten-trajectory deployment.

The shared ``EmbodiedEnv`` and Task Program adapter then lower Semantic Calls
to Atomic Skills while retaining the normal stepping, recording, and reset
lifecycle.

The official repeated Pick/Place example is a complete reference:

* ``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.yaml``;
* ``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task_program/``;
* ``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml``;
* ``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml``;
* ``embodichain_tasks/configs/components/embodiments/ur5_dh_pgi_140_80.yaml``;
* ``embodichain_tasks/configs/components/execution_policies/trajectory_open_loop.yaml``.

These files have explicit ownership rather than generic deep-merge semantics.
The deployment must select components that satisfy the task's scene and
embodiment contracts.

Declare object-centric intent
-----------------------------

The program names semantic scene entities and robot resources. It contains no
joint names, planners, controller commands, simulator objects, or Python
callables:

.. code-block:: yaml

   program_id: repeated_cube_pick_place
   targets:
     drop_pose:
       kind: cyclic_pose
       values:
         - position: [-0.40, 0.48, 0.10]
           quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
         - position: [-0.42, -0.08, 0.10]
           quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
   program:
     kind: repeat
     count: 3
     body:
       kind: segment
       name: move_cube
       steps:
         kind: sequence
         items:
           - kind: invoke
             call:
               kind: pick
               object: cube
           - kind: invoke
             call:
               kind: place
               object: cube
               at: {kind: target_ref, target: drop_pose}

``cube`` is a canonical semantic object rather than a simulator UID. The
program deliberately contains no integration selection or robot-specific
resource ID. The task integration supplies the logical Pick and Place resource
defaults, and the selected embodiment supplies compatible concrete endpoints.

Compose the deployment
----------------------

The reusable environment contains only physical simulation entities and
ordinary environment values. This abbreviated shape is intentionally free of
Task Program and embodiment fields:

.. code-block:: yaml

   environment_id: repeated_pick_place
   max_episode_steps: 1200

   simulation:
     rigid_object:
       - uid: cube
         # Physical shape, dynamics, and initial pose.

   env:
     sim_steps_per_control: 4
     events: {}
     dataset: {}

The thin runnable deployment chooses that environment, all three Task Program
files, and one embodiment without copying any of them:

.. code-block:: yaml

   id: TaskProgramRepeatedPickPlace-v1

   environment:
     component: env.yaml

   task_program:
     program: task_program/program.yaml
     integration: task_program/integration.yaml
     execution_policy: ../../../components/execution_policies/trajectory_open_loop.yaml

   embodiment:
     component: ../../../components/embodiments/ur5_dh_pgi_140_80.yaml

During ``config_to_cfg()``, the component resolver expands these selections,
checks that every semantic ``simulation_uid`` exists in the physical scene,
validates the scene and embodiment contracts, binds the exact scene-registry,
robot-profile, and policy-preset IDs into the otherwise unbound program, and
registers the common ``EmbodiedEnv``. Component references resolve relative to
the runnable deployment file.

Select effect assurance explicitly
----------------------------------

Every execution policy must declare how semantic state is authorized to
advance. The trajectory-only component selects projected assurance:

.. code-block:: yaml

   policy_id: trajectory_open_loop_v1
   preset_id: trajectory
   effect_assurance: projected

``projected`` applies the action plan's expected symbolic effect after command
completion. The matching task integration must declare an empty
``effect_monitors`` mapping, and successful completion is not proof that the
physical grasp or release occurred.

For a physically verified curated call, select ``verified`` in the execution
policy and map every used Pick, Place, or HandOver call to an explicit monitor
in the task integration:

.. code-block:: yaml

   profile:
     effect_monitors:
       hand_over:
         monitor_id: builtin.composite_effect
         revision: "1"
         params: {consecutive_samples: 10}

The integration must also install the evidence providers required by that
monitor. Command acknowledgement alone is not physical evidence.

Load and validate without simulation
------------------------------------

The provider-independent API performs strict file decoding before any live
environment is constructed:

.. code-block:: python

   from embodichain.lab.task_program import (
       TaskProgramIntegrationCfg,
       load_task_program,
   )

   program = load_task_program(
       "embodichain_tasks/configs/tasks/manipulation/"
       "repeated_pick_place/task_program/program.yaml",
       integration=TaskProgramIntegrationCfg(
           robot_profile="ur5_dh_pgi_140_80",
           scene_registry="task_program_repeated_pick_place",
           runtime_preset="trajectory",
       ),
   )
   print(program.program_id)

The explicit ``integration`` argument represents the trusted deployment
selection that ``config_to_cfg()`` normally derives from the selected
embodiment, scene binding, and execution policy. A configured ``program.yaml``
must not declare those IDs itself.

Unknown fields, duplicate YAML keys, invalid references, executable payloads,
and bounded-expansion violations raise a Task Program config error. Static
compilation additionally checks the program against a provider-free
``SceneManifest``; it does not observe simulation state or generate controller
actions.

Run through the environment entry point
---------------------------------------

Run the complete configuration with the normal environment CLI:

.. code-block:: bash

   embodichain run-env \
     --gym_config \
     embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
     --headless

To select another compatible program while keeping the trusted integration:

.. code-block:: bash

   embodichain run-env \
     --gym_config path/to/task.ur5.yaml \
     --task-program path/to/program.yaml \
     --headless

The adapter produces lazy ``DemoSegment`` actions. The normal demo executor,
not the semantic package, calls ``env.step()`` and owns recording and reset.

Add settling and task validation
--------------------------------

A segment can distinguish physical-call verification from task acceptance:

.. code-block:: yaml

   program:
     kind: segment
     name: deliver_can
     steps:
       kind: invoke
       call:
         kind: hand_over
         object: can
         final_target: {kind: target_ref, target: delivery_pose}
     post:
       - kind: wait_stable
         entity: can
         preset: rigid_object
     validators:
       - kind: object_near_target
         object: can
         target: delivery_pose
         position_tolerance: 0.12

The effect monitor verifies the semantic HandOver transition. ``wait_stable``
advances environment behavior after motion. ``object_near_target`` decides
whether the final task segment is acceptable. These are intentionally separate
boundaries.

Use registered semantic calls carefully
---------------------------------------

Use ``RegisteredSemanticCallCfg`` only when a task needs an allowlisted shared
lowering that is not one of Pick, Place, or HandOver. Its arguments remain
declarative. The trusted simulation registration owns the matching lowerer
factory and fingerprints its call ID, revision, and target atomic descriptor.

Do not place a callable, dotted import, simulator object, or task-local motion
generator in the program. If a reusable behavior is fundamentally a new motion
primitive, add it to Atomic Actions first and then expose a declarative
registered call through Task Program.

Next steps
----------

* :doc:`/overview/task_program/scene_registry` — canonical IDs, affordances,
  and live providers;
* :doc:`/overview/task_program/robot_profiles` — resource graphs,
  command presets, and effect assurance;
* :doc:`/overview/task_program/index` — schema, compilation,
  parallel safety, and adapter lifecycle;
* :doc:`/overview/sim/atomic_actions/builtin_actions` — the atomic primitives
  reached after lowering.
