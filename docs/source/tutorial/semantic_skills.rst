Semantic skills through Expert Program
=======================================

Semantic skills are declarative contracts, not an execution API. This tutorial
shows how to use the built-in semantic vocabulary through the supported
task-level entry point: Expert Program.

Use :doc:`atomic_actions` instead when Python code should construct and execute
typed ``ActionInvocation`` values directly. The architecture and ownership
rules are described in :doc:`/overview/sim/semantic_skills` and
:doc:`/overview/sim/atomic_actions/expert_programs`.

What you will configure
-----------------------

One runnable semantic task has three pieces:

1. an Expert Program YAML file containing object-centric calls;
2. a Gym JSON file containing the trusted scene, robot profile, policy preset,
   and allowlisted runtime services; and
3. the shared ``EmbodiedEnv`` and Expert Program adapter that lower the calls
   to Atomic Actions.

The official repeated Pick/Place example is a complete reference:

* ``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/expert/program.yaml``;
* ``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json``.

Declare object-centric intent
-----------------------------

The program names semantic scene entities and robot resources. It contains no
joint names, planners, controller commands, simulator objects, or Python
callables:

.. code-block:: yaml

   program_id: repeated_cube_pick_place
   integration:
     robot_profile: expert_program_ur5_pick_place
     scene_registry: expert_program_repeated_pick_place
     runtime_preset: trajectory
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
               resources: {primary: manipulator}
           - kind: invoke
             call:
               kind: place
               object: cube
               at: {kind: target_ref, target: drop_pose}
               resources: {primary: manipulator}

``cube`` must be a canonical scene object. ``manipulator`` must be a robot
resource that satisfies the atomic Pick and Place binding contracts. The
integration IDs in the program must exactly match the trusted Gym runtime.

Select effect assurance explicitly
----------------------------------

Every policy preset must declare how semantic state is authorized to advance.
For a trajectory-only example:

.. code-block:: json

   {
     "preset_id": "trajectory",
     "action_options": {
       "pick": {"kind": "pick_up"},
       "place": {"kind": "place"}
     },
     "effect_assurance": "projected",
     "effect_monitors": {}
   }

``projected`` applies the action plan's expected symbolic effect after command
completion. It forbids monitors and is not proof that the physical grasp or
release succeeded.

For a physically verified curated call, select ``verified`` and map every used
Pick, Place, or HandOver call to an explicit monitor:

.. code-block:: json

   {
     "preset_id": "safe",
     "action_options": {
       "hand_over": {"kind": "hand_over"}
     },
     "effect_assurance": "verified",
     "effect_monitors": {
       "hand_over": {
         "monitor_id": "builtin.composite_effect",
         "revision": "1",
         "params": {"consecutive_samples": 10}
       }
     }
   }

The integration must also install the evidence providers required by that
monitor. Command acknowledgement alone is not physical evidence.

Load and validate without simulation
------------------------------------

The provider-independent API performs strict file decoding before any live
environment is constructed:

.. code-block:: python

   from embodichain.lab.expert_program import load_expert_program

   program = load_expert_program(
       "embodichain_tasks/configs/tasks/manipulation/"
       "repeated_pick_place/expert/program.yaml"
   )
   print(program.program_id)

Unknown fields, duplicate YAML keys, invalid references, executable payloads,
and bounded-expansion violations raise an Expert Program config error. Static
compilation additionally checks the program against a provider-free
``SceneManifest``; it does not observe simulation state or generate controller
actions.

Run through the environment entry point
---------------------------------------

Run the complete configuration with the normal environment CLI:

.. code-block:: bash

   python -m embodichain.lab.scripts.run_env \
     --gym_config \
     embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json \
     --headless

To select another compatible program while keeping the trusted integration:

.. code-block:: bash

   python -m embodichain.lab.scripts.run_env \
     --gym_config path/to/env.json \
     --expert-program path/to/program.yaml \
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
registered call through Expert Program.

Next steps
----------

* :doc:`/overview/sim/scene_registry` — canonical IDs, affordances, and live
  providers;
* :doc:`/overview/sim/atomic_actions/robot_skill_profiles` — resource graphs,
  command presets, and effect assurance;
* :doc:`/overview/sim/atomic_actions/expert_programs` — schema, compilation,
  parallel safety, and adapter lifecycle;
* :doc:`/overview/sim/atomic_actions/builtin_actions` — the atomic primitives
  reached after lowering.
