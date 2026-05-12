.. _tutorial_atomic_actions:

Atomic Actions
==============

EmbodiChain's **atomic action** layer provides a high-level, composable interface for common
manipulation primitives such as *move*, *pick up*, and *place*.  Each action encapsulates the
full planning pipeline — grasp-pose estimation, IK, trajectory generation, and gripper
interpolation — behind a single ``execute()`` call, making it straightforward to chain
multiple actions together into complex robot behaviours.

Key Features
------------

- **Semantic-aware execution** — actions accept either a raw pose tensor or an
  ``ObjectSemantics`` descriptor that bundles affordance data (grasp poses, interaction
  points) with the simulation entity.
- **Three built-in primitives** — ``MoveAction``, ``PickUpAction``, and ``PlaceAction``
  cover the most common tabletop manipulation workflows out of the box.
  See the :ref:`supported_atomic_actions` table for configs and target types.
- **Extensible registry** — custom actions can be registered globally with
  ``register_action`` and discovered by the engine at runtime.
- **Engine orchestration** — ``AtomicActionEngine`` sequences multiple actions,
  threads ``start_qpos`` from one action to the next, and returns a single concatenated
  trajectory ready to replay in the simulator.

For the full design overview, architecture diagram, and extension guide see
:doc:`/overview/sim/atomic_actions`.

The Code
--------

The tutorial corresponds to the ``atomic_actions.py`` script in the ``scripts/tutorials/sim``
directory.

.. dropdown:: Code for atomic_actions.py
    :icon: code

    .. literalinclude:: ../../../scripts/tutorials/sim/atomic_actions.py
        :language: python
        :linenos:

Typical Usage
-------------

Setting up the engine
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg
   from embodichain.lab.sim.atomic_actions import (
       AtomicActionEngine,
       PickUpActionCfg,
       PlaceActionCfg,
       MoveActionCfg,
   )

   motion_gen = MotionGenerator(cfg=MotionGenCfg(...))

   hand_open  = torch.tensor([0.00,  0.00],  dtype=torch.float32, device=device)
   hand_close = torch.tensor([0.025, 0.025], dtype=torch.float32, device=device)

   pickup_cfg = PickUpActionCfg(
       hand_open_qpos=hand_open,
       hand_close_qpos=hand_close,
       control_part="arm",
       hand_control_part="hand",
       approach_direction=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device),
       pre_grasp_distance=0.15,
       lift_height=0.15,
   )
   place_cfg = PlaceActionCfg(
       hand_open_qpos=hand_open,
       hand_close_qpos=hand_close,
       control_part="arm",
       hand_control_part="hand",
       lift_height=0.15,
   )
   move_cfg = MoveActionCfg(control_part="arm")

   engine = AtomicActionEngine(
       motion_generator=motion_gen,
       actions_cfg_list=[pickup_cfg, place_cfg, move_cfg],
   )

Defining object semantics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from embodichain.lab.sim.atomic_actions import (
       ObjectSemantics,
       AntipodalAffordance,
   )
   from embodichain.toolkits.graspkit.pg_grasp import GraspGeneratorCfg, AntipodalSamplerCfg
   from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import GripperCollisionCfg

   affordance = AntipodalAffordance(
       object_label="mug",
       force_reannotate=False,
       custom_config={
           "gripper_collision_cfg": GripperCollisionCfg(
               max_open_length=0.088, finger_length=0.078, point_sample_dense=0.012
           ),
           "generator_cfg": GraspGeneratorCfg(
               antipodal_sampler_cfg=AntipodalSamplerCfg(
                   n_sample=20000, max_length=0.088, min_length=0.003
               )
           ),
       },
   )

   semantics = ObjectSemantics(
       label="mug",
       geometry={
           "mesh_vertices": mug.get_vertices(env_ids=[0], scale=True)[0],
           "mesh_triangles": mug.get_triangles(env_ids=[0])[0],
       },
       affordance=affordance,
       entity=mug,   # required so the action can query the live object pose
   )

Executing a pick-place-move sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   place_xpos = ...  # torch.Tensor [4, 4] — target placement pose
   rest_xpos  = ...  # torch.Tensor [4, 4] — resting pose after placing

   is_success, trajectory = engine.execute_static(
       target_list=[semantics, place_xpos, rest_xpos]
   )
   # trajectory: [n_envs, n_waypoints, robot_dof]

   for i in range(trajectory.shape[1]):
       robot.set_qpos(trajectory[:, i])
       sim.update(step=4)

Registering custom actions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from embodichain.lab.sim.atomic_actions import AtomicAction, ActionCfg, register_action

   class PushAction(AtomicAction):
       def execute(self, target, start_qpos=None, **kwargs):
           # ... your planning logic ...
           return is_success, trajectory, joint_ids

       def validate(self, target, start_qpos=None, **kwargs):
           return True   # quick feasibility check

   register_action("push", PushAction)

Notes & Best Practices
----------------------

- ``PickUpAction`` expects an ``AntipodalAffordance`` with valid mesh data
  (``mesh_vertices`` / ``mesh_triangles``) so the grasp generator can annotate the object.
  Set ``force_reannotate=False`` (the default) to reuse cached annotations across episodes.
- ``ObjectSemantics.entity`` must be set when using semantic targets so the action can read
  the object's current world pose at planning time.
- For static (non-physics) playback, iterate over ``trajectory[:, i]`` and call
  ``robot.set_qpos`` directly; for physics-enabled playback, feed waypoints through your
  controller or gym wrapper instead.
- To add a new action type, see :doc:`/overview/sim/atomic_actions`.
