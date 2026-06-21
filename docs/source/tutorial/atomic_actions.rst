.. _tutorial_atomic_actions:

Atomic Actions
==============

EmbodiChain's **atomic action** layer provides a high-level, composable interface for common
manipulation primitives such as *move*, *pick up*, *move object*, and *place*.  Each action
encapsulates the full planning pipeline — grasp-pose estimation, IK, trajectory generation, and
gripper interpolation — behind a single ``execute(target, state)`` call, making it straightforward
to chain multiple actions together into complex robot behaviours.

Key Features
------------

- **Typed targets** — every action accepts one of three small dataclasses: ``PoseTarget``,
  ``GraspTarget`` (wrapping an ``ObjectSemantics``), or ``HeldObjectTarget``. The engine
  checks each step's target against the action's declared ``TargetType`` before running.
- **Built-in primitives** — ``MoveAction``, ``PickUpAction``, ``MoveObjectAction``,
  and ``PlaceAction``
  cover the most common tabletop manipulation workflows out of the box.
  See the :ref:`supported_atomic_actions` table for configs and target types.
- **Extensible registry** — custom action *classes* can be registered globally with
  ``register_action``; action *instances* are registered per-engine under a name.
- **Engine orchestration** — ``AtomicActionEngine.run(steps, state)`` sequences named
  ``(name, typed_target)`` steps, threads a ``WorldState`` (``last_qpos`` + ``held_object``)
  from one action into the next, and returns a single concatenated full-DOF trajectory
  ready to replay in the simulator.

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
       PickUpAction, PickUpActionCfg,
       PlaceAction, PlaceActionCfg,
       MoveAction, MoveActionCfg,
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

   # The engine takes only the motion generator; register each action instance
   # by name (defaults to the action's cfg.name).
   engine = AtomicActionEngine(motion_generator=motion_gen)
   engine.register(PickUpAction(motion_gen, cfg=pickup_cfg))
   engine.register(PlaceAction(motion_gen, cfg=place_cfg))
   engine.register(MoveAction(motion_gen, cfg=move_cfg))

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
       mesh_vertices=mug.get_vertices(env_ids=[0], scale=True)[0],
       mesh_triangles=mug.get_triangles(env_ids=[0])[0],
       gripper_collision_cfg=GripperCollisionCfg(
           max_open_length=0.088, finger_length=0.078, point_sample_dense=0.012
       ),
       generator_cfg=GraspGeneratorCfg(
           antipodal_sampler_cfg=AntipodalSamplerCfg(
               n_sample=20000, max_length=0.088, min_length=0.003
           )
       ),
       force_reannotate=False,
   )

   semantics = ObjectSemantics(
       affordance=affordance,
       geometry={},                 # plain metadata; mesh data lives on the affordance
       label="mug",                 # also bound onto affordance.object_label
       entity=mug,                  # required so the action can query the live object pose
   )

Executing a pick-place-move sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from embodichain.lab.sim.atomic_actions import GraspTarget, PoseTarget

   place_xpos = ...  # torch.Tensor [4, 4] — target placement pose
   rest_xpos  = ...  # torch.Tensor [4, 4] — resting pose after placing

   is_success, trajectory, _ = engine.run(
       steps=[
           ("pick_up", GraspTarget(semantics=semantics)),
           ("place",   PoseTarget(xpos=place_xpos)),
           ("move",    PoseTarget(xpos=rest_xpos)),
       ]
   )
   # trajectory: [n_envs, n_waypoints, robot_dof]

   for i in range(trajectory.shape[1]):
       robot.set_qpos(trajectory[:, i])
       sim.update(step=4)

Moving a held object
~~~~~~~~~~~~~~~~~~~~

``MoveObjectAction`` consumes the runtime ``HeldObjectState`` produced by a previous
``PickUpAction`` (read from the threaded ``WorldState.held_object``). The target is
object-centric: the caller specifies where the held object should move, and the action
converts that pose into an end-effector target via the stored object-to-EEF transform while
keeping the gripper closed.

.. code-block:: python

   from embodichain.lab.sim.atomic_actions import (
       MoveObjectAction, MoveObjectActionCfg,
       GraspTarget, HeldObjectTarget,
   )

   move_object_cfg = MoveObjectActionCfg(
       hand_close_qpos=hand_close,
       control_part="arm",
       hand_control_part="hand",
   )
   engine.register(MoveObjectAction(motion_gen, cfg=move_object_cfg))

   object_target_pose = torch.eye(4, dtype=torch.float32, device=device)
   object_target_pose[:3, 3] = torch.tensor([0.3, -0.2, 0.25], device=device)

   is_success, trajectory, _ = engine.run(
       steps=[
           ("pick_up",     GraspTarget(semantics=semantics)),
           ("move_object", HeldObjectTarget(object_target_pose=object_target_pose)),
       ]
   )

Registering custom actions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import ClassVar
   from embodichain.lab.sim.atomic_actions import (
       AtomicAction, ActionResult, ActionCfg, PoseTarget, WorldState, TrajectoryBuilder,
   )

   class PushAction(AtomicAction):
       TargetType: ClassVar[type] = PoseTarget

       def __init__(self, motion_generator, cfg: PushActionCfg | None = None):
           super().__init__(motion_generator, cfg or PushActionCfg())
           self.builder = TrajectoryBuilder(motion_generator)

       def execute(self, target: PoseTarget, state: WorldState) -> ActionResult:
           # ... your planning logic, using self.builder ...
           return ActionResult(success=is_success, trajectory=full, next_state=...)

   # Register an instance with an engine so it can be referenced by name in run():
   engine.register(PushAction(motion_gen, cfg=PushActionCfg()))

   # Or publish the class globally for third-party discovery:
   from embodichain.lab.sim.atomic_actions import register_action
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
