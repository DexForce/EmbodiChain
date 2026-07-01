Rigid constraint tutorial
==========================

.. currentmodule:: embodichain.lab.sim

This tutorial shows how to attach two rigid objects via a fixed physics
constraint, observe the constraint holding their relative pose, and then remove
it. It follows the style used in the :doc:`rigid_object_group` tutorial and
references the example script located in
``scripts/tutorials/sim/create_rigid_constraint.py``.

A *fixed constraint* (a weld) binds two dynamic bodies so that their relative
pose is held constant by the physics solver — they move as a single rigid
assembly until the constraint is removed. This is useful for grasping and
assembly tasks, where an object must be "held" to a gripper or two parts must
be joined temporarily.

.. tip::

   Constraints are created and removed through the
   :class:`SimulationManager`, which owns one constraint handle per arena. The
   same API is exposed as on-demand event functors (``create_rigid_constraint``
   / ``remove_rigid_constraint`` in ``embodichain.lab.gym.envs.managers.events``)
   so a task environment can attach/detach mid-episode via
   ``env.event_manager.apply(mode="attach", env_ids=...)``.

The Code
~~~~~~~~

The tutorial corresponds to the ``create_rigid_constraint.py`` script in the
``scripts/tutorials/sim`` directory.

.. dropdown:: Code for create_rigid_constraint.py
	:icon: code

	.. literalinclude:: ../../../scripts/tutorials/sim/create_rigid_constraint.py
		:language: python
		:linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~


Adding two cubes
----------------

Two dynamic cubes are added with :meth:`SimulationManager.add_rigid_object`.
Each uses a :class:`CubeCfg` shape (a primitive cube, so no mesh asset file is
needed) and a :class:`RigidBodyAttributesCfg` for mass and friction. ``cube_a``
is placed slightly higher than ``cube_b`` so that, once detached, the lower
cube lands first and the relative pose visibly changes.

.. literalinclude:: ../../../scripts/tutorials/sim/create_rigid_constraint.py
	:language: python
	:start-at: cube_a = sim.add_rigid_object(
	:end-at: print("[INFO]: Scene setup complete with two cubes (cube_a, cube_b).")


Attaching the cubes
-------------------

The two cubes are welded with :meth:`SimulationManager.create_rigid_constraint`.
A :class:`RigidConstraintCfg` names the constraint and points at the two object
UIDs. ``local_frame_a`` / ``local_frame_b`` default to ``None``, so the
constraint welds the cubes at their *current* relative pose: ``local_frame_a``
defaults to identity (object A's origin) and ``local_frame_b`` is computed from
the objects' current poses so that the offset is preserved rather than the two
origins being pulled together. Pass explicit ``(4, 4)`` matrices — or an
``(N, 4, 4)`` array for one frame per arena — to define a specific joint frame
instead.

.. literalinclude:: ../../../scripts/tutorials/sim/create_rigid_constraint.py
	:language: python
	:start-at: constraint = sim.create_rigid_constraint(
	:end-at: print("[INFO]: Created constraint 'cube_weld' between cube_a and cube_b.")

While attached, the cubes' relative pose stays essentially constant across
physics steps because the solver enforces the constraint. (``constraint.get_relative_transform()``
returns the constraint-frame transform, which is ~0 while the constraint is
satisfied; the tutorial instead prints the bodies' relative z, ``cube_b.z -
cube_a.z``, to make the held offset visible.)


Removing the constraint
-----------------------

The constraint is removed by name with
:meth:`SimulationManager.remove_rigid_constraint`. After removal,
:meth:`SimulationManager.get_rigid_constraint` returns ``None``, and the two
cubes are independent again — their relative pose is no longer enforced and
will drift as they interact with gravity and the ground.

.. literalinclude:: ../../../scripts/tutorials/sim/create_rigid_constraint.py
	:language: python
	:start-at: sim.remove_rigid_constraint("cube_weld")
	:end-at: print("\n[INFO]: Removed constraint 'cube_weld'. cube_a and cube_b are now free.")

.. attention::

   ``remove_rigid_constraint`` accepts an ``env_ids`` argument, so in a
   vectorized simulation you can detach a subset of arenas while leaving the
   rest attached. Likewise, ``create_rigid_constraint`` accepts ``env_ids`` to
   attach only specific arenas.


Using the constraint from a task environment
--------------------------------------------

Inside a Gym environment the same operations are triggered on demand through
event functors registered under custom modes. A task wires up the attach and
detach functors, then calls ``event_manager.apply`` when its own logic decides
(for example, when a gripper closes or opens):

.. code-block:: python

    from embodichain.lab.gym.envs.managers.cfg import EventCfg, SceneEntityCfg
    from embodichain.lab.gym.envs.managers.events import (
        create_rigid_constraint,
        remove_rigid_constraint,
    )
    from embodichain.utils import configclass

    @configclass
    class MyTaskEventsCfg:
        attach_objects: EventCfg = EventCfg(
            func=create_rigid_constraint,
            mode="attach",
            params={
                "obj_a_cfg": SceneEntityCfg(uid="cube_a"),
                "obj_b_cfg": SceneEntityCfg(uid="cube_b"),
                "name": "cube_weld",
            },
        )
        detach_objects: EventCfg = EventCfg(
            func=remove_rigid_constraint,
            mode="detach",
            params={"name": "cube_weld"},
        )

    # Triggered from the task's own step / reset logic:
    self.event_manager.apply(mode="attach", env_ids=gripping_env_ids)
    self.event_manager.apply(mode="detach", env_ids=released_env_ids)


Running the tutorial
~~~~~~~~~~~~~~~~~~~~

To run the script from the repository root:

.. code-block:: bash

	python scripts/tutorials/sim/create_rigid_constraint.py

You can pass flags such as ``--headless``, ``--num_envs <n>``, and
``--device <cpu|cuda>`` to customize the run. With the default settings the
script prints the cubes' relative z-position every 20 steps, first while
attached (held constant) and then after removal (free to drift).
