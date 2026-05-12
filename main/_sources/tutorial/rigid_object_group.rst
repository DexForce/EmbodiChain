Rigid object group tutorial
==========================

.. currentmodule:: embodichain.lab.sim

This tutorial shows how to create and use a `RigidObjectGroup` in SimulationManager.
It follows the style used in the `create_scene` tutorial and references the
example script located in ``scripts/tutorials/sim/create_rigid_object_group.py``.

The Code
~~~~~~~~

The tutorial corresponds to the ``create_rigid_object_group.py`` script in the
``scripts/tutorials/sim`` directory.

.. dropdown:: Code for create_rigid_object_group.py
	:icon: code

	.. literalinclude:: ../../../scripts/tutorials/sim/create_rigid_object_group.py
		:language: python
		:linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~


Adding a RigidObjectGroup
-------------------------

The key part of the tutorial demonstrates creating a ``RigidObjectGroup`` via
``sim.add_rigid_object_group``. The group is configured with a mapping of
object UIDs to ``RigidObjectCfg`` entries. Each entry defines a shape
(here ``CubeCfg``), physics attributes, and initial pose.

.. literalinclude:: ../../../scripts/tutorials/sim/create_rigid_object_group.py
	:language: python
	:start-at: obj_group: RigidObjectGroup = sim.add_rigid_object_group(
	:end-at: print("[INFO]: Scene setup complete!")


Running the tutorial
~~~~~~~~~~~~~~~~~~~~

To run the script from the repository root:

.. code-block:: bash

	python scripts/tutorials/sim/create_rigid_object_group.py

You can pass flags such as ``--headless``, ``--num_envs <n>``, and
``--device <cpu|cuda>`` to customize the run.
