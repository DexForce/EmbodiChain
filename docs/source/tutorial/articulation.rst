.. _tutorial_load_articulation:

Loading an Articulation
=======================

.. currentmodule:: embodichain.lab.sim

This tutorial loads a URDF as a generic :class:`objects.Articulation`, inspects
its links and joints, and verifies the drive type applied to the constructed
physics entities. Generic articulations are passive by default: unlike
:class:`objects.Robot`, their joints use ``drive_type="none"`` unless a drive is
configured explicitly.

The Code
~~~~~~~~

The complete example is available in
``scripts/tutorials/sim/create_articulation.py``.

.. dropdown:: Code for create_articulation.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/create_articulation.py
      :language: python
      :linenos:

The Code Explained
~~~~~~~~~~~~~~~~~~

Configuring the simulation
--------------------------

Create a :class:`SimulationManager` in manual-update mode. The common launcher
arguments let the same script run on CPU or CUDA, in multiple environments, or
with native or browser visualization.

.. literalinclude:: ../../../scripts/tutorials/sim/create_articulation.py
   :language: python
   :start-at: # Configure the simulation.
   :end-at: sim = SimulationManager(sim_cfg)

Loading the URDF
----------------

Resolve the bundled drawer asset, then pass its path to
:class:`cfg.ArticulationCfg`. The example intentionally does not set
``joint_drive_props``. Therefore the configuration uses the Articulation default,
``drive_type="none"``. ``SimulationManager.add_articulation`` loads one drawer
into each configured environment and returns a batched
:class:`objects.Articulation` handle.

The URDF defines ``slide_rails`` over ``[0.0, 0.2]`` metres. The example also
sets ``qpos_limits={"slide_rails": [0.0, 0.18]}``, retaining 90% of the asset's
travel while keeping the fully closed position valid. This becomes the
effective physics limit used by both the backend and the force-control loop.

.. literalinclude:: ../../../scripts/tutorials/sim/create_articulation.py
   :language: python
   :start-at: # Resolve the drawer URDF
   :end-at: articulation: Articulation = sim.add_articulation(cfg=articulation_cfg)

Verifying the constructed drive type
------------------------------------

Checking ``articulation.cfg.joint_drive_props.drive_type`` confirms the requested
configuration, but it does not prove what the physics backend received. The
example therefore calls :meth:`objects.Articulation.get_joint_drive_type`,
which reads the drive type from every constructed DexSim entity. It raises an
error unless every joint in every environment is ``DriveType.NONE``.

.. literalinclude:: ../../../scripts/tutorials/sim/create_articulation.py
   :language: python
   :start-at: # Query the constructed DexSim entities
   :end-before: print(f"[INFO]: Loaded articulation

For the bundled drawer, the relevant output is:

.. code-block:: text

   [INFO]: Loaded articulation with 1 joint(s)
   [INFO]: Config drive type: none
   [INFO]: Backend drive types: [[<DriveType.NONE: ...>]]
   [INFO]: Effective qpos limits: tensor([[[0.0000, 0.1800]]])

The numeric enum value represented by ``...`` is backend-version dependent;
the semantic value is always ``DriveType.NONE``.

Opening and closing the drawer
------------------------------

The articulation state APIs are batched. For example,
:meth:`objects.Articulation.get_qpos` returns a tensor with shape
``(num_envs, dof)``, while :meth:`objects.Articulation.get_qpos_limits` returns
``(num_envs, dof, 2)``. The tutorial treats each lower limit as the closed
position and each upper limit as the open position.

Because this Articulation has ``drive_type="none"``, it does not track position
or velocity targets. Instead, the example uses :meth:`objects.Articulation.set_qf`
to apply a ``+1 N`` generalized joint force while opening and a ``-1 N`` force
while closing. Applying external effort does not change the passive drive type.

.. literalinclude:: ../../../scripts/tutorials/sim/create_articulation.py
   :language: python
   :start-at: def apply_drawer_force(
   :end-before: def run_simulation(

The simulation loop reads the current joint position on every step. When every
drawer instance reaches its upper limit, it reverses the force to close; at the
lower limit, it reverses again to open. The force is cleared before the loop
returns.

.. literalinclude:: ../../../scripts/tutorials/sim/create_articulation.py
   :language: python
   :start-at: def run_simulation(
   :end-before: def main

The finite run reports both limit transitions:

.. code-block:: text

   [INFO]: Applying +1.0 N to open the drawer
   [INFO]: Drawer reached open limit: tensor([[0.1800]])
   [INFO]: Applying -1.0 N to close the drawer
   [INFO]: Drawer reached closed limit: tensor([[0.]])

Run the interactive example from the repository root:

.. code-block:: bash

   python scripts/tutorials/sim/create_articulation.py

For a finite, headless CPU verification:

.. code-block:: bash

   python scripts/tutorials/sim/create_articulation.py \
       --headless \
       --device cpu \
       --max-steps 210

To inspect the articulation in a browser:

.. code-block:: bash

   python scripts/tutorials/sim/create_articulation.py --viser

Enabling a drive explicitly
---------------------------

Passive joints react to contacts, gravity, friction, and externally applied
forces, but they do not track position or velocity targets. If a generic
articulation needs an actuator, opt in with
:class:`cfg.JointDrivePropertiesCfg`:

.. code-block:: python

   from embodichain.lab.sim.cfg import ArticulationCfg, JointDrivePropertiesCfg

   articulation_cfg = ArticulationCfg(
       fpath="path/to/articulation.urdf",
       joint_drive_props=JointDrivePropertiesCfg(
           drive_type="force",
           stiffness=1.0e4,
           damping=1.0e3,
       ),
   )

For a controllable robot, prefer :class:`cfg.RobotCfg` and
:meth:`SimulationManager.add_robot`; robots default to ``drive_type="force"``.

.. attention::

   For file-backed assets, ``asset_physics_mode="preserve"`` keeps source
   physics, while ``"overlay"`` applies explicitly configured values.

Next Steps
~~~~~~~~~~

- :doc:`robot` — Load and control a driven robot
- :doc:`sensor` — Attach cameras and collect sensor data
- :doc:`/overview/sim/sim_articulation` — Articulation configuration and API reference
