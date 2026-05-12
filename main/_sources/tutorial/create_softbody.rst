Creating a soft-body simulation
===============================

.. currentmodule:: embodichain.lab.sim

This tutorial shows how to create a soft-body simulation using ``SimulationManager``. It covers the setup of the simulation context, adding a deformable mesh (soft object), and running the simulation loop.

The Code
~~~~~~~~

The tutorial corresponds to the ``create_softbody.py`` script in the ``scripts/tutorials/sim`` directory.

.. dropdown:: Code for create_softbody.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/create_softbody.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Configuring the simulation
--------------------------

The first step is to configure the simulation environment. This is done using the :class:`SimulationManagerCfg` data class, which allows you to specify parameters like window dimensions, headless mode, physics timestep, simulation device (CPU/GPU), and rendering options like ray tracing. Reminded that soft body simulation can only run on cuda deive.


.. literalinclude:: ../../../scripts/tutorials/sim/create_softbody.py
   :language: python
   :start-at: # Configure the simulation
   :end-at:     print("[INFO]: Scene setup complete!")

Adding a soft body to the scene
-------------------------------

With the simulation context created, we can add a soft (deformable) object. This tutorial demonstrates adding a soft-body cow mesh to the scene using the :meth:`SimulationManager.add_soft_object` method. The object's geometry and physical parameters are defined through configuration objects:

- :class:`cfg.MeshCfg` for the mesh shape (``cow.obj``)
- :class:`cfg.SoftbodyVoxelAttributesCfg` for voxelization and simulation mesh resolution
- :class:`cfg.SoftbodyPhysicalAttributesCfg` for material properties (Young's modulus, Poisson's ratio, density, frictions, solver iterations)

.. literalinclude:: ../../../scripts/tutorials/sim/create_softbody.py
   :language: python
   :start-at: # add softbody to the scene
   :end-at: print("[INFO]: Add soft object complete!")

The Code Execution
~~~~~~~~~~~~~~~~~~

To run the script and see the result, execute the following command:

.. code-block:: bash

   python scripts/tutorials/sim/create_softbody.py

A window should appear showing a soft-body cow mesh falling onto a ground plane. To stop the simulation, you can either close the window or press ``Ctrl+C`` in the terminal.

You can also pass arguments to customize the simulation. For example, to run in headless mode with ``n`` parallel environments using the specified device:

.. code-block:: bash

   python scripts/tutorials/sim/create_softbody.py --headless --num_envs <n> --device <cuda/cpu>

Now that we have a basic understanding of how to create a soft-body scene, let's move on to more advanced topics.
