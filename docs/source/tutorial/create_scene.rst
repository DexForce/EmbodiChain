Creating a simulation scene
==========================

.. currentmodule:: embodichain.lab.sim

This tutorial shows how to create a basic simulation scene using SimulationManager. It covers the setup of the simulation context, adding rigid objects, and running the simulation loop.

The Code
~~~~~~~~

The tutorial corresponds to the ``create_scene.py`` script in the ``scripts/tutorials/sim`` directory.

.. dropdown:: Code for create_scene.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Configuring the simulation
--------------------------

The first step is to configure the simulation environment. This is done using the :class:`SimulationManagerCfg` data class, which allows you to specify various parameters like window dimensions, headless mode, physics timestep, simulation device (CPU/GPU), and rendering options like ray tracing.

Command-line arguments are parsed using ``argparse`` to allow for easy customization of the simulation from the terminal.

.. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
   :language: python
   :start-at: # Parse command line arguments
   :end-at: sim = SimulationManager(sim_cfg)

There are two kinds of physics mode in :class:`SimulationManager`:

- `manual`: The physics updates only when the user calls the :meth:`SimulationManager.update` function. This mode is used for robot learning tasks where precise control over simulation steps is required. Enabled by setting :meth:`SimulationManager.set_manual_update` to True.
- `auto`: The physics updates in a standalone thread, which enable asynchronous rendering and physics stepping. This mode is suitable for visualizations and demos for digital twins applications. This is the default mode.

Adding objects to the scene
---------------------------

With the simulation context created, we can add objects. This tutorial demonstrates adding a dynamic rigid cube to the scene using the :meth:`SimulationManager.add_rigid_object` method. The object's properties, such as its shape, initial position, and physics attributes (mass, friction, restitution), are defined through a configuration object, :class:`cfg.RigidObjectCfg`.

.. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
   :language: python
   :start-at: # Add objects to the scene
   :end-at: init_pos=[0.0, 0.0, 1.0],

Running the simulation
----------------------

The simulation is advanced through a loop in the ``run_simulation`` function. Before starting the loop, GPU physics is initialized if a CUDA device is used.

Inside the loop, :meth:`SimulationManager.update` is called to step the physics simulation forward. The script also includes logic to calculate and print the Frames Per Second (FPS) to monitor performance. The simulation runs until it's manually stopped with ``Ctrl+C``.

.. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
   :language: python
   :start-at: def run_simulation(sim: SimulationManager):
   :end-at: last_step = step_count

Exiting the simulation
----------------------

Upon exiting the simulation loop (e.g., by a ``KeyboardInterrupt``), it's important to clean up resources. The :meth:`SimulationManager.destroy` method is called in a ``finally`` block to ensure that the simulation is properly terminated and all allocated resources are released.

.. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
   :language: python
   :start-at: except KeyboardInterrupt:
   :end-at: sim.destroy()


The Code Execution
~~~~~~~~~~~~~~~~~~

To run the script and see the result, execute the following command:

.. code-block:: bash

   python scripts/tutorials/sim/create_scene.py

A window should appear showing a cube dropping onto a flat plane. To stop the simulation, you can either close the window or press ``Ctrl+C`` in the terminal.

You can also pass arguments to customize the simulation. For example, to run in headless mode with `n` parallel environments using specified device:

.. code-block:: bash

   python scripts/tutorials/sim/create_scene.py --headless --num_envs <n> --device <cuda/cpu>

Now that we have a basic understanding of how to create a scene, let's move on to more advanced topics.
