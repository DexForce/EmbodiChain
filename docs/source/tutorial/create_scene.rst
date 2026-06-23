Creating a simulation scene
==========================

.. currentmodule:: embodichain.lab.sim

This tutorial shows how to create a basic simulation scene using SimulationManager. It covers the setup of the simulation context, adding rigid objects, running the simulation loop, and exporting a video automatically when the example runs in headless mode.

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

Command-line arguments are parsed using ``argparse`` to allow for easy customization of the simulation from the terminal. In addition to the common launcher flags, this tutorial adds ``--record-steps``, ``--record-fps``, and ``--record-save-path`` for headless recording.

.. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
   :language: python
   :start-at: # Parse command line arguments
   :end-at: sim = SimulationManager(sim_cfg)

There are two kinds of physics mode in :class:`SimulationManager`:

- `manual`: The physics updates only when the user calls the :meth:`SimulationManager.update` function. This mode is used for robot learning tasks where precise control over simulation steps is required. Enabled by setting :meth:`SimulationManager.set_manual_update` to True.
- `auto`: The physics updates in a standalone thread, which enable asynchronous rendering and physics stepping. This mode is suitable for visualizations and demos for digital twins applications. This is the default mode.

Adding objects to the scene
---------------------------

With the simulation context created, we can add objects. This tutorial demonstrates adding a dynamic rigid cube and a chair mesh to the scene using the :meth:`SimulationManager.add_rigid_object` method. Their properties, such as shape, initial pose, and physics attributes (mass, friction, restitution), are defined through :class:`cfg.RigidObjectCfg`.

.. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
   :language: python
   :start-at: # Add cube object to the scene
   :end-before: print("[INFO]: Scene setup complete!")

Headless recording
------------------

When the script runs with ``--headless``, it uses :meth:`SimulationManager.start_window_record` with a fixed ``look_at`` camera pose. This is the same public recorder API used for viewer recording, but it now also supports headless execution without depending on a live window.

The example starts recording before the simulation loop, runs for ``--record-steps`` physics steps, then stops the recorder and waits for the video export to finish before destroying the simulation.

.. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
   :language: python
   :start-at: if args.headless:
   :end-at:         print(f"[INFO]: Running {args.record_steps} steps before exporting the video")

Running the simulation
----------------------

The simulation is advanced through a loop in the ``run_simulation`` function. Before starting the loop, GPU physics is initialized if a CUDA device is used.

Inside the loop, :meth:`SimulationManager.update` is called to step the physics simulation forward. The script also includes logic to calculate and print the Frames Per Second (FPS) to monitor performance. In GUI mode the simulation runs until it is manually stopped with ``Ctrl+C``. In headless mode the loop exits automatically after the configured number of recording steps.

.. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
   :language: python
   :start-at: def run_simulation(
   :end-before: except KeyboardInterrupt:

Exiting the simulation
----------------------

Upon exiting the simulation loop (e.g., by a ``KeyboardInterrupt``), it's important to clean up resources. The example stops any active recording, waits for the background video export to finish, then calls :meth:`SimulationManager.destroy` in a ``finally`` block to ensure that the simulation is properly terminated and all allocated resources are released.

.. literalinclude:: ../../../scripts/tutorials/sim/create_scene.py
   :language: python
   :start-at: except KeyboardInterrupt:
   :end-at:         print("[INFO]: Simulation terminated successfully")


The Code Execution
~~~~~~~~~~~~~~~~~~

To run the script and see the result, execute the following command:

.. code-block:: bash

   python scripts/tutorials/sim/create_scene.py

A window should appear showing a cube dropping onto a flat plane. To stop the simulation, you can either close the window or press ``Ctrl+C`` in the terminal.

You can also pass arguments to customize the simulation. For example, to run in headless mode with ``n`` parallel environments using the specified device:

.. code-block:: bash

   python scripts/tutorials/sim/create_scene.py --headless --num_envs <n> --device <cuda/cpu>

In headless mode, the script records a video and saves it under ``outputs/videos`` by default. You can control the exported clip length and destination:

.. code-block:: bash

   python scripts/tutorials/sim/create_scene.py \
       --headless \
       --record-steps 1000 \
       --record-fps 20 \
       --record-save-path outputs/videos/my_scene.mp4

Now that we have a basic understanding of how to create a scene, let's move on to more advanced topics.

Next Steps
~~~~~~~~~~

- :doc:`create_softbody` — Add deformable bodies to your scene
- :doc:`robot` — Load and control a robot
- :doc:`sensor` — Add cameras and capture sensor data
- :doc:`basic_env` — Create your first Gymnasium environment
- :doc:`/overview/sim/sim_manager` — Full SimulationManager API reference
