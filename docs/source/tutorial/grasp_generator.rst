Generating and Executing Robot Grasps
======================================

.. currentmodule:: embodichain.lab.sim

This tutorial demonstrates how to generate antipodal grasp poses for a target object and execute a full grasp trajectory with a robot arm. It covers scene initialization, robot and object creation, interactive grasp region annotation, grasp pose computation, and trajectory execution in the simulation loop.

The Code
~~~~~~~~

The tutorial corresponds to the ``grasp_generator.py`` script in the ``scripts/tutorials/grasp`` directory.

.. dropdown:: Code for grasp_generator.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/grasp/grasp_generator.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Configuring the simulation
--------------------------

Command-line arguments are parsed with ``argparse`` to select the number of parallel environments, the compute device, and optional rendering features such as ray tracing and headless mode.

.. literalinclude:: ../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def parse_arguments():
   :end-at: return parser.parse_args()

The parsed arguments are passed to ``initialize_simulation``, which builds a :class:`SimulationManagerCfg` and creates the :class:`SimulationManager` instance. When ray tracing is enabled a directional :class:`cfg.LightCfg` is also added to the scene.

.. literalinclude:: ../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def initialize_simulation(args) -> SimulationManager:
   :end-at: return sim

Annotating and computing grasp poses
-------------------------------------

Grasp generation is performed by :meth:`objects.RigidObject.get_grasp_pose`, which internally runs an antipodal sampler on the object mesh. A :class:`toolkits.graspkit.pg_grasp.GraspAnnotatorCfg` controls sampler parameters (sample count, gripper jaw limits) and the interactive annotation workflow:

1. Open the visualization in a browser at the reported port (e.g. ``http://localhost:11801``).
2. Use *Rect Select Region* to highlight the area of the object that should be grasped.
3. Click *Confirm Selection* to finalize the region.

The function returns a batch of ``(N_envs, 4, 4)`` homogeneous transformation matrices representing candidate grasp frames in the world coordinate system.

For each grasp pose, gripper approach direction in world coordinate is required to compute the antipodal grasp. In this tutorial, we use a fixed approach direction (straight down in world frame) for simplicity, but it can be customized based on the task or object geometry.

.. literalinclude:: ../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: # get mug grasp pose
   :end-at: logger.log_info(f"Get grasp pose cost time: {cost_time:.2f} seconds")


The Code Execution
~~~~~~~~~~~~~~~~~~

To run the script, execute the following command from the project root:

.. code-block:: bash

   python scripts/tutorials/grasp/grasp_generator.py

A simulation window will open showing the robot and the mug. A browser-based visualizer will also launch (default port ``11801``) for interactive grasp region annotation.

You can customize the run with additional arguments:

.. code-block:: bash

   python scripts/tutorials/grasp/grasp_generator.py --num_envs <n> --device <cuda/cpu> --enable_rt --headless

After confirming the grasp region in the browser, the script will compute a grasp pose, print the elapsed time, and then wait for you to press **Enter** before executing the full grasp trajectory in the simulation. Press **Enter** again to exit once the motion is complete.
