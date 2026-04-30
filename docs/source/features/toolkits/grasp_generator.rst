Generating and Executing Robot Grasps
======================================

.. currentmodule:: embodichain.lab.sim

This tutorial demonstrates how to generate antipodal grasp poses for a target object and execute a full grasp trajectory with a robot arm. It covers scene initialization, robot and object creation, interactive grasp region annotation, grasp pose computation, and trajectory execution in the simulation loop.

The Code
~~~~~~~~

The tutorial corresponds to the ``grasp_generator.py`` script in the ``scripts/tutorials/grasp`` directory.

.. dropdown:: Code for grasp_generator.py
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Configuring the simulation
--------------------------

Command-line arguments are parsed with ``argparse`` to select the number of parallel environments, the compute device, and optional rendering features such as renderer backend and headless mode.

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def parse_arguments():
   :end-at: return parser.parse_args()

The parsed arguments are passed to ``initialize_simulation``, which builds a :class:`SimulationManagerCfg` and creates the :class:`SimulationManager` instance. When ray tracing is enabled a directional :class:`cfg.LightCfg` is also added to the scene.

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def initialize_simulation(args) -> SimulationManager:
   :end-at: return sim

Creating a robot and a target object
------------------------------------

A UR10 arm with a parallel-jaw gripper is created via :meth:`SimulationManager.add_robot`. The gripper URDF and drive properties are configured so that the arm joints and finger joints can be controlled independently.

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def create_robot(sim: SimulationManager
   :end-at: return sim.add_robot(cfg=cfg)

The target object (a mug) is loaded as a :class:`objects.RigidObject` from a PLY mesh file:

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def create_mug(sim: SimulationManager):
   :end-at: return mug

Annotating and computing grasp poses
-------------------------------------

Grasp generation is performed by :class:`~embodichain.toolkits.graspkit.pg_grasp.GraspGenerator`, which runs an antipodal sampler on the object mesh. The mesh data (vertices and triangles) is extracted from the :class:`objects.RigidObject` via its accessor methods. A :class:`~embodichain.toolkits.graspkit.pg_grasp.GraspGeneratorCfg` controls sampler parameters (sample count, gripper jaw limits) and the interactive annotation workflow:

1. Open the visualization in a browser at the reported port (e.g. ``http://localhost:11801``).
2. Use *Rect Select Region* to highlight the area of the object that should be grasped.
3. Click *Confirm Selection* to finalize the region.

After annotation, antipodal point pairs are cached to disk and automatically reused unless user call `GraspGenerator.annotate()`.

For each environment, a grasp pose is computed by calling :meth:`~embodichain.toolkits.graspkit.pg_grasp.GraspGenerator.get_grasp_poses` with the object pose and desired approach direction. The result is a ``(4, 4)`` homogeneous transformation matrix representing the grasp frame in world coordinates. Set ``visualize=True`` to open an Open3D window showing the selected grasp on the object.

The approach direction is the unit vector along which the gripper approaches the object. In this tutorial, we use a fixed approach direction (straight down in world frame) for simplicity, but it can be customized based on the task or object geometry.

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: gripper_collision_cfg = GripperCollisionCfg(
   :end-at: logger.log_info(f"Get grasp pose cost time: {cost_time:.2f} seconds")

Building and executing the grasp trajectory
-------------------------------------------

Once a grasp pose is obtained, a waypoint trajectory is built that moves the arm from its rest configuration to an approach pose (offset above the grasp), down to the grasp pose, closes the fingers, lifts, and returns. The trajectory is interpolated for smooth motion and executed step-by-step in the simulation loop.

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def get_grasp_traj(sim: SimulationManager
   :end-at: return interp_trajectory

Configuring GraspGeneratorCfg
------------------------------

:class:`~embodichain.toolkits.graspkit.pg_grasp.GraspGeneratorCfg` controls the overall grasp annotation workflow. The key parameters are listed below.

.. list-table:: GraspGeneratorCfg parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``viser_port``
     - ``15531``
     - Port used by the Viser browser-based visualizer for interactive grasp region annotation.
   * - ``use_largest_connected_component``
     - ``False``
     - When ``True``, only the largest connected component of the object mesh is used for sampling. Useful for meshes that contain disconnected fragments.
   * - ``antipodal_sampler_cfg``
     - ``AntipodalSamplerCfg()``
     - Nested configuration for the antipodal point sampler. See the table below for its parameters.
   * - ``max_deviation_angle``
     - ``π / 12``
     - Maximum allowed angle (in radians) between the specified approach direction and the axis connecting an antipodal point pair. Pairs that deviate more than this threshold are discarded.

The ``antipodal_sampler_cfg`` field accepts an :class:`~embodichain.toolkits.graspkit.pg_grasp.AntipodalSamplerCfg` instance, which controls how antipodal point pairs are sampled on the mesh surface.

.. list-table:: AntipodalSamplerCfg parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``n_sample``
     - ``20000``
     - Number of surface points uniformly sampled from the mesh before ray casting. Higher values yield denser coverage but increase computation time.
   * - ``max_angle``
     - ``π / 12``
     - Maximum angle (in radians) used to randomly perturb the ray direction away from the inward normal. Larger values increase diversity of sampled antipodal pairs. Setting this to ``0`` disables perturbation and samples strictly along surface normals.
   * - ``max_length``
     - ``0.1``
     - Maximum allowed distance (in metres) between an antipodal pair. Pairs farther apart than this value are discarded; set this to match the maximum gripper jaw opening width.
   * - ``min_length``
     - ``0.001``
     - Minimum allowed distance (in metres) between an antipodal pair. Pairs closer together than this value are discarded to avoid degenerate or self-intersecting grasps.

Configuring GripperCollisionCfg
-------------------------------

:class:`~embodichain.toolkits.graspkit.pg_grasp.GripperCollisionCfg` models the geometry of a parallel-jaw gripper as a point cloud and is used to filter out grasp candidates that would collide with the object. All length parameters are in metres.

.. list-table:: GripperCollisionCfg parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``max_open_length``
     - ``0.1``
     - Maximum finger separation of the gripper when fully open. Should match the physical gripper specification.
   * - ``finger_length``
     - ``0.08``
     - Length of each finger along the Z-axis (depth direction from the root). Should match the physical gripper specification.
   * - ``y_thickness``
     - ``0.03``
     - Thickness of the gripper body and fingers along the Y-axis (perpendicular to the opening direction).
   * - ``x_thickness``
     - ``0.01``
     - Thickness of each finger along the X-axis (parallel to the opening direction).
   * - ``root_z_width``
     - ``0.08``
     - Extent of the gripper root block along the Z-axis.
   * - ``point_sample_dense``
     - ``0.01``
     - Approximate number of sample points per unit length along each edge of the gripper point cloud. Higher values produce denser point clouds and improve collision-check accuracy at the cost of additional computation.
   * - ``max_decomposition_hulls``
     - ``16``
     - Maximum number of convex hulls used when decomposing the object mesh for collision checking. More hulls give a tighter shape approximation but increase cost.
   * - ``open_check_margin``
     - ``0.01``
     - Extra clearance added to the gripper open length during collision checking to account for pose uncertainty or mesh inaccuracies.


The Code Execution
~~~~~~~~~~~~~~~~~~

To run the script, execute the following command from the project root:

.. code-block:: bash

   python scripts/tutorials/grasp/grasp_generator.py

A simulation window will open showing the robot and the mug. A browser-based visualizer will also launch (default port ``11801``) for interactive grasp region annotation.

You can customize the run with additional arguments:

.. code-block:: bash

   python scripts/tutorials/grasp/grasp_generator.py --num_envs <n> --device <cuda/cpu> --renderer <legacy|hybrid|fast-rt|rt> --headless

After confirming the grasp region in the browser, the script will compute a grasp pose, print the elapsed time, and then wait for you to press **Enter** before executing the full grasp trajectory in the simulation. Press **Enter** again to exit once the motion is complete.


Grasp Annotation CLI
~~~~~~~~~~~~~~~~~~~~

EmbodiChain provides a dedicated CLI for interactively annotating grasp regions on a mesh and caching the resulting antipodal point pairs, without requiring a full simulation environment.

Basic usage::

   python -m embodichain annotate-grasp --mesh_path /path/to/object.ply

This will:

1. Load the mesh file via ``trimesh``.
2. Launch a browser-based annotator (default port ``15531``).
3. Open http://localhost:15531 in your browser, use *Rect Select Region* to highlight the graspable area, then click *Confirm Selection*.
4. Compute antipodal point pairs on the selected region and cache them to disk.

Common options::

   python -m embodichain annotate-grasp \
       --mesh_path /path/to/object.ply \
       --viser_port 15531 \
       --n_sample 20000 \
       --max_length 0.1 \
       --min_length 0.001 \

.. list-table:: CLI options
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``--mesh_path``
     - *(required)*
     - Path to the mesh file (``.ply``, ``.obj``, ``.stl``, etc.).
   * - ``--viser_port``
     - ``15531``
     - Port for the browser-based annotation UI.
   * - ``--n_sample``
     - ``20000``
     - Number of surface points to sample for antipodal pair detection.
   * - ``--max_length``
     - ``0.1``
     - Maximum distance (metres) between antipodal pairs; should match the gripper's maximum opening width.
   * - ``--min_length``
     - ``0.001``
     - Minimum distance (metres) between antipodal pairs; filters out degenerate pairs.
   * - ``--device``
     - ``cpu``
     - Compute device (``cpu`` or ``cuda``).
