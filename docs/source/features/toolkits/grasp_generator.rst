Parallel-Gripper Grasp Generation
=================================

.. currentmodule:: embodichain.lab.sim

The GraspKit toolkit generates feasible grasp poses for parallel-jaw grippers
from a target object's triangle mesh. It supports programmatic antipodal
sampling, browser-based grasp-region annotation, collision filtering, candidate
ranking, and on-disk caching of sampled contact pairs.

This page also demonstrates how to execute a generated pose with a robot arm. It
covers scene initialization, robot and object creation, grasp pose computation,
and trajectory execution in the simulation loop.

Processing Pipeline
-------------------

Grasp generation has three stages:

1. Sample surface points and find antipodal contact pairs on the full mesh or an
   annotated region.
2. Construct 6-DoF grasp frames that align the gripper opening axis with each
   contact pair and respect the requested approach direction.
3. Remove candidates that collide with the object or ground, rank the remaining
   poses, and return the best candidates with their required opening lengths.

Tutorial Source
---------------

The tutorial corresponds to the ``grasp_generator.py`` script in the ``scripts/tutorials/grasp`` directory.

.. dropdown:: Code for grasp_generator.py
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
      :language: python
      :linenos:


Tutorial Walkthrough
--------------------

Configuring the simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A UR10 arm with a parallel-jaw gripper is created via :meth:`SimulationManager.add_robot`. The gripper URDF and drive properties are configured so that the arm joints and finger joints can be controlled independently.

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def create_robot(sim: SimulationManager
   :end-at: return sim.add_robot(cfg=cfg)

The target object (a mug) is loaded as a :class:`objects.RigidObject` from a PLY mesh file:

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def create_obj(sim: SimulationManager):
   :end-at: return mug

Annotating and computing grasp poses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grasp generation is performed by
:class:`~embodichain.toolkits.graspkit.pg_grasp.AntipodalGraspPoseGenerator`.
It implements the robot-independent
:class:`~embodichain.toolkits.graspkit.GraspPoseGenerator` contract and the
shared
:class:`~embodichain.toolkits.graspkit.ParallelJawGraspPoseGenerator` base for
two-finger parallel-jaw grippers.

The generator owns the gripper model, algorithm, collision, and annotation
configuration. Target-local mesh vertices and triangles are supplied to each
call. This separation lets a handwritten environment call the service directly
and lets an ``AtomicActionEngine`` install the same instance under
``grasp_pose_generators={"hand": generator}``. Scene affordances do not own a
live generator or robot-specific parameters.

For each environment,
:meth:`~embodichain.toolkits.graspkit.pg_grasp.AntipodalGraspPoseGenerator.get_best_grasp_poses`
returns a success flag, a ``(4, 4)`` world-frame grasp pose, and the required
opening width. Antipodal contact pairs are cached and reused automatically.
Set ``GraspAnnotationCfg.selection_mode="interactive"`` to select a partial
region through Viser, or use ``"whole_mesh"`` for unattended generation.

The approach direction is the unit vector along which the gripper approaches the object. In this tutorial, we use a fixed approach direction (straight down in world frame) for simplicity, but it can be customized based on the task or object geometry.

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: # Construct one standalone generator.
   :end-at: logger.log_info(f"Get grasp pose cost time: {cost_time:.2f} seconds")

Building and executing the grasp trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a grasp pose is obtained, a waypoint trajectory is built that moves the arm from its rest configuration to an approach pose (offset above the grasp), down to the grasp pose, closes the fingers, lifts, and returns. The trajectory is interpolated for smooth motion and executed step-by-step in the simulation loop.

.. literalinclude:: ../../../../scripts/tutorials/grasp/grasp_generator.py
   :language: python
   :start-at: def get_grasp_traj(sim: SimulationManager
   :end-at: return interp_trajectory

Configuration
-------------

Configuration ownership is split by meaning:

* :class:`~embodichain.toolkits.graspkit.ParallelJawGripperModelCfg` describes
  physical opening limits, finger dimensions, and palm depth. A concrete EEF
  name belongs only in its ``model_id``.
* :class:`~embodichain.toolkits.graspkit.pg_grasp.AntipodalGraspPoseGeneratorCfg`
  controls sample count, angular deviations, approach variants, and result
  count.
* :class:`~embodichain.toolkits.graspkit.pg_grasp.ParallelJawGraspCollisionCfg`
  controls collision margin, point density, decomposition cost, and ground
  filtering.
* :class:`~embodichain.toolkits.graspkit.pg_grasp.GraspAnnotationCfg` controls
  whole-mesh versus interactive selection, the Viser port, and explicit cache
  refresh.


Running the Tutorial
--------------------

To run the script, execute the following command from the project root:

.. code-block:: bash

   python scripts/tutorials/grasp/grasp_generator.py

A simulation window will open showing the robot and the mug. The tutorial uses
whole-mesh annotation by default and therefore does not require a browser.

You can customize the run with additional arguments:

.. code-block:: bash

   python scripts/tutorials/grasp/grasp_generator.py --num_envs <n> --device <cuda/cpu> --renderer <auto|hybrid|fast-rt|offline-rt> --headless

The script computes a grasp pose, prints the elapsed time, and then waits for
you to press **Enter** before executing the full grasp trajectory. Press
**Enter** again to exit once the motion is complete.


Grasp Annotation CLI
--------------------

EmbodiChain provides a dedicated CLI for interactively annotating grasp regions on a mesh and caching the resulting antipodal point pairs, without requiring a full simulation environment.
The CLI constructs the same :class:`~embodichain.toolkits.graspkit.pg_grasp.AntipodalGraspPoseGenerator`
used by tasks and calls its :meth:`~embodichain.toolkits.graspkit.pg_grasp.AntipodalGraspPoseGenerator.prepare_mesh`
method; it does not use a separate annotation-time generator.

Basic usage::

   embodichain annotate-grasp --mesh_path /path/to/object.ply

This will:

1. Load the mesh file via ``trimesh``.
2. Launch a browser-based annotator (default port ``15531``).
3. Open ``http://localhost:15531`` in your browser, use *Rect Select Region* to highlight the graspable area, then click *Confirm Selection*.
4. Compute antipodal point pairs on the selected region and cache them to disk.

Common options::

   embodichain annotate-grasp \
       --mesh_path /path/to/object.ply \
       --viser_port 15531 \
       --n_sample 20000 \
       --max_length 0.1 \
       --min_length 0.001

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
