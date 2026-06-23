Simulation Framework
====================

.. currentmodule:: embodichain.lab.sim

The ``embodichain.lab.sim`` module is the runtime layer that connects robot
assets, physics, rendering, sensing, kinematics, and motion generation. It is
designed around a small set of composable components: a
:class:`SimulationManager` owns the simulation lifecycle, asset classes
represent objects in the scene, sensors produce batched observations, solvers
convert between joint space and task space, planners generate feasible
trajectories, and atomic actions package common manipulation skills.

Like EmbodiChain's environment and learning modules, the simulation framework is
configuration driven. Scene elements are declared through config classes, spawned
through the manager, and then stepped explicitly in a simulation loop. This makes
the same scene description usable for interactive visualization, scripted data
generation, and vectorized robot-learning environments.

Architecture
------------

The simulation stack can be read from the bottom up:

.. code-block:: text

    SimulationManager
    |-- global physics, rendering, arenas, stepping, USD import/export
    |-- assets
    |   |-- rigid objects and rigid object groups
    |   |-- articulations and robots
    |   |-- soft objects and cloth
    |   `-- lights, materials, shapes, and gizmos
    |-- sensors
    |   |-- cameras and stereo cameras
    |   `-- contact sensors
    |-- solvers
    |   |-- forward kinematics
    |   |-- inverse kinematics
    |   `-- differential kinematics
    |-- planners
    |   |-- joint-space and Cartesian trajectory generation
    |   `-- time parameterization and sampling utilities
    `-- atomic actions
        `-- reusable manipulation primitives built from assets, solvers, and planners

The :class:`SimulationManager` is the entry point for most workflows. It creates
the physics world, configures rendering and time stepping, lays out multiple
parallel arenas, and exposes ``add_*`` methods for scene construction. Every
asset and sensor is registered with the manager so that state updates, resets,
rendering, and batched queries remain synchronized.

Submodule Relationships
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Submodule
     - Responsibility
     - Relationship to other modules
   * - Simulation manager
     - Owns lifecycle, stepping, rendering, multiple arenas, and scene import/export.
     - Creates and coordinates assets, sensors, and physics state.
   * - Assets
     - Define the simulated entities: rigid objects, object groups,
       articulations, robots, soft bodies, cloth, lights, shapes, and
       materials.
     - Provide the state and control surfaces consumed by sensors, solvers,
       planners, environments, and atomic actions.
   * - Sensors
     - Produce perception data such as color, depth, segmentation, stereo disparity, and contacts.
     - Attach to world frames, robot links, or monitored bodies and return
       batched tensors for downstream policies or datasets.
   * - Solvers
     - Compute FK, IK, and differential kinematics for robots and articulations.
     - Translate task-space goals into joint-space commands used by planners,
       controllers, and actions.
   * - Planners
     - Generate joint-space or Cartesian trajectories with interpolation,
       timing, and feasibility handling.
     - Use robot state and solver results to produce trajectories that can be
       replayed in the manager loop.
   * - Atomic actions
     - Package complete manipulation primitives such as move, pick, and place.
     - Compose semantic targets, solvers, planners, and robot control into
       reusable higher-level skills.

Typical Data Flow
-----------------

A typical robot-learning or data-generation workflow follows this sequence:

1. Create a :class:`SimulationManager` from :class:`SimulationManagerCfg`.
2. Add assets such as objects, articulations, robots, lights, and materials.
3. Add sensors for camera, stereo, or contact observations.
4. Use solvers and planners to convert task goals into robot trajectories.
5. Step the simulation with :meth:`SimulationManager.update` and collect state or sensor tensors.
6. Wrap the same simulation logic in a Gym environment when training or evaluating agents.

For manipulation tasks, atomic actions can replace the lower-level solver and
planner calls. An action engine receives semantic targets or poses, resolves the
motion primitive sequence, and returns a trajectory that can be replayed in the
simulation.

Choosing Where to Start
-----------------------

- Start with :doc:`sim_manager` when creating a new simulation scene or learning
  how stepping, rendering, and parallel arenas work.
- Use :doc:`sim_assets` when adding physical entities, materials, lights, or USD
  assets. The asset pages underneath it cover each object family in detail.
- Use :doc:`sim_sensor` when adding camera, stereo, or contact observations.
- Use :doc:`solvers/index` when a robot needs FK, IK, or velocity-level
  kinematics.
- Use :doc:`planners/index` when a target pose or joint goal must become a
  time-ordered trajectory.
- Use :doc:`atomic_actions` when building scripted manipulation from reusable
  motion primitives.

Documentation Quality Notes
---------------------------

The pages in this section should stay organized around the same workflow:
configuration, construction through :class:`SimulationManager`, runtime state or
control APIs, and integration with sensors, planners, or Gym environments. When
adding new simulation documentation, include tensor shapes for batched data,
state the coordinate frame for poses and contacts, and link to the relevant
object, solver, or planner page instead of duplicating API tables.

See Also
--------

.. toctree::
   :maxdepth: 1

   sim_manager.md
   sim_assets.md
   sim_sensor.md
   solvers/index
   planners/index
   atomic_actions.md
