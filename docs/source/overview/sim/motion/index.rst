Robot Motion
============

Use :doc:`MotionGenerator <motion_generator>` to turn joint or end-effector
goals into timed trajectories. It combines trajectory planners with kinematic
solvers; simulation playback and Atomic Skills execute the resulting motion.

Choose a capability
-------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Goal
     - Start here
   * - Generate a trajectory from robot goals
     - :doc:`motion_generator`: strategies, batched inputs, results, and playback.
   * - Choose a timing or collision-aware planning backend
     - :doc:`planners/index`: backend selection and configuration.
   * - Compute forward or inverse kinematics
     - :doc:`solvers/index`: solver selection and shared FK/IK conventions.
   * - Analyze reachability or sample reachable poses
     - :doc:`/features/workspace_analyzer/index`: workspace analysis and caches.
   * - Generate variations of an expert trajectory
     - :doc:`Trajectory augmentation API </api_reference/embodichain/embodichain.lab.sim.motion.expansion>`:
       candidates, variation operators, coverage, and generation budgets.
   * - Collect several ways of executing one fixed set of waypoints
     - :doc:`trajectory_variants`: path, posture, timing, and approach variation
       for imitation learning and reinforcement-learning post-training.

Trajectory augmentation requires host integrations for scene reset, planning,
rollout, task validation, and dataset persistence.

Import APIs from their owning module, such as
``embodichain.lab.sim.motion.motion_generator`` or
``embodichain.lab.sim.motion.planners``. See the
:doc:`motion API </api_reference/embodichain/embodichain.lab.sim.motion>`
for the complete public interface.

.. toctree::
   :maxdepth: 1

   motion_generator
   trajectory_variants
   planners/index
   solvers/index
