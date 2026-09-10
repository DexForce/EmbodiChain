Robot Motion
============

The :mod:`embodichain.lab.sim.motion` package groups four related capabilities:
kinematic solvers, trajectory planners, workspace analysis, and expert trajectory
augmentation. Simulation objects and atomic actions use these capabilities to
translate robot goals into executable motion.

Choose the owning subpackage when importing an API. The ``motion`` parent loads
its children on access and does not re-export their classes and functions;
public imports still use the normal ``lab`` and ``sim`` initialization path.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Subpackage
     - Responsibility
   * - ``motion.solvers``
     - Forward, inverse, and differential kinematics.
   * - ``motion.planners``
     - Joint-space and Cartesian paths, collision-aware planning, and time
       parameterization.
   * - ``motion.workspace``
     - Offline reachability analysis, workspace caches, and runtime sampling.
   * - ``motion.expansion``
     - Explicit qpos templates and candidates, constrained geometric and timing
       variation, measured coverage, and bounded generation accounting.

Trajectory augmentation provides the core contracts and operators. The
:doc:`generation host layer </overview/trajectory_generation>` supplies
full-batch restoration, qpos execution, measured validation and confirmed
persistence for free motion and offline atomic PickUp sources. See the
:doc:`augmentation API </api_reference/embodichain/embodichain.lab.sim.motion.expansion>`
for the core boundaries.

Migrating Existing Imports
--------------------------

The previous packages have moved without compatibility aliases. Update Python
imports, string-based module references in configuration, and custom extensions:

.. list-table::
   :header-rows: 1

   * - Previous import
     - Current import
   * - ``embodichain.lab.sim.solvers``
     - ``embodichain.lab.sim.motion.solvers``
   * - ``embodichain.lab.sim.planners``
     - ``embodichain.lab.sim.motion.planners``
   * - ``embodichain.lab.sim.workspace``
     - ``embodichain.lab.sim.motion.workspace``

Their nested modules follow the same mapping. Solver ``class_type`` names such
as ``URSolver`` still resolve through ``RobotCfg.from_dict()``. Existing tests
and examples now live under ``tests/sim/motion/`` and ``examples/sim/motion/``.
Warp kinematics kernels live under ``embodichain.compute.kinematics._warp``;
``embodichain.utils.warp.kinematics`` retains compatibility exports.

.. toctree::
   :maxdepth: 1

   solvers/index
   planners/index

Workspace workflows are documented in
:doc:`/features/workspace_analyzer/index`. The complete public package surface is
listed in the :doc:`motion API </api_reference/embodichain/embodichain.lab.sim.motion>`.
