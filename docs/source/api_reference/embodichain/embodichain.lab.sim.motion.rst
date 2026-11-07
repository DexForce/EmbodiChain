embodichain.lab.sim.motion
==========================

.. automodule:: embodichain.lab.sim.motion

The ``motion`` package groups the robot motion capabilities used by simulation
objects, atomic actions, and environment integrations. Import the required
capability from its owning subpackage:

.. code-block:: python

   from embodichain.lab.sim.motion.solvers import SolverCfg
   from embodichain.lab.sim.motion.motion_generator import MotionGenerator, MotionGenCfg
   from embodichain.lab.sim.motion.execution import play_joint_trajectory
   from embodichain.lab.sim.motion.workspace import RobotWorkspaceCfg
   from embodichain.lab.sim.motion.expansion import TrajectoryTemplate

The parent package exposes its subpackages and ``motion_generator`` module
through lazy attributes;
it does not eagerly import them or re-export their classes and functions.
This keeps their initialization boundaries separate: robot configuration can
use solver and workspace contracts without ``motion`` also initializing the
planner layer. Normal ``lab`` and ``sim`` package initialization still applies.

Solvers own FK, IK, and differential kinematics; planners turn targets into
paths and timed trajectories; execution maps timed joint paths onto a fixed
standalone simulation cadence; workspace provides offline reachability analysis
and runtime cache queries; trajectory augmentation owns candidate variation,
coverage, and generation accounting. Environment rollout, reset, task
validation, and dataset persistence belong to explicit host integrations.

.. currentmodule:: embodichain.lab.sim.motion

.. autosummary::
   :nosignatures:

   motion_generator
   mesh_collision
   execution
   solvers
   planners
   workspace
   expansion

Discrete native motion queries
------------------------------

The B=1 Default query borrows the exact simulation-owned Robot and captures all
ordered native collision parts. It preserves cooked box/convex shape semantics,
local poses and scale. Full named-joint FK consumes measured mimic states; the
observed robot root is composed once. URDF data supplies topology and explicit
neighbour/mount exclusions, rather than replacement geometry or projected joint
states. Actual triangle meshes, incomplete/disabled shapes and uncertain native
predicates are unavailable.

``SimulationFreeMotionQuery.inspect`` captures a fresh rigid world and checks
every supplied full-robot sample. The scoped robot-only query validates its
captured observation, solver and model identity and closes owned Newton resources
on scope exit, including cancellation. Neither query steps physics, checks a
continuous swept volume, or certifies acquisition or task success. Controlled
providers leave backend and geometry-source fields unset; only an actual native
provider supplies those identities.

.. currentmodule:: embodichain.lab.sim.motion.mesh_collision

.. autosummary::

   MotionCollisionReport
   FreeMotionQuery
   SimulationFreeMotionQuery
   RobotMeshCollisionSnapshot
   robot_collision_query_scope
   inspect_robot_collision

.. autoclass:: MotionCollisionReport
   :members:

.. autoclass:: FreeMotionQuery
   :members:

.. autoclass:: SimulationFreeMotionQuery
   :members:

.. autoclass:: RobotMeshCollisionSnapshot
   :members:

.. autofunction:: robot_collision_query_scope

.. autofunction:: inspect_robot_collision

.. currentmodule:: embodichain.lab.sim.motion

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.motion.motion_generator
   embodichain.lab.sim.motion.execution
   embodichain.lab.sim.motion.solvers
   embodichain.lab.sim.motion.planners
   embodichain.lab.sim.motion.workspace
   embodichain.lab.sim.motion.expansion
