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
   execution
   solvers
   planners
   workspace
   expansion

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.motion.motion_generator
   embodichain.lab.sim.motion.execution
   embodichain.lab.sim.motion.solvers
   embodichain.lab.sim.motion.planners
   embodichain.lab.sim.motion.workspace
   embodichain.lab.sim.motion.expansion

Request-level joint limits
--------------------------

``motion.limits`` validates complete positive velocity/acceleration requests in
controlled-joint order. Scalars expand to the controlled width; vectors must
match it exactly. The resolver intersects requested speed with effective robot
caps, including mimic restrictions and the most restrictive selected scene row.
Acceleration is a planning budget unless reliable hardware caps are explicitly
supplied. This positive-limit contract rejects frozen joints, signed bounds and
heterogeneous B-by-J requests; legacy planner constraint APIs are unchanged.

``ResolvedMotionLimits`` retains requested, robot-effective and resolved values.
Its ``constraints`` property supplies those same effective tensors to a retimer
and validator; metadata records unknown caps without implying calibration. A
model owns its fixed normalization references and must serialize them instead
of changing scales when a request changes. This adapter has no policy/model
dependency and does not change the retiming solver.

.. currentmodule:: embodichain.lab.sim.motion.limits

.. autosummary::

   ResolvedMotionLimits
   resolve_motion_limits
   parse_motion_limits

.. autoclass:: ResolvedMotionLimits
   :members: constraints, metadata
   :exclude-members: __new__
   :no-autosummary:

.. autofunction:: resolve_motion_limits

.. autofunction:: parse_motion_limits
