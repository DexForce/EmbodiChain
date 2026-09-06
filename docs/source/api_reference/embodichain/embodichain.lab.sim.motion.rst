embodichain.lab.sim.motion
==========================

.. automodule:: embodichain.lab.sim.motion

The ``motion`` package groups the robot motion capabilities used by simulation
objects, atomic actions, and environment integrations. Import the required
capability from its owning subpackage:

.. code-block:: python

   from embodichain.lab.sim.motion.solvers import SolverCfg
   from embodichain.lab.sim.motion.motion_generator import MotionGenerator, MotionGenCfg
   from embodichain.lab.sim.motion.workspace import RobotWorkspaceCfg
   from embodichain.lab.sim.motion.expansion import TrajectoryTemplate

The parent package exposes its subpackages and ``motion_generator`` module
through lazy attributes;
it does not eagerly import them or re-export their classes and functions.
This keeps their initialization boundaries separate: robot configuration can
use solver and workspace contracts without ``motion`` also initializing the
planner layer. Normal ``lab`` and ``sim`` package initialization still applies.

Solvers own FK, IK, and differential kinematics; planners turn targets into
paths and timed trajectories; workspace provides offline reachability analysis
and runtime cache queries; trajectory augmentation owns candidate variation,
coverage, and generation accounting. Physical rollout, reset, task validation,
and dataset persistence belong to explicit host integrations.

.. currentmodule:: embodichain.lab.sim.motion

.. autosummary::
   :nosignatures:

   motion_generator
   solvers
   planners
   workspace
   expansion

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.motion.motion_generator
   embodichain.lab.sim.motion.solvers
   embodichain.lab.sim.motion.planners
   embodichain.lab.sim.motion.workspace
   embodichain.lab.sim.motion.expansion
