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

Explicit seeded IK services
---------------------------

These services accept an explicitly selected Robot, control part and observed
poses. They solve and qualify paths without creating a simulator, executing
commands or establishing a physical task effect. Source paths retain their B=1
boundary, ordered seeds, authored joint limits and original time grids.

.. automodule:: embodichain.lab.sim.motion.ik_branches
   :members:

``CpuIKBranchSearch.capture`` reads the declared solver root once. Its fixed-seed
CPU search preserves the original RNG scope: ``torch.manual_seed`` also reseeds
visible CUDA streams, so the legacy CPU-only fork does not promise CUDA isolation.
The supplied-root constructor avoids an additional live root-pose read.

.. automodule:: embodichain.lab.sim.motion.goal_seeded
   :members:

Endpoint and path qualification use the configured full-pose validation budgets.
The callbacks and original mandatory samples retain their source order; these
checks provide no physical attachment, collision or acceptance certificate.

.. automodule:: embodichain.lab.sim.motion.solvers.tolerances
   :members:

Explicit CPU worker scope
------------------------

This opt-in context owns process-wide Torch intra-op settings for an isolated
worker. Capture capacity before applying serial overrides. Large CPU
PytorchSolver pose products can reuse the captured bounded thread capacity;
ordinary runtime calls outside the scope perform no thread-setting changes.
The owner must serialize concurrent use of these process-wide settings.

.. automodule:: embodichain.lab.sim.motion.cpu_threads
   :members:
