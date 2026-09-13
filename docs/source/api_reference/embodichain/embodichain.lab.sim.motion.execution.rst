embodichain.lab.sim.motion.execution
====================================

.. automodule:: embodichain.lab.sim.motion.execution

Standalone playback retimes planner output onto a fixed command period while
leaving the simulation physics period unchanged. The command period must be an
integer multiple of ``physics_dt``; position-plus-velocity targets are opt-in
and velocities are recomputed on the executed grid.

.. currentmodule:: embodichain.lab.sim.motion.execution

.. autosummary::
   :nosignatures:

   JointTrajectoryPlaybackCfg
   play_joint_trajectory

.. autoclass:: JointTrajectoryPlaybackCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autofunction:: play_joint_trajectory
