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
   InitialStatePort
   MeasuredExecutor
   EpisodeSink
   LocalArtifactSink
   FixedSceneInitialStatePort
   SingleSlotOutcome
   SingleSlotRunner
   MultiSlotRunner

.. autoclass:: JointTrajectoryPlaybackCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autofunction:: play_joint_trajectory

Generation host integration
---------------------------

The single-slot runner coordinates injected restore, measured-execution, and
persistence ports. Candidate identity, budgets, coverage, and receipts remain
owned by :class:`embodichain.lab.sim.motion.expansion.GenerationSession`.
``LocalArtifactSink`` is the synchronous local implementation: it atomically
publishes one candidate directory and returns a receipt tied to the episode
and commit IDs.
``MultiSlotRunner`` adds generic compatible-slot reservation and exact release
around the existing runner; its current drain policy is synchronous FIFO.

.. autoclass:: InitialStatePort
   :members:

.. autoclass:: MeasuredExecutor
   :members:

.. autoclass:: EpisodeSink
   :members:

.. autoclass:: LocalArtifactSink
   :members:

.. autoclass:: FixedSceneInitialStatePort
   :members:

.. autoclass:: SingleSlotOutcome
   :members:

.. autoclass:: SingleSlotRunner
   :members:

.. autoclass:: MultiSlotRunner
   :members:
