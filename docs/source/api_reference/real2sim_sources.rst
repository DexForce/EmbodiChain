:orphan:

Real2sim source foundations
==========================

These APIs normalize explicit real joint recordings and adapt reviewed units to
the existing candidate pipeline. They do not reconstruct scenes or certify
physical task success. EEF evidence is distinguished from independently measured
poses; the initial normalizer implements qpos authority only.

Data contracts
--------------

.. autosummary::

   embodichain.data_pipeline.real2sim

.. automodule:: embodichain.data_pipeline.real2sim
   :members: EefPoseSamples, MimicJoint, Real2SimTrajectory, RobotJointLayout, SourcePhase

.. automodule:: embodichain.data_pipeline.real2sim.contracts
   :members: SourcePhase, MimicJoint, RobotJointLayout, Real2SimTrajectory, EefPoseSamples

Normalization and timing
------------------------

.. automodule:: embodichain.data_pipeline.real2sim.normalization
   :members: Real2SimNormalizationCfg, NormalizedReal2SimUnit, normalize_real2sim

.. automodule:: embodichain.compute.trajectory.anchored
   :members: AnchoredTimeMap, resample_with_anchors

Candidate adaptation
--------------------

.. autoclass:: embodichain.lab.sim.motion.expansion.Real2SimSourceAdapter
   :members:

.. automodule:: embodichain.lab.sim.motion.expansion.real2sim
   :members: Real2SimSourceAdapter

Nominal plan integration
------------------------

A trusted nominal source supplies a complete ActionPlan before ordinary motion
planning. The engine still validates and snapshots its result. The session
retains the source during recovery and invocation revision. Existing plan
transforms run after nominal-plan validation. Registered source catalogs,
configured source services and real2sim-specific Pick assembly are separate
follow-up work; the hook itself does not certify their capabilities.

.. autodata:: embodichain.lab.sim.atomic_actions.NominalPlanProvider

.. automodule:: embodichain.lab.task_program.runtime.expansion
   :members: TaskProgramNominalPlanFactory
