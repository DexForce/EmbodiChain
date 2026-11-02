:orphan:

Real2Sim kinematic trajectory refinement
========================================

Refinement repairs a scene-instantiated nominal Cartesian path before Atomic
Skills freeze their expansion baseline. The source evidence, control period,
phase events and gripper semantics remain separately traceable. A bounded
anchor warp preserves gravity-aligned translation; sparse fixed-grid fitting
retains exact contact/hold constraints and phase-edge derivative stencils.
Robot IK, motion budgets, collision evidence and measured task acceptance are
separate checks. This API does not estimate robot calibration or contact forces.

Numerical pose fitting
----------------------

.. currentmodule:: embodichain.compute.trajectory.refinement

.. autosummary::

   PoseRefinementResult
   refine_pose_trajectory

.. automodule:: embodichain.compute.trajectory.refinement
   :members:

Grounded interaction constraints
--------------------------------

.. autosummary::

   embodichain.lab.sim.motion.refinement

.. currentmodule:: embodichain.lab.sim.motion.refinement

.. autosummary::

   MotionRefinementCfg
   InteractionConstraintSet
   MotionRefinementResult
   KinematicMotionRefiner

.. automodule:: embodichain.lab.sim.motion.refinement
   :members:

Configured integration
----------------------

Each ``runtime_services.nominal_motion_sources`` entry may declare a closed
``refinement`` mapping containing ``MotionRefinementCfg`` fields. The source
must explicitly use ``eef_pose`` authority. Pick requires an antipodal object
mesh and a calibrated parallel-jaw generator, including for a configured grasp.
Place requires the tabletop host's physical support validator. A failed repair
does not fall back to source replay or native planning.

The source provider returns the existing ``InstantiatedSkillMotion``. Its
``refinement_report`` and original/refined TCP evidence are copied into typed
episode trace attachments. The synchronous sink checks their digest, sample
count and measured correction budget before confirming persistence.
