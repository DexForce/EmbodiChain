Video-selected Atomic Motion Reconstruction
===========================================

The fit-trajectory command maps an annotator's inclusive video interval to
explicit joint timestamps, estimates a motion goal and invokes the existing
Atomic Skill engine. PickUpMotionGoal, SlideMotionGoal, TwistMotionGoal and
HandOverMotionGoal accept explicit world TCP anchors without an object identity
or mesh. These goals declare no held-object effect. Collision evidence and
physical success remain unverified.

The fit preserves the original joint clock. A bounded monotone projection fits
the phase clock to the independently generated native path. Phase endpoints and
Pickup's internal pre-grasp anchor remain mandatory. Native reconstruction
errors are checked separately from the final boundary correction. Three
original samples at each outer boundary retain the original discrete
position/velocity/acceleration stencils. This does not establish continuous-time
smoothness of noisy measurements or physical task success.

The source joint order, calibration, open/grasp commands, derivative limits and
video/joint time offset must be explicit. HDF5 image timing derived from FPS is
identified as derived, not measured. Missing hand events can be supplied as a
complete segment-relative phase partition; video pixels are not automatically
interpreted as contact evidence.

Recording and persistence
-------------------------

.. automodule:: embodichain.data_pipeline.skill_fitting
   :members:

Native skill fitting
--------------------

.. automodule:: embodichain.lab.sim.atomic_skills.fitting
   :members:

Task Program execution bindings
-------------------------------

Use ``export_task_program`` to bind an accepted Pickup fit to a supplied
execution context and its explicitly assigned demonstration spans. The export
preserves the execution cadence and records how timed source events map to
compiled invocations. Use ``load_invocation_bindings`` to verify the sealed files
and validate the invocation policies against a fresh compile before execution.

.. automodule:: embodichain.toolkits.skill_fitting.task_program
   :members:

Numerical primitives
--------------------

.. automodule:: embodichain.compute.trajectory.phase_alignment
   :members:

.. automodule:: embodichain.compute.trajectory.splicing
   :members:

Command line
------------

.. automodule:: embodichain.toolkits.skill_fitting.cli
   :members:
