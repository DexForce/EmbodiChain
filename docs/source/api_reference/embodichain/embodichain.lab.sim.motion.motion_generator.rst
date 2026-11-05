embodichain.lab.sim.motion.motion_generator
===========================================

.. automodule:: embodichain.lab.sim.motion.motion_generator

Motion generation coordinates planner backends, IK interpolation, collision-world
options and normalized trajectory results. Import its configuration and strategy
options from this module; backend contracts remain under ``motion.planners``.

.. currentmodule:: embodichain.lab.sim.motion.motion_generator

.. autosummary::
   :nosignatures:

   MotionGenerator
   MotionGenCfg
   MotionGenOptions

Motion Generator
----------------

.. autoclass:: MotionGenCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: MotionGenerator
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: MotionGenOptions
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

Explicit linear keyframes and bounded recovery
----------------------------------------------

``MotionGenerator.generate`` is the common entry for backend planning and IK
interpolation. With ``strategy="ik_interp"``, ``interpolation_mode="fixed_grid"``,
``ik_seed_mode="sequential"`` and ``is_linear=True``, callers may supply unsampled
end-effector keyframes. The generator samples each required interval before IK.
``preserve_cartesian_samples=True`` instead retains already-sampled input;
segment recovery does not apply to that route or custom path samplers.

``segment_sample_count_retries`` lists ordered denser alternatives and
``segment_continuous_ik_recovery`` enables a final continuous IK attempt on the
original grid. Both are disabled by default. Recovery remains B=1-only. Each
sample-count alternative retains the interval, so it may lengthen total duration;
continuous IK preserves the nominal targets and time grid. The first alternative
also retains the source eligibility boundary: an original count above it skips
both dense and continuous recovery. Exhaustion keeps the original failed result
and candidate evidence, rather than the last failed alternative.

.. code-block:: python

   from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
   from embodichain.lab.sim.motion.planners import PlanState

   options = MotionGenOptions(
       strategy="ik_interp", interpolation_mode="fixed_grid",
       ik_seed_mode="sequential", is_linear=True,
       start_qpos=start_qpos, control_part="arm",
       sample_count=100, interpolation_dt=0.04,
       segment_sample_count_retries=(260, 320),
       segment_continuous_ik_recovery=True,
   )
   result = generator.generate([PlanState.from_xpos(goal_pose)], options)
   attempted_counts = result.recovery.attempted_counts

The result remains a ``PlanResult``. ``result.recovery`` records actual local
attempts and whether continuity supplied the accepted path. Dense attempts do
not reset the random stream; continuous IK retains its existing Torch-only fork.
``generate(..., defer_recovery=True)`` performs only the nominal linear attempt,
so the caller can inspect it without spending the remaining motion alternatives.
Ordinary calls, planner-owned retries, sampled input paths and complete-action
retry policies keep their existing owners and defaults. The internal helper is
not a second public generation entry. Geometric collision and physical-effect
acceptance remain caller-owned.


Bounded drawer endpoint ranking
-------------------------------

The prefilter tests explicit outside/rim/release poses from the original seed,
retaining stage failure records and the bounded complete-plan candidate list.
Ranking does not certify a complete path; a failed prefilter still permits the
source's first ranked complete-planner attempt.

.. automodule:: embodichain.lab.sim.motion.drawer_ranking
   :members:
