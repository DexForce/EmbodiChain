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

``mask_generation_overspeed=True`` opts into a check of each fully normalized
generation result when ``control_part`` is supplied. It intersects ordered
runtime speed caps with the authored URDF caps and masks only success, retaining
positions, time intervals, derivatives and existing candidate metadata. The
interval tolerance is ``1e-5`` joint units, including zero-duration intervals.
The check follows resampling and existing failed-row normalization; newly
rejected proposals remain available to the caller. It applies once per result
and replaces the earlier fixed-grid speed mask when explicitly enabled. The
default remains disabled. This is separate from complete-action final-speed
qualification and does not enable retiming, recovery or acceleration validation.

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


Scoped revolute-source queries
------------------------------

These query/factory interfaces prepare source collision geometry independently
of action phases or task objects. Each B1/Default query retains exact physical,
source, model and observation ownership. Partial collision subtrees, changed
sources or a closed query reject rather than return clearance. Queries preserve
the original sampling/FK order and close only owned resources.

The reports distinguish unsupported inputs from supported decisions. Their
qualification is a source-input command-frame proxy and does not certify native
cooking, continuous collisions, contact or task success.

.. currentmodule:: embodichain.lab.sim.motion.revolute_query

.. autosummary::
   :nosignatures:

   RevolutePathQueryFactory
   RevolutePathQuery
   RevolutePathReport
   RevoluteDependencyReport

.. autoclass:: RevolutePathQueryFactory
   :members:

.. autoclass:: RevolutePathQuery
   :members:

.. autoclass:: RevolutePathReport
   :members:

.. autoclass:: RevoluteDependencyReport

Bounded drawer endpoint ranking
-------------------------------

The prefilter tests explicit outside/rim/release poses from the original seed,
retaining stage failure records and the bounded complete-plan candidate list.
Ranking does not certify a complete path; a failed prefilter still permits the
source's first ranked complete-planner attempt.

.. automodule:: embodichain.lab.sim.motion.drawer_ranking
   :members:
