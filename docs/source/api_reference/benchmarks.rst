Motion generation benchmarks
============================

The bundled ``embodichain_benchmarks.motion_generation`` extension evaluates
Motion Planning and Atomic Skills suites from another Python project. The
benchmark owns case generation, simulation, lifecycle timing, evaluation and
artifact persistence. It consumes the simulation, planning and Atomic Skill
APIs in ``embodichain`` and ships in the main wheel with the same version and
dependencies. Callers select a suite and provide planner configuration.

Calling from another project
----------------------------

.. code-block:: python

   from embodichain_benchmarks.motion_generation import (
       PlannerSpecCfg,
       load_suite,
       run_benchmark,
   )

   suite = load_suite("motion_panda_free_space_smoke")
   suite.protocol.measured_trials = 5
   result = run_benchmark(
       suite,
       planners=[PlannerSpecCfg(id="toppra", adapter="toppra")],
       device="cuda",
       output_root="./evaluation_results",
   )
   print(result.aggregates)
   print(result.report_path)

Installed suite resources resolve independently of the working directory. An
external YAML file resolves its component references relative to that file.
The caller's suite and planner specifications are copied. Explicit ``planners``
replace the suite's planner list and are enabled; omitting the argument selects
the suite's enabled algorithms.

Pass ``adapters={"my_method": MyPlannerAdapter}`` and a matching
``PlannerSpecCfg(adapter="my_method", ...)`` to evaluate a class defined in
another repository. The mapping applies to one call, including overrides of
registered adapter names. Implement ``build()``, ``plan(case)`` and ``close()``;
use ``prepare()`` for distinct lazy initialization. Declare supported
capabilities and return a canonical batched ``PlanResult``. Atomic Skills also
require an exposed ``MotionGenerator`` and the ``atomic_skill`` capability.

Calls own their simulation and return structured results after teardown. Run
calls serially within a process. Install the selected backend and resolve its
robot assets as for ordinary EmbodiChain use. The installed command is
``embodichain benchmark motion-generation``; the module form is
``python -m embodichain_benchmarks.motion_generation``.

Public entry points
-------------------

.. currentmodule:: embodichain_benchmarks.motion_generation

.. autosummary::

   BenchmarkCase
   BenchmarkRunResult
   PlannerAdapter
   PlannerContext
   PlannerSpecCfg
   SuiteCfg
   load_suite
   run_all_benchmarks
   run_benchmark

.. autofunction:: embodichain_benchmarks.motion_generation.run_benchmark

.. autofunction:: embodichain_benchmarks.motion_generation.load_suite

.. autoclass:: embodichain_benchmarks.motion_generation.BenchmarkRunResult

.. autoclass:: embodichain_benchmarks.motion_generation.PlannerAdapter
   :members:

Implementation reference
------------------------

The modules below expose the suite schema, extension contracts, evaluators and
reporting helpers used by the shared runner. Motion Planning geometry and
solved-timing rankings remain separate; Atomic Skills use measured physical
success. Unsupported populations have explicit coverage semantics. Raw trials,
the frozen case manifest, resolved suite, environment metadata and aggregate
schema are persisted together with each report.


embodichain_benchmarks.motion_generation.aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.aggregation
   :members:
   :undoc-members:
   :exclude-members: aggregate_generalization

.. currentmodule:: embodichain_benchmarks.motion_generation.aggregation

.. autosummary::

   aggregate_generalization


embodichain_benchmarks.motion_generation.api
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.api

.. currentmodule:: embodichain_benchmarks.motion_generation.api

.. autosummary::

   run_benchmark


embodichain_benchmarks.motion_generation.artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.artifacts
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.atomic_skill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.atomic_skill
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.cli
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.config
   :members:
   :undoc-members:
   :exclude-members: load_suite

.. currentmodule:: embodichain_benchmarks.motion_generation.config

.. autosummary::

   load_suite


embodichain_benchmarks.motion_generation.contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.contracts
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.embodiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.embodiments
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.evaluation
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.generalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.generalization
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.limits
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.metrics

.. currentmodule:: embodichain_benchmarks.motion_generation.metrics

.. autosummary::

   TimedCall
   compute_case_outcomes
   compute_waypoint_errors
   get_pose_err
   make_failure_outcomes
   match_ordered_waypoints
   timed_call


embodichain_benchmarks.motion_generation.metrics.performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.metrics.performance
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.metrics.stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.metrics.stats
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.metrics.trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.metrics.trajectory
   :members:
   :undoc-members:
   :exclude-members: resolve_dynamic_limit, resolve_velocity_limits

.. currentmodule:: embodichain_benchmarks.motion_generation.metrics.trajectory

.. autosummary::

   resolve_dynamic_limit
   resolve_velocity_limits


embodichain_benchmarks.motion_generation.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.models
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.planners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.planners

.. currentmodule:: embodichain_benchmarks.motion_generation.planners

.. autosummary::

   CuroboAdapter
   IkInterpolateAdapter
   NmgOnnxAdapter
   PlannerAdapter
   PlannerContext
   ToppraAdapter


embodichain_benchmarks.motion_generation.planners.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.planners.base
   :members:
   :undoc-members:
   :exclude-members: PlannerAdapter

.. currentmodule:: embodichain_benchmarks.motion_generation.planners.base

.. autosummary::

   PlannerAdapter


embodichain_benchmarks.motion_generation.planners.curobo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.planners.curobo
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.planners.ik_interpolate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.planners.ik_interpolate
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.planners.nmg_onnx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.planners.nmg_onnx
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.planners.toppra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.planners.toppra
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.registry
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.reporting
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.robots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.robots

.. currentmodule:: embodichain_benchmarks.motion_generation.robots

.. autosummary::

   DualArmProvider
   FrankaPandaProvider
   FrankaPgiProvider
   RobotProvider
   UR5PgiProvider
   UR5Provider


embodichain_benchmarks.motion_generation.robots.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.robots.base
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.robots.dual_arm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.robots.dual_arm
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.robots.franka
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.robots.franka
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.robots.ur5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.robots.ur5
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.runner
   :members:
   :undoc-members:
   :exclude-members: BenchmarkRunResult

.. currentmodule:: embodichain_benchmarks.motion_generation.runner

.. autosummary::

   BenchmarkRunResult


embodichain_benchmarks.motion_generation.scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios

.. currentmodule:: embodichain_benchmarks.motion_generation.scenarios

.. autosummary::

   AtomicObjectHandle
   AtomicSkillCaseProvider
   AtomicSkillScenario
   atomic_object_kind_names
   atomic_skill_provider_names
   create_atomic_object
   create_atomic_skill_provider
   FreeSpaceScenario
   register_atomic_object_kind
   register_atomic_skill_provider
   ScenarioEvaluation
   ScenarioProvider


embodichain_benchmarks.motion_generation.scenarios.atomic_case_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_case_utils
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.atomic_contact_cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_contact_cases
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.atomic_evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_evaluation
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.atomic_grasps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_grasps
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.atomic_manipulation_cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_manipulation_cases
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.atomic_motion_cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_motion_cases
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.atomic_objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_objects
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.atomic_policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_policy
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.atomic_replay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_replay
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.atomic_skill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.atomic_skill
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.base
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.scenarios.free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.scenarios.free_space
   :members:
   :undoc-members:


embodichain_benchmarks.motion_generation.video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain_benchmarks.motion_generation.video
   :members:
   :undoc-members:


embodichain.cli.benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain.cli.benchmark
   :members:
   :undoc-members:
