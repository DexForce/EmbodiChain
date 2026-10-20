Motion generation benchmarks
============================

The installed ``embodichain.benchmarks.motion_generation`` package evaluates
Motion Planning and Atomic Skills suites from another Python project. The
benchmark owns case generation, simulation, lifecycle timing, evaluation and
artifact persistence. Callers select a suite and provide planner configuration.

Calling from another project
----------------------------

.. code-block:: python

   from embodichain.benchmarks.motion_generation import (
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
require an exposed ``MotionGenerator`` and the ``atomic_action`` capability.

Calls own their simulation and return structured results after teardown. Run
calls serially within a process. Install the selected backend and resolve its
robot assets as for ordinary EmbodiChain use. The installed command is
``embodichain benchmark motion-generation``; the module form is
``python -m embodichain.benchmarks.motion_generation``.

Public entry points
-------------------

.. currentmodule:: embodichain.benchmarks.motion_generation

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

.. autofunction:: embodichain.benchmarks.motion_generation.run_benchmark

.. autofunction:: embodichain.benchmarks.motion_generation.load_suite

.. autoclass:: embodichain.benchmarks.motion_generation.BenchmarkRunResult

.. autoclass:: embodichain.benchmarks.motion_generation.PlannerAdapter
   :members:
   :no-index:

Implementation reference
------------------------

The modules below expose the suite schema, extension contracts, evaluators and
reporting helpers used by the shared runner. Motion Planning geometry and
solved-timing rankings remain separate; Atomic Skills use measured physical
success. Unsupported populations have explicit coverage semantics. Raw trials,
the frozen case manifest, resolved suite, environment metadata and aggregate
schema are persisted together with each report.


embodichain.benchmarks.motion_generation.aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregate raw planner trials and build a complete benchmark leaderboard.

.. automodule:: embodichain.benchmarks.motion_generation.aggregation
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.api
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Programmatic evaluation of installed suites and external planner adapters.

.. automodule:: embodichain.benchmarks.motion_generation.api
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reproducibility and raw-result artifacts for planner benchmarks.

.. automodule:: embodichain.benchmarks.motion_generation.artifacts
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.atomic_skill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Case construction contract for Atomic Skill benchmark integrations.

.. automodule:: embodichain.benchmarks.motion_generation.atomic_skill
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run Motion Planning and Atomic Skills suites through the shared benchmark.

.. automodule:: embodichain.benchmarks.motion_generation.cli
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration loading and hashing for motion-generation benchmark suites.

.. automodule:: embodichain.benchmarks.motion_generation.config
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulator-independent contracts shared by Atomic Skill benchmark owners.

.. automodule:: embodichain.benchmarks.motion_generation.contracts
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.embodiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resolve official embodiment components for benchmark metadata and runtime.

.. automodule:: embodichain.benchmarks.motion_generation.embodiments
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Versioned Atomic Skill criteria applied to measured, row-local evidence.

.. automodule:: embodichain.benchmarks.motion_generation.evaluation
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.generalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Domain-aware aggregation for configuration-driven benchmark suites.

.. automodule:: embodichain.benchmarks.motion_generation.generalization
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared benchmark motion budgets for planning and independent validation.

.. automodule:: embodichain.benchmarks.motion_generation.limits
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metric evaluators for motion-generation benchmarks.

.. automodule:: embodichain.benchmarks.motion_generation.metrics
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.metrics.performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wall-clock and process-memory measurement helpers.

.. automodule:: embodichain.benchmarks.motion_generation.metrics.performance
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.metrics.stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared numeric helpers for benchmark metric aggregation.

.. automodule:: embodichain.benchmarks.motion_generation.metrics.stats
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.metrics.trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ordered waypoint and free-space trajectory validity metrics.

.. automodule:: embodichain.benchmarks.motion_generation.metrics.trajectory
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typed records shared by the motion-generation benchmark modules.

.. automodule:: embodichain.benchmarks.motion_generation.models
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.planners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Built-in planner adapters and their registry side effects.

.. automodule:: embodichain.benchmarks.motion_generation.planners
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.planners.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Backend-independent planner adapter contract.

.. automodule:: embodichain.benchmarks.motion_generation.planners.base
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.planners.curobo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cuRobo primary-baseline adapter for an empty collision world.

.. automodule:: embodichain.benchmarks.motion_generation.planners.curobo
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.planners.ik_interpolate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequential IK plus joint interpolation diagnostic adapter.

.. automodule:: embodichain.benchmarks.motion_generation.planners.ik_interpolate
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.planners.nmg_onnx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NMG ONNX benchmark adapter.

.. automodule:: embodichain.benchmarks.motion_generation.planners.nmg_onnx
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.planners.toppra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IK plus TOPPRA time-parameterization diagnostic adapter.

.. automodule:: embodichain.benchmarks.motion_generation.planners.toppra
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Planner and scenario registries used by the generic benchmark runner.

.. automodule:: embodichain.benchmarks.motion_generation.registry
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Render separate planning and skill views with three tables per view.

.. automodule:: embodichain.benchmarks.motion_generation.reporting
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.robots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Built-in robot providers for motion-generation benchmarks.

.. automodule:: embodichain.benchmarks.motion_generation.robots
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.robots.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robot-provider contract for planner benchmark suites.

.. automodule:: embodichain.benchmarks.motion_generation.robots.base
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.robots.dual_arm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dual-arm robot provider for coordinated Atomic Skill benchmarks.

.. automodule:: embodichain.benchmarks.motion_generation.robots.dual_arm
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.robots.franka
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Built-in Franka robot providers.

.. automodule:: embodichain.benchmarks.motion_generation.robots.franka
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.robots.ur5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UR5 robot provider for motion-generation benchmarks.

.. automodule:: embodichain.benchmarks.motion_generation.robots.ur5
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generic lifecycle runner for motion-generation benchmark tracks.

.. automodule:: embodichain.benchmarks.motion_generation.runner
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scenario providers for motion-generation benchmarks.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_case_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic case identity, pose randomization, and goal construction.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_case_utils
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_contact_cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Press, slide, and twist benchmark cases.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_contact_cases
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measured physical stage evaluators and observation-only evidence helpers.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_evaluation
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_grasps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Frozen and case-scoped grasp generators for reproducible skill cases.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_grasps
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_manipulation_cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pickup, held-object motion, and placement benchmark cases.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_manipulation_cases
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_motion_cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Representative end-effector and joint-motion integration cases.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_motion_cases
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Config-driven scene objects for Atomic Skill benchmark cases.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_objects
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmark motion policy, attribution, and shared motion budgets.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_policy
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_replay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Physical replay and observation collection for Atomic Skill benchmarks.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_replay
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.atomic_skill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Physics-backed Atomic Skill scenario orchestration.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.atomic_skill
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scenario provider contract for motion-generation tracks.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.base
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.scenarios.free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic reachable cases for the ``free-space-common`` track.

.. automodule:: embodichain.benchmarks.motion_generation.scenarios.free_space
   :members:
   :undoc-members:


embodichain.benchmarks.motion_generation.video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional Atomic Skill replay recording helpers.

.. automodule:: embodichain.benchmarks.motion_generation.video
   :members:
   :undoc-members:


embodichain.cli.benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~

Unified benchmark CLI.

.. automodule:: embodichain.cli.benchmark
   :members:
   :undoc-members:
