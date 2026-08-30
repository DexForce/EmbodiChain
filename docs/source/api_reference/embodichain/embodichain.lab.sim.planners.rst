embodichain.lab.sim.planners
==========================================

.. automodule:: embodichain.lab.sim.planners

  .. rubric:: Classes

  .. autosummary::
    BasePlannerCfg
    BasePlanner
    ToppraPlannerCfg
    ToppraPlanner
    TrapezoidalPlanOptions
    TrapezoidalPlannerCfg
    TrapezoidalPlanner
    MotionGenCfg
    MotionGenerator
    TrajectorySampleMethod
    MovePart
    MoveType
    PlanResult
    PlanState

.. currentmodule:: embodichain.lab.sim.planners

Base Planner
------------

.. autoclass:: BasePlannerCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: BasePlanner
    :members:
    :inherited-members:
    :show-inheritance:

Toppra Planner
--------------

.. autoclass:: ToppraPlannerCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: ToppraPlanner
    :members:
    :inherited-members:
    :show-inheritance:

Trapezoidal Planner
-------------------

The trapezoidal planner applies either acceleration-limited trapezoidal timing
or jerk-limited Double-S timing to batched, piecewise-linear joint paths. Use
``TrapezoidalPlanOptions.minimum_duration`` to slow a trajectory without
changing its path or violating derivative limits.
For densely interpolated straight paths, set ``stop_at_waypoints=False`` to
remove redundant same-direction interior points while preserving real corners.
``backend="auto"`` selects Warp profile construction and sampling for CUDA
float32 inputs and retains the Torch reference implementation for CPU or
float64. The Warp path builds trapezoidal or Double-S phases in parallel per
batch segment, then evaluates scalar samples and composes all joints.
Both backends avoid a dense sample-by-segment lookup tensor: Torch uses batched
``searchsorted``, while Warp performs one binary lookup per batch sample before
parallel joint composition.
Torch also gathers profile coefficients directly by selected phase, avoiding
full sample-by-phase copies of position, velocity, acceleration, and jerk data.
All-stationary batches use a hold fast path and skip constraint projection,
profile construction, segment lookup, and backend dispatch.

.. autoclass:: TrapezoidalPlanOptions
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: TrapezoidalPlannerCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: TrapezoidalPlanner
    :members:
    :inherited-members:
    :show-inheritance:

Example
~~~~~~~

Select the Double-S profile when jerk continuity is required while retaining
the same ``MotionGenerator`` entry point::

    generator = MotionGenerator(
        MotionGenCfg(
            planner_cfg=TrapezoidalPlannerCfg(robot_uid=robot.uid),
        )
    )
    result = generator.generate(
        [PlanState.from_qpos(start), PlanState.from_qpos(goal)],
        MotionGenOptions(
            plan_opts=TrapezoidalPlanOptions(
                profile="double_s",
                constraints={
                    "velocity": 0.5,
                    "acceleration": 1.0,
                    "jerk": 3.0,
                },
                sample_interval=200,
                stop_at_waypoints=False,
                backend="auto",
            )
        ),
    )

The complete batched simulation tutorial is
``scripts/tutorials/sim/trapezoidal_planner.py``.
Pass ``--backend torch`` or ``--backend warp`` to compare implementations.
For repeatable timing and memory measurements, run
``scripts/benchmark/motion_generation/trapezoidal_planner.py``.

Motion Generator
----------------

.. autoclass:: MotionGenCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: MotionGenerator
    :members:
    :inherited-members:
    :show-inheritance:

Utilities
---------

.. autoclass:: TrajectorySampleMethod
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: MovePart
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: MoveType
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: PlanResult
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: PlanState
    :members:
    :undoc-members:
    :show-inheritance:
