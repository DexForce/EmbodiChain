Planners
========

Planners produce timed trajectories from robot waypoints. Start with
:doc:`MotionGenerator <../motion_generator>` for the shared planning workflow;
use a backend directly when you already have its required waypoint inputs.

Choose a planner
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Planner
     - Use it for
     - Native targets and requirements
   * - :doc:`toppra_planner`
     - Time-parameterizing joint paths under velocity and acceleration limits.
     - Joint waypoints; requires TOPPRA.
   * - :doc:`trapezoidal_planner`
     - Batched joint paths with trapezoidal or jerk-limited Double-S timing.
     - Joint waypoints; Torch or Warp.
   * - :doc:`curobo_planner`
     - Collision-aware planning against an explicit world.
     - Joint or Cartesian goals; requires CUDA and cuRobo V2.
   * - :doc:`neural_planner`
     - Experimental learned end-effector waypoint rollout.
     - Cartesian goals; requires an NMG ONNX policy, validated on Franka Panda.

For joint-only planners, ``MotionGenerator`` can convert Cartesian targets
through IK when ``is_interpolate=True``. Only cuRobo provides collision-aware
planning against the configured world; consult its page for supported geometry
and collision-checking limits.

Sampling is configured per backend. :doc:`trajectory_sample_method` explains
the shared enum; each planner page documents the modes it accepts.

Custom backends
---------------

Subclass ``BasePlanner`` with a matching ``BasePlannerCfg``, declare
``supported_move_types``, and return the timed ``PlanResult`` contract described
in the MotionGenerator guide. Register the pair with
``MotionGenerator.register_planner_type(name, planner_class, planner_cfg_class)``.
Full signatures are in the
:doc:`planner API </api_reference/embodichain/embodichain.lab.sim.motion.planners>`.

.. toctree::
   :maxdepth: 1

   toppra_planner
   trapezoidal_planner
   curobo_planner
   neural_planner
   trajectory_sample_method
