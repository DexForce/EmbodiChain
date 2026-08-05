API Reference
=============

This section provides the API-level documentation for EmbodiChain's public Python
modules.

Use this reference when you need:

* module-level overviews and responsibilities,
* public classes, functions, and configuration objects,
* links into specialized subpackages (simulation, gym environments, RL, and utilities).

The pages are organized from high-level package namespaces to concrete submodules.

Core Framework
--------------

The core ``embodichain`` framework is split into six top-level packages:

``data``
    Dataset resolution, asset-download helpers, shared constants, and enums used
    by simulation tasks and training pipelines.

``data_pipeline``
    Online data streaming and recording: a process-safe trajectory engine,
    online datasets/samplers for live-simulation training, and compressed
    depth-sidecar storage for LeRobot datasets.

``lab``
    The simulation laboratory: the ``sim`` core (scene, objects, sensors, IK
    solvers, planners, atomic actions), the ``gym`` environment framework,
    real-device controllers (``devices``), and browser ``visualization``.

``toolkits``
    Standalone asset-preparation and manipulation utilities (parallel-gripper
    grasp sampling, URDF convex decomposition, URDF assembly) usable
    independently of the simulation loop.

``learning``
    Learning systems, currently the ``rl`` subpackage: on-policy RL algorithms
    (PPO/GRPO), rollout buffers, collectors, policy/model builders, and the
    training entry point.

``utils``
    Shared utilities: the ``@configclass`` decorator, logging, math/tensor
    helpers, file/string/device helpers, and high-performance ``warp`` kernels
    for kinematics and image processing.

.. currentmodule:: embodichain

.. autosummary::
   :toctree: embodichain

   data
   data_pipeline
   lab
   toolkits
   learning
   utils
