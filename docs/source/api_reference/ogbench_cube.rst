:orphan:

Official OGBench single-Cube task
===============================

The task consumes the audited assets and physical contract exported by the LeWM
official-reference example. It executes five-dimensional end-effector and gripper
commands through the Default DexSim backend. The reference tools separately check
kinematics, composite inertia, controller outputs and physical trajectories.

.. currentmodule:: embodichain_tasks.manipulation.ogbench_cube

.. autosummary::

   OGBenchCubeCfg
   OGBenchCubeEnv

.. autoclass:: OGBenchCubeCfg
   :members:

.. autoclass:: OGBenchCubeEnv
   :members:
