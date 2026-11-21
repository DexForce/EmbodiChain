:orphan:

Official OGBench single-Cube task
=================================

The task consumes the audited assets and physical contract exported by the LeWM
official-reference example. It executes five-dimensional end-effector and gripper
commands through the DexUni CPU rigid-body solver. The reference tools check
kinematics, composite inertia, controller outputs and physical trajectories.

.. currentmodule:: embodichain.extensions.tasks.manipulation.ogbench_cube

.. autosummary::

   OGBenchCubeCfg
   OGBenchCubeEnv

.. autoclass:: OGBenchCubeCfg
   :members:

.. autoclass:: OGBenchCubeEnv
   :members:
