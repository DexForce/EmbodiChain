embodichain.lab.sim.robots
======================================

.. automodule:: embodichain.lab.sim.robots

Overview
--------

Generic dual-arm composition built from compatible single-arm configurations.
:func:`build_dual_arm_cfg` assembles two single-arm configs into a synchronized
dual-arm robot. Concrete presets are imported from ``embodichain_robots``;
see :doc:`/api_reference/robot_catalogs` for those configuration classes and
:doc:`/guides/robot_catalog` for discovery and variant selection.

.. rubric:: Classes

.. autosummary::

   DualArmRobotCfg

.. rubric:: Functions

.. autosummary::

   build_dual_arm_cfg

.. currentmodule:: embodichain.lab.sim.robots

.. autoclass:: DualArmRobotCfg
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autofunction:: build_dual_arm_cfg
