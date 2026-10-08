embodichain.lab.sim.robots
======================================

.. automodule:: embodichain.lab.sim.robots

Overview
--------

Robot-specific configuration presets ready to drop into a simulation scene.
Each preset is a :class:`~embodichain.lab.sim.cfg.RobotCfg` subclass that
fixes the URDF, control parts, drive properties, and IK solver configuration
for a particular robot. The package also exposes
:func:`build_dual_arm_cfg`, a helper for assembling two single-arm configs into
a synchronized dual-arm robot.

.. rubric:: Classes

.. autosummary::

   DexforceW1Cfg
   CobotMagicCfg
   TianjiMarvinCfg
   FrankaPandaCfg
   URRobotCfg
   DualArmRobotCfg

.. rubric:: Functions

.. autosummary::

   build_dual_arm_cfg

.. currentmodule:: embodichain.lab.sim.robots

.. autoclass:: DexforceW1Cfg
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: CobotMagicCfg
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: TianjiMarvinCfg
   :members: from_dict, build_pk_serial_chain, with_gripper

See :doc:`/resources/robot/tianji_marvin` for variant selection, control parts,
the FK/IK frame contract, and runnable examples. The same configuration is
available through its implementation module:

.. autosummary::

   embodichain.lab.sim.robots.tianji_marvin.TianjiMarvinCfg

.. autoclass:: FrankaPandaCfg
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: URRobotCfg
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: DualArmRobotCfg
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autofunction:: build_dual_arm_cfg
