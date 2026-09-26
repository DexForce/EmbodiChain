Tianji Marvin robot configuration
================================

The preset selects a complete dual-arm URDF with or without parallel-jaw
grippers. It owns the named control parts, arm FK/IK frames, and simulation
drive defaults. See :doc:`../resources/robot/tianji_marvin` for variant
selection and runnable examples.

.. autosummary::

   embodichain.lab.sim.robots.TianjiMarvinCfg

.. autoclass:: embodichain.lab.sim.robots.TianjiMarvinCfg
   :members: from_dict, build_pk_serial_chain, with_gripper

Implementation import path
--------------------------

The same configuration is available through the single robot module.

.. autosummary::

   embodichain.lab.sim.robots.tianji_marvin.TianjiMarvinCfg
