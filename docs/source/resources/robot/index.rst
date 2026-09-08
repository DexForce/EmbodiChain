Supported Robots
================

EmbodiChain provides configuration classes for the robot families and composed
layouts below. Each page documents construction, control parts, kinematics, and
the currently supported variants.

.. list-table::
   :header-rows: 1
   :widths: 24 25 51

   * - Robot
     - Configuration class
     - Current coverage
   * - :doc:`Franka Panda <franka_panda>`
     - ``FrankaPandaCfg``
     - Panda arm and gripper with numerical FK/IK.
   * - :doc:`UR family <ur_robot>`
     - ``URRobotCfg``
     - UR3, UR3e, UR5, UR5e, UR10, and UR10e with analytic IK.
   * - :doc:`Dexforce W1 <dexforce_w1>`
     - ``DexforceW1Cfg``
     - Dual 7-DOF arms, versioned assets, and configurable hands/grippers.
   * - :doc:`CobotMagic <cobotmagic>`
     - ``CobotMagicCfg``
     - Fixed-base dual arms and grippers for bimanual tasks.
   * - :doc:`Dual-arm composition <dual_arm>`
     - ``DualArmRobotCfg``
     - Registry-based composition of compatible single-arm configs in multiple layouts.

.. toctree::
   :maxdepth: 1
   :hidden:

   franka_panda
   ur_robot
   dexforce_w1
   cobotmagic
   dual_arm

To add another model, follow :doc:`/guides/add_robot` and include its page in
this catalog and toctree.
