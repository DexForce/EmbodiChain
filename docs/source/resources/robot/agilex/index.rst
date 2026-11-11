AgileX
======

Official AgileX robot configurations live in the
``embodichain_robots.agilex`` manufacturer namespace.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 24 25 51

   * - Robot
     - Configuration class
     - Current coverage
   * - :doc:`Pika Gripper <pika>`
     - Asset class
     - Standalone tool asset for host-owned mounting and control configuration.
   * - :doc:`CobotMagic <cobotmagic>`
     - ``CobotMagicCfg``
     - Fixed-base dual arms and grippers for bimanual tasks.

.. toctree::
   :maxdepth: 1
   :hidden:

   cobotmagic
   pika
   cobotmagic_gripper
