Universal Robots
================

Official Universal Robots configurations live in the
``embodichain_robots.universal_robots`` manufacturer namespace. The six
supported UR models share one family page and configuration class.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 24 25 51

   * - Robot
     - Configuration class
     - Current coverage
   * - :doc:`UR family <ur_robot>`
     - ``URRobotCfg``
     - UR3, UR3e, UR5, UR5e, UR10, and UR10e with analytic IK.

.. toctree::
   :maxdepth: 1
   :hidden:

   ur_robot
