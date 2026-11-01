Franka
======

Official Franka robot configurations live in the
``embodichain_robots.franka`` manufacturer namespace.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 24 25 51

   * - Robot
     - Configuration class
     - Current coverage
   * - :doc:`Franka family <cfg>`
     - ``FrankaCfg``
     - Bare seven-DOF FR3 and original Panda arms; tools use explicit assembly.
   * - :doc:`Franka Panda <panda>`
     - ``FrankaPandaCfg``
     - Compatibility entry selecting the historical FR3 model as a bare arm.

.. toctree::
   :maxdepth: 1
   :hidden:

   panda
   cfg
