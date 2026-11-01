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
     - Explicit FR3/Panda-Hand or original-Panda/Robotiq 2F-85 combinations.
   * - :doc:`Franka Panda <panda>`
     - ``FrankaPandaCfg``
     - Compatibility entry preserving the historical FR3-named asset and Panda Hand.

.. toctree::
   :maxdepth: 1
   :hidden:

   panda
   cfg
