Franka
======

Official Franka robot configurations live in the
``embodichain_robots.franka`` manufacturer namespace.
``Franka`` is the unified bare-arm catalog entry. ``FrankaPanda`` preserves
the historical FR3 selector; native-hand task deployments select an explicit
embodiment component.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 24 25 51

   * - Robot
     - Configuration class
     - Current coverage
   * - :doc:`Franka bare arms <cfg>`
     - ``FrankaCfg``
     - Bare seven-DOF FR3 and original Panda arms; tools use explicit assembly.
   * - :doc:`FrankaPanda compatibility and native hand <panda>`
     - ``FrankaPandaCfg``
     - Compatibility entry selecting the historical FR3 model as a bare arm.

The model pages distinguish bare-arm construction from the separately assembled
``franka_panda`` native-hand embodiment and the Panda/Robotiq embodiment.

.. toctree::
   :maxdepth: 1
   :hidden:

   panda
   cfg
