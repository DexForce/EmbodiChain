Robotiq
========

Model-owned assets and any standalone configurations live under
``embodichain_robots.robotiq``. Asset availability does not imply a
standalone robot preset or a grasping capability.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 30 25 45

   * - Model / asset
     - Interface
     - Coverage
   * - :doc:`Shared 2F archive <assets>`
     - ``Asset class``
     - Separate 85 mm and 140 mm source models.
   * - :doc:`2F-85 standalone archive <robotiq_2f85>`
     - ``Asset class``
     - Independent archive/cache from shared 2F models.

.. toctree::
   :maxdepth: 1
   :hidden:

   assets
   robotiq_2f85
