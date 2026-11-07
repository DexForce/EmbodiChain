DH-Robotics
===========

Model-owned assets and any standalone configurations live under
``embodichain_robots.dh_robotics``. Asset availability does not imply a
standalone robot preset or a grasping capability.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 30 25 45

   * - Model / asset
     - Interface
     - Coverage
   * - :doc:`PGC-140-50 <pgc_140_50>`
     - ``Asset classes``
     - Separate base and DexForce-modified archives.
   * - :doc:`PGI-140-80 <pgi_140_80>`
     - ``Asset class``
     - Parallel-jaw asset for host-owned assembly.
   * - :doc:`AG95 <ag95>`
     - ``Asset class``
     - DexForce-modified connector and fingers.

.. toctree::
   :maxdepth: 1
   :hidden:

   pgc_140_50
   pgi_140_80
   ag95
