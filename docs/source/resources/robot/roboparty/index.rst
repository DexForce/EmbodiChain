RoboParty
=========

RoboParty configurations and asset definitions live in
``embodichain.extensions.robots.roboparty``. The robot catalog lists three
presets covering seven model configurations.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 26 30 44

   * - Model
     - Catalog ID
     - Variants
   * - :doc:`Roboto Origin <rpo>`
     - ``RoboPartyRPO``
     - 23-joint humanoid
   * - :doc:`RP1 <rp1>`
     - ``RoboPartyRP1``
     - ``28dof``, ``24dof``, ``hand``, ``hand_fixed``
   * - :doc:`RP Hand <rp_hand>`
     - ``RoboPartyRPHand``
     - Standalone ``left`` and ``right`` hands

.. toctree::
   :maxdepth: 1
   :hidden:

   rpo
   rp1
   rp_hand
