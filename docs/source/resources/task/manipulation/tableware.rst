Tableware Manipulation Tasks
============================

Tableware tasks live under
``embodichain_tasks/configs/tasks/manipulation/tableware``. They use task-local
configuration files and remain grouped beneath the manipulation domain in the
catalog.

.. list-table::
   :header-rows: 1
   :widths: 24 31 25 20

   * - Task
     - Environment ID
     - Embodiment
     - Capability
   * - Blocks Ranking RGB
     - ``BlocksRankingRGB-v1``
     - CobotMagic
     - Environment only
   * - Blocks Ranking Size
     - ``BlocksRankingSize-v1``
     - CobotMagic
     - Environment only
   * - Match Object Container
     - ``MatchObjectContainer-v1``
     - CobotMagic
     - Environment only
   * - Place Object Drawer
     - ``PlaceObjectDrawer-v1``
     - CobotMagic
     - Environment only
   * - Pour Water
     - ``PourWater-v1``
     - CobotMagic
     - Task Program expert demo
   * - Scoop Ice
     - ``ScoopIce-v1``
     - Dexforce W1
     - Environment only
   * - Stack Two Blocks
     - ``StackBlocksTwo-v1``
     - CobotMagic
     - Environment only
   * - Stack Cups
     - ``StackCups-v1``
     - CobotMagic
     - Environment only

``Environment only`` means the task exposes a runnable environment but the
catalog currently finds neither a supported expert-demo route nor an RL trainer
configuration. It does not describe task quality or validation status.

For example, launch Pour Water through its Task Program deployment:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml
