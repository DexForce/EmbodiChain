Supported Tasks
===============

The official task environments are bundled in the ``embodichain`` wheel under
the ``embodichain_tasks`` import package. Installing EmbodiChain registers the
environment IDs below automatically; no second package installation is needed.

Run a task by passing one of its gym configuration files to the unified CLI:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/tableware/pour_water/env.json

Use ``--preview`` to inspect a configured environment without starting a data
generation run. See :doc:`/guides/run_env` for all launch options and
:doc:`/tutorial/data_generation` for the dataset workflow.

Environment catalog
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - Domain
     - Environment ID
     - Example gym config
   * - Classic control
     - ``CartPoleRL``
     - ``embodichain_tasks/configs/tasks/classic_control/cart_pole/env.yaml``
   * - Manipulation
     - ``PushCubeRL``
     - ``embodichain_tasks/configs/tasks/manipulation/push_cube/env.json``
   * - Manipulation
     - ``ExpertProgramRepeatedPickPlace-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json``
   * - Special
     - ``SimpleTask-v1``
     - ``embodichain_tasks/configs/tasks/special/simple_task/env_ur10.json``
   * - Special
     - ``StayStillSave-v1``
     - ``embodichain_tasks/configs/tasks/special/stay_still_save/env_ur10.json``
   * - Tableware
     - ``BlocksRankingRGB-v1``
     - ``embodichain_tasks/configs/tasks/tableware/blocks_ranking_rgb/env.json``
   * - Tableware
     - ``BlocksRankingSize-v1``
     - ``embodichain_tasks/configs/tasks/tableware/blocks_ranking_size/env.json``
   * - Tableware
     - ``MatchObjectContainer-v1``
     - ``embodichain_tasks/configs/tasks/tableware/match_object_container/env.json``
   * - Manipulation
     - ``ExpertProgramOpenDrawer-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/open_drawer/env.json``
   * - Manipulation
     - ``HandOver-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/hand_over/env.json``
   * - Tableware
     - ``PlaceObjectDrawer-v1``
     - ``embodichain_tasks/configs/tasks/tableware/place_object_drawer/env.json``
   * - Tableware
     - ``PourWater-v3``
     - ``embodichain_tasks/configs/tasks/tableware/pour_water/env.json``
   * - Tableware
     - ``ScoopIce-v1``
     - ``embodichain_tasks/configs/tasks/tableware/scoop_ice/env.json``
   * - Tableware
     - ``StackBlocksTwo-v1``
     - ``embodichain_tasks/configs/tasks/tableware/stack_blocks_two/env.json``
   * - Tableware
     - ``StackCups-v1``
     - ``embodichain_tasks/configs/tasks/tableware/stack_cups/env.json``
   * - Tableware
     - ``PourWaterAgent-v3``
     - Uses the Pour Water scene together with
       ``embodichain_tasks/configs/tasks/tableware/pour_water/agent/``.
   * - Tableware
     - ``Rearrangement-v3`` / ``RearrangementAgent-v3``
     - Registered task variants; no standalone gym config is currently shipped.

The value of ``id`` inside a gym config must match a registered environment ID.
When adding a task, update this catalog together with its runnable config.
