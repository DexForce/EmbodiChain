Supported Tasks
===============

The official task environments are bundled in the ``embodichain`` wheel under
the ``embodichain_tasks`` import package. Installing EmbodiChain registers the
environment IDs below automatically; no second package installation is needed.

Run a task by passing one of its gym configuration files to the unified CLI:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/gym/pour_water/gym_config.json

Use ``--preview`` to inspect a configured environment without starting a data
generation run. See :doc:`/guides/run_env` for all launch options and
:doc:`/tutorial/data_generation` for the dataset workflow.

Environment catalog
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - Category
     - Environment ID
     - Example gym config
   * - Reinforcement learning
     - ``CartPoleRL``
     - ``embodichain_tasks/configs/agents/rl/basic/cart_pole/gym_config.yaml``
   * - Reinforcement learning
     - ``PushCubeRL``
     - ``embodichain_tasks/configs/agents/rl/push_cube/gym_config.json``
   * - Multi-segment
     - ``MultiSegmentsCubePickPlace-v1``
     - ``embodichain_tasks/configs/gym/multi_segments/cube_pick_place.json``
   * - Special
     - ``SimpleTask-v1``
     - ``embodichain_tasks/configs/gym/special/simple_task_ur10.json``
   * - Special
     - ``StayStillSave-v1``
     - ``embodichain_tasks/configs/gym/special/stay_still_save_ur10.json``
   * - Tableware
     - ``BlocksRankingRGB-v1``
     - ``embodichain_tasks/configs/gym/blocks_ranking_rgb/cobot_magic_3cam.json``
   * - Tableware
     - ``BlocksRankingSize-v1``
     - ``embodichain_tasks/configs/gym/blocks_ranking_size/cobot_magic_3cam.json``
   * - Tableware
     - ``MatchObjectContainer-v1``
     - ``embodichain_tasks/configs/gym/match_object_container/cobot_magic_3cam.json``
   * - Tableware
     - ``OpenDrawer-v1``
     - ``embodichain_tasks/configs/gym/open_drawer/cobot_magic_3cam.json``
   * - Tableware
     - ``PlaceObjectDrawer-v1``
     - ``embodichain_tasks/configs/gym/place_object_drawer/cobot_magic_3cam.json``
   * - Tableware
     - ``PourWater-v3``
     - ``embodichain_tasks/configs/gym/pour_water/gym_config.json``
   * - Tableware
     - ``ScoopIce-v1``
     - ``embodichain_tasks/configs/gym/scoop_ice/gym_config.json``
   * - Tableware
     - ``StackBlocksTwo-v1``
     - ``embodichain_tasks/configs/gym/stack_blocks_two/cobot_magic_3cam.json``
   * - Tableware
     - ``StackCups-v1``
     - ``embodichain_tasks/configs/gym/stack_cups/cobot_magic_3cam.json``
   * - Agent variant
     - ``PourWaterAgent-v3``
     - Uses the Pour Water scene together with
       ``embodichain_tasks/configs/gym/agent/pour_water_agent/``.
   * - Agent variant
     - ``Rearrangement-v3`` / ``RearrangementAgent-v3``
     - Registered task variants; no standalone gym config is currently shipped.

The value of ``id`` inside a gym config must match a registered environment ID.
When adding a task, update this catalog together with its runnable config.
