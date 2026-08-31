Supported Tasks
===============

The official task environments are bundled in the ``embodichain`` wheel under
the ``embodichain_tasks`` import package. Import-backed tasks register during
package discovery; configuration-defined Task Program tasks register when
their gym config is loaded. No second package installation is needed.

Run a task by passing one of its gym configuration files to the unified CLI:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml

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
     - ``TaskProgramRepeatedPickPlace-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml``
   * - Manipulation
     - ``TaskProgramRepeatedPickPlace-Franka-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml``
   * - Special
     - ``SimpleTask-v1``
     - ``embodichain_tasks/configs/tasks/special/simple_task/env_ur10.json``
   * - Special
     - ``StayStillSave-v1``
     - ``embodichain_tasks/configs/tasks/special/stay_still_save/env_ur10.json``
   * - Tableware
     - ``BlocksRankingRGB-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/tableware/blocks_ranking_rgb/env.json``
   * - Tableware
     - ``BlocksRankingSize-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/tableware/blocks_ranking_size/env.json``
   * - Tableware
     - ``MatchObjectContainer-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/tableware/match_object_container/env.json``
   * - Manipulation
     - ``TaskProgramOpenDrawer-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/open_drawer/task.ur5.yaml``
   * - Manipulation
     - ``TaskProgramOpenDrawer-Franka-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/open_drawer/task.franka.yaml``
   * - Manipulation
     - ``HandOver-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/hand_over/task.dual_ur5_dh_pgi_140_80.yaml``
   * - Tableware
     - ``PlaceObjectDrawer-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/tableware/place_object_drawer/env.json``
   * - Tableware
     - ``PourWater-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml``
   * - Tableware
     - ``ScoopIce-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/tableware/scoop_ice/env.json``
   * - Tableware
     - ``StackBlocksTwo-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/env.json``
   * - Tableware
     - ``StackCups-v1``
     - ``embodichain_tasks/configs/tasks/manipulation/tableware/stack_cups/env.json``

The value of ``id`` inside a conventional gym config must match a registered
environment ID. A supported configuration-defined Task Program deployment
declares ``environment.component``,
``task_program.{program,integration,execution_policy}``, and
``embodiment.component``; loading it registers its free ``id`` against the
common ``EmbodiedEnv``. A pure ``env.yaml`` component has ``environment_id``
but no runnable ``id``, so task discovery does not list it. Discovery is based
on the top-level ``id`` schema rather than an ``env*`` filename prefix.
When adding a task, update this catalog together with its runnable config.
