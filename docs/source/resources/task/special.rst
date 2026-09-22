Special Tasks
=============

Special tasks are standalone examples or specialized environments that do not
belong to the main control, locomotion, or manipulation families.

.. list-table::
   :header-rows: 1
   :widths: 22 26 32 20

   * - Task
     - Environment ID
     - Entry point
     - Capability
   * - Franka Reach APG
     - ``FrankaReachApg-v0``
     - Python registration in
       ``embodichain_tasks.special.franka_reach_apg``; no standard Gym config
     - Differentiable RL / APG
   * - Simple Task
     - ``SimpleTask-v1``
     - ``simple_task/env_ur10.json``
     - Environment only
   * - Stay Still Save
     - ``StayStillSave-v1``
     - ``stay_still_save/env_ur10.json`` or
       ``stay_still_save/env_async_ur10.json``
     - Environment only

Launch a configuration-backed special task in the same way as other task
families:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/special/simple_task/env_ur10.json
