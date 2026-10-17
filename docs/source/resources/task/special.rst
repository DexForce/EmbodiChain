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

The simple UR10 end-effector motion example is a Task Program tutorial rather
than an official task catalog entry:

.. code-block:: bash

   python scripts/tutorials/gym/simple_task_program.py --headless --seed 7

The tutorial moves through four Cartesian waypoints and returns near the initial
TCP pose. ``ScoopIce-v1``, ``StayStillSave-v1``, ``StayStillSave3Cam-v1``, and
``SimpleTask-v1`` are retired from the official task catalog.
