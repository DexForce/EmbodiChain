Manipulation Tasks
==================

The manipulation domain combines RL environments, Task Program expert
demonstrations, and environment-only tasks. Tasks directly under
``manipulation/`` are listed here; object and tableware tasks are grouped on a
separate child page.

.. list-table::
   :header-rows: 1
   :widths: 18 30 31 21

   * - Task
     - Environment ID and deployment
     - Embodiment
     - Capability
   * - Hand Over
     - ``HandOver-v1``
     - Dual UR5 with PGI-140-80 grippers
     - Task Program expert demo
   * - Open Drawer
     - ``TaskProgramOpenDrawer-v1`` (UR5, default),
       ``TaskProgramOpenDrawer-Newton-v1`` (UR5, Newton), and
       ``TaskProgramOpenDrawer-Franka-v1`` (Franka, default)
     - UR5 or Franka Panda
     - Task Program expert demo
   * - Push Cube
     - ``PushCubeRL``
     - UR robot
     - RL
   * - Repeated Pick and Place
     - ``TaskProgramRepeatedPickPlace-v1`` and
       ``TaskProgramRepeatedPickPlace-Franka-v1`` (default), plus their
       ``-Newton-v1`` deployments
     - UR5 or Franka Panda
     - Task Program expert demo

Use ``show-task`` before launch when a logical task has several deployments:

.. code-block:: bash

   embodichain show-task embodichain_tasks:repeated_pick_place
   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.newton.yaml

.. toctree::
   :maxdepth: 1

   tableware
