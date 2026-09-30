Classic Control Tasks
=====================

Classic-control tasks provide compact environments for validating learning
algorithms and backend behavior. Their configurations live under
``embodichain_tasks/configs/tasks/classic_control``.

.. list-table::
   :header-rows: 1
   :widths: 18 22 38 22

   * - Task
     - Environment ID
     - Configuration or learning entry
     - Capability
   * - Cart Pole
     - ``CartPoleRL``
     - ``cart_pole/env.json`` and ``cart_pole/env.yaml``
     - RL
   * - Humanoid Run
     - ``HumanoidRun-v1``
     - ``humanoid/env.yaml`` (default) and ``humanoid/env.newton.yaml``
       (Newton)
     - RL
   * - Point Mass
     - ``PointMassRL``
     - ``point_mass/agents/apg.yaml`` and ``point_mass/agents/ppo.yaml``;
       registered as a lightweight learning environment
     - RL

For configuration-backed tasks, pass the selected environment file to
``run-env``. Point Mass is selected by its trainer configuration instead of a
Gym configuration file.

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/classic_control/cart_pole/env.yaml
