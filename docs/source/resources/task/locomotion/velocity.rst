Velocity-Tracking Tasks
=======================

These tasks live under ``embodichain_tasks/configs/tasks/locomotion/velocity``.
Every task provides ``env.yaml`` for the default physics backend and
``env.newton.yaml`` for Newton, plus a matching PPO trainer configuration.

.. list-table::
   :header-rows: 1
   :widths: 24 34 22 20

   * - Task
     - Environment ID
     - Embodiment
     - Backends
   * - ANYmal C Flat
     - ``ANYmalCFlatRL-v1``
     - ANYmal C
     - Default, Newton
   * - Unitree G1 Flat
     - ``UnitreeG1FlatRL-v1``
     - Unitree G1
     - Default, Newton
   * - Unitree Go1 Flat
     - ``UnitreeGo1FlatRL-v1``
     - Unitree Go1
     - Default, Newton
   * - Unitree Go2 Flat
     - ``UnitreeGo2FlatRL-v1``
     - Unitree Go2
     - Default, Newton
   * - Unitree H1-2 Flat
     - ``UnitreeH1_2FlatRL-v1``
     - Unitree H1-2
     - Default, Newton
   * - MicroDuck Flat
     - ``MicroDuckFlatRL-v1``
     - MicroDuck
     - Default, Newton

Select the backend through the configuration file rather than using the
launcher to switch physics:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/locomotion/velocity/go2_flat/env.newton.yaml
