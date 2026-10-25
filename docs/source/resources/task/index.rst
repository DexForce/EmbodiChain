Supported Tasks
===============

Task environments combine a simulation scene, robot, and task logic behind a
Gymnasium-compatible interface.

The official task environments are bundled in the ``embodichain`` wheel under
the ``embodichain_tasks`` import package. This section follows the task-first
layout under ``embodichain_tasks/configs/tasks`` so that documentation,
configuration paths, and ``list-task`` output use the same hierarchy.

Find a Task
-----------

Use the catalog commands to inspect the installed task set:

.. code-block:: bash

   embodichain list-task
   embodichain list-task --category manipulation
   embodichain show-task embodichain_tasks:repeated_pick_place

Preview a task with its runnable JSON or YAML configuration:

.. code-block:: bash

   embodichain run-task \
       --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
       --preview

Configuration-backed environments run through ``embodichain run-task``. Tasks
that exist only as learning environments or Python registrations are identified
on their category page. See :doc:`/guides/task_catalog` for catalog metadata and
static gallery export, and :doc:`/guides/run_task` for launch options.
See :doc:`/tutorial/rl` to train an RL policy.

Task hierarchy
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Domain
     - Configuration hierarchy
     - Scope
   * - :doc:`Classic control <classic_control>`
     - ``classic_control/``
     - Compact control and learning environments.
   * - :doc:`Locomotion <locomotion/index>`
     - ``locomotion/velocity/``
     - Velocity-tracking tasks for legged robots.
   * - :doc:`Manipulation <manipulation/index>`
     - ``manipulation/`` and ``manipulation/tableware/``
     - RL, Task Program, and environment-only manipulation tasks.
   * - :doc:`Special <special>`
     - ``special/``
     - Standalone examples and specialized differentiable environments.

The top-level ``id`` in a runnable JSON or YAML configuration is the Gym
environment ID. A reusable ``env.yaml`` component may instead expose only
``environment_id`` and is not runnable by itself. Import-backed tasks register
during package discovery; configuration-defined Task Program tasks register
when their runnable task configuration is loaded.

Create a Task
-------------

Start from the
`task template <https://github.com/DexForce/embodichain_task_template>`_.
Use an ``EmbodiedEnv`` subclass for custom Python behavior, or compose a
configuration-defined Task Program from reusable components. Third-party
task packages are discovered through the ``embodichain.tasks`` entry point.
See :doc:`/overview/gym/env` for environment authoring and
:doc:`/tutorial/task_program` for Task Programs. The
`official task package README <https://github.com/DexForce/EmbodiChain/blob/main/extensions/embodichain_tasks/README.md>`_
describes package layout and registration.

.. toctree::
   :maxdepth: 2
   :hidden:

   classic_control
   locomotion/index
   manipulation/index
   special
