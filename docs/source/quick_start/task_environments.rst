Task Environments
=================

Task environments combine a simulation scene, robot, and task logic behind a
Gymnasium-compatible interface. Official classic-control, locomotion,
manipulation, and special tasks are included in the main ``embodichain``
installation through the ``embodichain_tasks`` package.

Find a Task
-----------

.. code-block:: bash

   embodichain list-task
   embodichain list-task --category manipulation
   embodichain show-task embodichain_tasks:repeated_pick_place

See :doc:`/resources/task/index` for supported tasks and their configuration
paths. Preview a task with its runnable JSON or YAML configuration:

.. code-block:: bash

   embodichain run-task \
       --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
       --preview

See :doc:`/guides/run_env` for rollout, recording, and replay options, or
:doc:`/tutorial/rl` to train an RL policy.

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
