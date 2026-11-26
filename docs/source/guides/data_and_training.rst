Data & Training
===============

1. :doc:`Add task and segment instructions <expert_task_language>` if needed,
   then :doc:`collect expert demonstrations </tutorial/data_expansion>`.
2. :doc:`Validate and split the dataset <offline_expert_datasets>`. Use
   :doc:`lerobot_export` when the training environment requires that format.
3. :doc:`Choose a training framework <training_frameworks>` and
   :doc:`evaluate the resulting policy <policy_evaluation>`.

For live training batches during collection, see :doc:`/features/online_data`.

.. toctree::
   :maxdepth: 1

   expert_task_language
   offline_expert_datasets
   lerobot_export
   training_frameworks
   configured_manipulation_rl
   policy_evaluation
