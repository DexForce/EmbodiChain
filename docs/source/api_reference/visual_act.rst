:orphan:

Visual action chunking and PushCube
==================================

These interfaces train a conditional variational action-chunking policy from
current RGB/proprioception and recorded future actions. The episode collector
and simulator evaluation are owned by the ``examples/learning/act`` example.

.. autosummary::

   embodichain.data_pipeline.datasets.lerobot_images.LeRobotImageReader
   embodichain_models.policy.act.ACTCfg
   embodichain_models.policy.act.ACTPolicy
   embodichain_models.policy.act.ACTObjective
   embodichain_models.policy.act.ACTTemporalEnsembler
   embodichain_models.policy.act.ACTSequenceSource
   embodichain_models.policy.act.build_act_trainer
   embodichain_models.policy.act.model.ACTCfg
   embodichain_models.policy.act.model.ACTPolicy
   embodichain_models.policy.act.model.ACTObjective
   embodichain_models.policy.act.data.ACTSequenceSource
   embodichain_models.policy.act.rollout.ACTTemporalEnsembler
   embodichain_models.policy.act.sl_recipe.build_act_trainer
   embodichain_tasks.manipulation.push_cube.VisualPushCubeEnv

.. autoclass:: embodichain.data_pipeline.datasets.lerobot_images.LeRobotImageReader
   :members:

.. autoclass:: embodichain_models.policy.act.ACTCfg
   :members:

.. autoclass:: embodichain_models.policy.act.ACTPolicy
   :members:

.. autoclass:: embodichain_models.policy.act.ACTObjective
   :members:

.. autoclass:: embodichain_models.policy.act.ACTTemporalEnsembler
   :members:

.. autoclass:: embodichain_models.policy.act.ACTSequenceSource
   :members:

.. autofunction:: embodichain_models.policy.act.build_act_trainer

.. autoclass:: embodichain_tasks.manipulation.push_cube.VisualPushCubeEnv
   :members:
