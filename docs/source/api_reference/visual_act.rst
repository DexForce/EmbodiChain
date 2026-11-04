:orphan:

Official LeRobot ACT and PushCube
========================================

The ACT integration uses LeRobot 0.6.1 on Python 3.12. EmbodiChain owns
collection, declared action contracts, corpus provenance and simulator evaluation.
LeRobot owns its model, processors, training loop and native checkpoints.
The example invokes the upstream trainer directly; it does not use the shared SL trainer.

Load a ``pretrained_model`` directory with its ``embodichain_act.json`` manifest.
Call ``reset()`` after every environment reset. The adapter returns physical
commands on the requested device and uses the official temporal ensemble.

Historical EmbodiChain ACT ``.pt`` checkpoints require retraining.

.. autosummary::

   embodichain_models.policy.act.ACTInferencePolicy
   embodichain_models.policy.act.build_act_train_config
   embodichain_models.policy.act.dataset_fingerprint
   embodichain_models.policy.act.load_act_policy
   embodichain_models.policy.act.policy_fingerprint
   embodichain_models.policy.act.runtime.ACTInferencePolicy
   embodichain_models.policy.act.runtime.load_act_policy
   embodichain_models.policy.act.runtime.policy_fingerprint
   embodichain_models.policy.act.training.build_act_train_config
   embodichain_models.policy.act.training.dataset_fingerprint

.. autoclass:: embodichain_models.policy.act.ACTInferencePolicy
   :members:

.. autofunction:: embodichain_models.policy.act.build_act_train_config

.. autofunction:: embodichain_models.policy.act.dataset_fingerprint

.. autofunction:: embodichain_models.policy.act.load_act_policy

.. autofunction:: embodichain_models.policy.act.policy_fingerprint
