LeWM visual dynamics
====================

LeWM predicts action-conditioned image embeddings. Its model uses normalized
image tensors and raw controller actions with frozen training statistics.
The sequence source reads complete HDF5 episode windows, the objective trains
the encoder and predictor jointly, and the CEM planner optimizes bounded
candidate controls against a target-image embedding. Portable exports include
the model configuration, action statistics and caller-provided data contract.

Public package
--------------

.. automodule:: embodichain_models.dynamics.lewm
   :members:
   :imported-members:

Implementation modules
----------------------

.. automodule:: embodichain_models.dynamics.lewm.model
   :members:
   :no-index:

.. automodule:: embodichain_models.dynamics.lewm.data
   :members:
   :no-index:

.. automodule:: embodichain_models.dynamics.lewm.supervised
   :members:
   :no-index:

.. automodule:: embodichain_models.dynamics.lewm.sl_recipe
   :members:
   :no-index:

.. automodule:: embodichain_models.dynamics.lewm.planner
   :members:
   :no-index:

.. automodule:: embodichain_models.dynamics.lewm.checkpoint
   :members:
   :no-index:
